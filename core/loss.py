import torch
import torch.nn.functional as F
from torch import nn
from fvcore.nn import sigmoid_focal_loss_jit
import torchvision.ops as ops
from .util import box_ops
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou
from typing import Dict, List, Optional, Tuple, Union
from detectron2.detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.detectron2.utils.memory import retry_if_cuda_oom
from detectron2.detectron2.modeling.sampling import subsample_labels
from detectron2.detectron2.utils.events import get_event_storage
from detectron2.detectron2.layers import Conv2d, ShapeSpec, cat
from detectron2.detectron2.modeling.box_regression import Box2BoxTransform
from core.util.utils import get_centerness,dense_box_regression_loss
import copy
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches

class SetCriterionDynamicK(nn.Module):
    """ This class computes the training loss.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, cfg, num_classes, matcher, weight_dict, eos_coef, losses, anchor_matcher, hidden_dim=256):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.start_count = 0
        self.start_iter = cfg.MODEL.CHANGE_START

        self.focal_loss_alpha = cfg.MODEL.ALPHA
        self.focal_loss_gamma = cfg.MODEL.GAMMA
        self.disentangled = cfg.MODEL.DISENTANGLED
        self.hidden_dim = hidden_dim
        self.max_prob = 0.9
        self.anchor_matcher = anchor_matcher
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.batch_size_per_image = 256
        self.loss_weight = {'loss_rpn_cls': 0.1}

        # ---------------- UFDM (paper-aligned) ----------------
        self.ufdm_dim = hidden_dim
        self.ufdm_gamma = getattr(cfg.MODEL, "UFDM_GAMMA", 2.0)  # s = r^gamma
        self.ufdm_tau_min = getattr(cfg.MODEL, "UFDM_TAU_MIN", 0.05)  # clip weight
        self.ufdm_ae_weight = getattr(cfg.MODEL, "UFDM_AE_WEIGHT", 1.0)  # AE reconstruction loss weight
        self.ufdm_bg_samples = getattr(cfg.MODEL, "UFDM_BG_SAMPLES", 256)
        self.ufdm_fit_min_n = getattr(cfg.MODEL, "UFDM_FIT_MIN_N", 32)

        # AE: simple MLP autoencoder
        h = max(self.ufdm_dim // 2, 64)
        z = max(self.ufdm_dim // 4, 32)
        self.ufdm_enc = nn.Sequential(nn.Linear(self.ufdm_dim, h), nn.ReLU(inplace=True), nn.Linear(h, z))
        self.ufdm_dec = nn.Sequential(nn.Linear(z, h), nn.ReLU(inplace=True), nn.Linear(h, self.ufdm_dim))

        # Weibull params (EMA-updated)
        self.register_buffer("ufdm_k_fg", torch.tensor(2.0))
        self.register_buffer("ufdm_l_fg", torch.tensor(1.0))
        self.register_buffer("ufdm_k_bg", torch.tensor(2.0))
        self.register_buffer("ufdm_l_bg", torch.tensor(1.0))
        self.ufdm_param_m = getattr(cfg.MODEL, "UFDM_PARAM_MOMENTUM", 0.9)

    def loss_labels(self, outputs, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        if self.disentangled == 0:
            src_logits = outputs['pred_logits']
        else:
            assert 'pred_objectness' in outputs
            src_prob = torch.softmax(outputs['pred_logits'], dim=-1) * outputs['pred_objectness']#计算分类得分
            src_logits = torch.log(src_prob / (1 - src_prob))
        batch_size = len(targets)

        if self.cfg.TEST.MASK == 2:
            seen_logits = list(range(0, self.cfg.TEST.PREV_INTRODUCED_CLS))
            masked_logit = src_logits.clone()
            masked_logit[..., seen_logits] = -10e10
            src_logits = masked_logit

        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        src_logits_list = []
        target_classes_o_list = []
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0] # valid query:匹配到GT的预测框
            gt_multi_idx = indices[batch_idx][1] # gt_multi_idx : 匹配到的GT的索引
            if len(gt_multi_idx) == 0:
                continue
            bz_src_logits = src_logits[batch_idx]
            target_classes_o = targets[batch_idx]["labels"] # 真实框标签
            target_classes[batch_idx, valid_query] = target_classes_o[gt_multi_idx]#将匹配到的GT的类别赋值给target_classes

            src_logits_list.append(bz_src_logits[valid_query]) # 取出匹配到GT的预测框的分类得分
            target_classes_o_list.append(target_classes_o[gt_multi_idx]) # 预测框对应的GT的类别

        num_boxes = torch.cat(target_classes_o_list).shape[0] if len(target_classes_o_list) != 0 else 1

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], self.num_classes + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout,
                                            device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)#在第2维上进行one-hot编码，预测框对应的GT的类别为1，其余为0
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        src_logits = src_logits.flatten(0, 1)
        target_classes_onehot = target_classes_onehot.flatten(0, 1)
        cls_loss = sigmoid_focal_loss_jit(src_logits, target_classes_onehot, alpha=self.focal_loss_alpha,
                                              gamma=self.focal_loss_gamma, reduction="none")
        if torch.isnan(cls_loss).any():
            print("cls_loss:" + str(cls_loss))
            print(" src_logits:"+str(src_logits))
            print(" target_classes_onehot:"+str(target_classes_onehot))
            print("target_classes:"+str(target_classes))
            cls_loss = torch.nan_to_num(cls_loss, nan=0)
        losses = {'loss_ce': torch.sum(cls_loss) / num_boxes}

        return losses

    def loss_nc_labels(self, outputs, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        if self.disentangled == 0:
            src_logits = outputs['pred_logits']
        else:
            assert 'pred_objectness' in outputs
            src_prob = torch.softmax(outputs['pred_logits'], dim=-1) * outputs['pred_objectness']
            src_logits = torch.log(src_prob / (1 - src_prob))
        batch_size = len(targets)

        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        src_logits_list = []
        target_classes_o_list = []
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0]
            gt_multi_idx = indices[batch_idx][1]
            if len(gt_multi_idx) == 0:
                continue
            bz_src_logits = src_logits[batch_idx]
            target_classes_o = targets[batch_idx]["labels"]
            target_classes[batch_idx, valid_query] = target_classes_o[gt_multi_idx]

            src_logits_list.append(bz_src_logits[valid_query]) # 10个未知类别的预测框的分类得分
            target_classes_o_list.append(target_classes_o[gt_multi_idx])

        num_boxes = torch.cat(target_classes_o_list).shape[0] if len(target_classes_o_list) != 0 else 1

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], self.num_classes + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout,
                                            device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        src_logits = src_logits.flatten(0, 1)
        target_classes_onehot = target_classes_onehot.flatten(0, 1)
        cls_loss = sigmoid_focal_loss_jit(src_logits, target_classes_onehot, alpha=self.focal_loss_alpha,
                                          gamma=self.focal_loss_gamma, reduction="none")
        if torch.isnan(cls_loss).any():
            print("cls_loss:" + str(cls_loss))
            print(" src_logits:"+str(src_logits))
            print(" target_classes_onehot:"+str(target_classes_onehot))
            print("target_classes:"+str(target_classes))
            cls_loss = torch.nan_to_num(cls_loss, nan=0)
        assert not torch.isnan(cls_loss).any()
        losses = {'loss_nc_ce': torch.sum(cls_loss) / num_boxes}

        return losses

    def loss_boxes(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
           src_boxes:[x_min, y_min, x_max, y_max]
        """
        assert 'pred_boxes' in outputs
        # idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes']

        batch_size = len(targets)
        pred_box_list = []
        pred_norm_box_list = []
        tgt_box_list = []
        tgt_box_xyxy_list = []
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0]
            gt_multi_idx = indices[batch_idx][1]
            if len(gt_multi_idx) == 0:
                continue
            bz_image_whwh = targets[batch_idx]['image_size_xyxy']
            bz_src_boxes = src_boxes[batch_idx]
            bz_target_boxes = targets[batch_idx]["boxes"]  # normalized (cx, cy, w, h)
            bz_target_boxes_xyxy = targets[batch_idx]["boxes_xyxy"]  # absolute (x1, y1, x2, y2)
            pred_box_list.append(bz_src_boxes[valid_query])
            pred_norm_box_list.append(bz_src_boxes[valid_query] / bz_image_whwh)  # normalize (x1, y1, x2, y2)
            tgt_box_list.append(bz_target_boxes[gt_multi_idx])
            tgt_box_xyxy_list.append(bz_target_boxes_xyxy[gt_multi_idx])

        if len(pred_box_list) != 0:
            src_boxes = torch.cat(pred_box_list)
            src_boxes_norm = torch.cat(pred_norm_box_list)  # normalized (x1, y1, x2, y2)
            target_boxes = torch.cat(tgt_box_list)
            target_boxes_abs_xyxy = torch.cat(tgt_box_xyxy_list)
            num_boxes = src_boxes.shape[0]

            losses = {}
            # require normalized (x1, y1, x2, y2)
            loss_bbox = F.l1_loss(src_boxes_norm, box_cxcywh_to_xyxy(target_boxes), reduction='none')
            losses['loss_bbox'] = loss_bbox.sum() / num_boxes

            # loss_giou = giou_loss(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes))
            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(src_boxes, target_boxes_abs_xyxy))
            losses['loss_giou'] = loss_giou.sum() / num_boxes
        else:
            losses = {'loss_bbox': outputs['pred_boxes'].sum() * 0,
                      'loss_giou': outputs['pred_boxes'].sum() * 0}

        return losses

    def loss_decorr(self, outputs, targets, indices):
        assert self.disentangled != 0
        cls_scores = outputs['pred_logits'].softmax(-1).flatten(0, 1).detach()
        obj_score = outputs['pred_objectness'].reshape(-1)

        cls_mean, obj_mean = cls_scores.mean(dim=0), obj_score.mean()
        cov = ((cls_scores - cls_mean) * (obj_score - obj_mean)[:, np.newaxis]).sum(dim=0)
        var = ((cls_scores - cls_mean) ** 2).sum(dim=0) * ((obj_score - obj_mean) ** 2).sum()
        loss_decorr = (cov ** 2 / var).mean()
        if torch.isnan(loss_decorr).any():
            loss_decorr = torch.nan_to_num(loss_decorr, nan=0)
        return {'loss_decorr': loss_decorr}

    def loss_obj_likelihood(self, outputs, targets, indices, num_boxes, num_pseudo_boxes, lvl, owod_targets,
                            owod_indices):
        assert "pred_objectness" in outputs
        temperature = 1 / self.hidden_dim
        indices = [(tensor1.cpu(), tensor2.cpu()) for tensor1, tensor2 in indices]
        idx = self._get_src_permutation_idx(indices)
        owod_idx = self._get_src_permutation_idx(owod_indices)
        # unmatch indices
        queries = torch.arange(outputs['pred_objectness'].shape[1])
        unmatched_indices = []
        for i in range(len(indices)):  # 通过对比 queries 和正样本索引，找到未匹配的索引
            combined = torch.cat(
                (queries, self._get_src_single_permutation_idx(indices[i], i)[-1]))  ## need to fix the indexing
            uniques, counts = combined.unique(return_counts=True)
            unmatched_indices.append(uniques[counts == 1])

        # positive samples
        pred_obj = outputs["pred_objectness"][idx]

        # negative samples
        region_boxes_list = [t['boxes'][self._filter_invalid(t['boxes'])]
                             for t in owod_targets]  # 从真实标签中提取的区域框
        neg_boxes_list = [t[i] for t, i in zip(outputs['pred_boxes'], unmatched_indices)]  # 未匹配的候选框
        neg_obj_list = [t[i].flatten() for t, i in zip(outputs['pred_objectness'], unmatched_indices)]  # 未匹配的候选框的预测概率
        neg_mask_list = []  # 通过计算未匹配框与区域框的交并比（IoU），筛选出真正的负样本。
        for i, (region_boxes, neg_boxes) in enumerate(zip(region_boxes_list, neg_boxes_list)):
            img_w, img_h = targets[i]['image_size_xyxy'][0], targets[i]['image_size_xyxy'][1]
            region_boxes = box_ops.box_cxcywh_to_xyxy(region_boxes)
            region_boxes = region_boxes * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(
                region_boxes.device)
            # neg_boxes = box_ops.box_cxcywh_to_xyxy(neg_boxes)
            # neg_boxes = neg_boxes * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(neg_boxes.device)
            if len(region_boxes) != 0:
                neg_ious, _ = box_ops.jaccard(neg_boxes, region_boxes).max(-1)  # 计算未匹配框与区域框的交并比
                neg_mask = neg_ious < 0.1  # 将未匹配的候选框与伪标签的IOU小于阈值的作为负样本
            else:
                neg_mask = torch.ones((len(neg_boxes)), dtype=bool)
            neg_mask_list.append(neg_mask)
        neg_obj = torch.cat([b[m] for b, m in zip(neg_obj_list, neg_mask_list)], dim=0)  # 最终的负样本对象概率。
        #负样本采样
        num_neg_samples = min(len(neg_obj), num_boxes * 3)  # 负样本数量不超过正样本的3倍
        neg_obj_sampled = neg_obj[torch.randperm(len(neg_obj))[:num_neg_samples]]

        # positve loss
        # pred_obj_prob = torch.exp(-temperature * pred_obj).flatten()
        pred_obj_prob = pred_obj.flatten()
        pred_obj_prob = torch.clamp(pred_obj_prob, max=self.max_prob)
        loss_pos_obj_ll = (- torch.log(pred_obj_prob)).sum() / num_boxes
        # negative loss
        # neg_obj_prob = torch.exp(-temperature * neg_obj)
        neg_obj_prob = neg_obj
        # neg_obj_prob = torch.clamp(neg_obj_prob, min=self.min_prob)
        epsilon = 1e-7  # 平滑项
        if len(neg_obj_prob) != 0:
            loss_neg_obj_ll = (- torch.log(1 - neg_obj_sampled + epsilon)).sum() / len(neg_obj_sampled)
        else:
            loss_neg_obj_ll = loss_pos_obj_ll * 0.

        # loss_neg_obj_ll = loss_pos_obj_ll * 0. # w/o sns
        # 伪标签样本的损失计算
        if owod_targets != None:
            eval_pseudo_pred_obj = outputs["pred_objectness"][owod_idx].flatten()  # 伪标注对象的概率
            # eval_pseudo_obj_prob = torch.exp(-temperature * eval_pseudo_pred_obj)
            obj_weights = eval_pseudo_pred_obj.detach()
            valid_mask = (obj_weights > 0.7).cpu()  # 筛选出概率大于阈值的伪标签样本
            owod_idx = [owod_idx[0][valid_mask], owod_idx[1][valid_mask]]
            pseudo_pred_obj = outputs["pred_objectness"][owod_idx].flatten()
            # pseudo_pred_obj_prob = torch.exp(-temperature * pseudo_pred_obj)  # 提取出伪标注对象对应的最终预测值
            pseudo_pred_obj_prob = torch.clamp(pseudo_pred_obj, max=self.max_prob)
            loss_pseudo_obj_ll = (- torch.log(pseudo_pred_obj_prob)).sum() / num_pseudo_boxes
        else:
            loss_pseudo_obj_ll = loss_pos_obj_ll * 0.

        return {'loss_pos_obj_ll': loss_pos_obj_ll, 'loss_neg_obj_ll': loss_neg_obj_ll,
                    'loss_pseudo_obj_ll': loss_pseudo_obj_ll}
        # if self.lr_decline_p == 0:
        #     return {'loss_pos_obj_ll': loss_pos_obj_ll, 'loss_neg_obj_ll': loss_neg_obj_ll,
        #             'loss_pseudo_obj_ll': loss_pseudo_obj_ll}
        # else:
        #     sum_geo = self.sum_geometric_sequence(self.lr_decline_p)
        #     lr_decay = (6 / sum_geo) * self.lr_decline_p ** (5 - lvl)
        #     return {'loss_pos_obj_ll': loss_pos_obj_ll * lr_decay, 'loss_neg_obj_ll': loss_neg_obj_ll * lr_decay,
        #             'loss_pseudo_obj_ll': loss_pseudo_obj_ll * lr_decay}

    def loss_ufdm(self, outputs, targets, indices, owod_targets, owod_indices):
        if ("pred_features" not in outputs) or (owod_targets is None) or (len(owod_indices) == 0):
            return {"loss_ufdm": outputs["pred_boxes"].sum() * 0.0}

        feats = outputs["pred_features"]  # (bs, Q, d)
        bs, Q, d = feats.shape
        device = feats.device

        # -------- collect fg (known matched), bg (unmatched), unk (unknown matched) features --------
        fg_list, bg_list, unk_list = [], [], []
        for b in range(bs):
            # known matched queries
            sel_k = indices[b][0]
            if sel_k.dtype == torch.bool:
                qk = torch.nonzero(sel_k, as_tuple=False).squeeze(1)
            else:
                qk = sel_k

            # unknown matched queries
            sel_u = owod_indices[b][0]
            if sel_u.dtype == torch.bool:
                qu = torch.nonzero(sel_u, as_tuple=False).squeeze(1)
            else:
                qu = sel_u

            if qk.numel() > 0:
                fg_list.append(feats[b, qk])

            if qu.numel() > 0:
                unk_list.append(feats[b, qu])

            # background = queries not in (known matched ∪ unknown matched)
            mask_bg = torch.ones((Q,), dtype=torch.bool, device=device)
            if qk.numel() > 0:
                mask_bg[qk] = False
            if qu.numel() > 0:
                mask_bg[qu] = False
            qb = torch.nonzero(mask_bg, as_tuple=False).squeeze(1)

            if qb.numel() > 0:
                # sample a fixed number for stability
                m = min(qb.numel(), self.ufdm_bg_samples)
                qb = qb[torch.randperm(qb.numel(), device=device)[:m]]
                bg_list.append(feats[b, qb])

        if (len(unk_list) == 0) or (len(fg_list) == 0) or (len(bg_list) == 0):
            return {"loss_ufdm": outputs["pred_boxes"].sum() * 0.0}

        F_fg = torch.cat(fg_list, dim=0)  # (Nfg, d)
        F_bg = torch.cat(bg_list, dim=0)  # (Nbg, d)
        F_uk = torch.cat(unk_list, dim=0)  # (Nuk, d)

        # -------- AE forward & reconstruction error --------
        def ae_recon(x):
            z = self.ufdm_enc(x)
            xh = self.ufdm_dec(z)
            return xh

        # normalize feature like paper’s “object-level features” typically stabilized
        F_fg_n = F.normalize(F_fg, dim=-1)
        F_bg_n = F.normalize(F_bg, dim=-1)
        F_uk_n = F.normalize(F_uk, dim=-1)

        R_fg = ae_recon(F_fg_n)
        R_bg = ae_recon(F_bg_n)
        R_uk = ae_recon(F_uk_n)

        # reconstruction error: L2 norm per sample
        e_fg = torch.norm(F_fg_n - R_fg, dim=-1)  # (Nfg,)
        e_bg = torch.norm(F_bg_n - R_bg, dim=-1)  # (Nbg,)
        e_uk = torch.norm(F_uk_n - R_uk, dim=-1)  # (Nuk,)

        # AE loss (train AE)
        loss_ae = (
                F.mse_loss(R_fg, F_fg_n, reduction="mean") +
                F.mse_loss(R_bg, F_bg_n, reduction="mean") +
                F.mse_loss(R_uk, F_uk_n, reduction="mean")
        )

        # -------- fit/update Weibull params (EMA) --------
        with torch.no_grad():
            self._update_weibull_params(e_fg, e_bg)

        # -------- compute unknown soft weights from fg/bg pdf ratio --------
        p_fg = self._weibull_pdf(e_uk, self.ufdm_k_fg, self.ufdm_l_fg)
        p_bg = self._weibull_pdf(e_uk, self.ufdm_k_bg, self.ufdm_l_bg)
        r = p_fg / (p_fg + p_bg + 1e-6)  # (Nuk,)
        s = torch.clamp(r ** self.ufdm_gamma, min=self.ufdm_tau_min, max=1.0)

        # -------- weighted unknown classification loss (paper: use soft label/weight to reweight pseudo unknown supervision) --------
        # build logits for the matched unknown queries
        if self.disentangled == 0:
            logits = outputs["pred_logits"]  # (bs,Q,C)
        else:
            src_prob = torch.softmax(outputs["pred_logits"], dim=-1) * outputs["pred_objectness"]
            logits = torch.log(src_prob / (1 - src_prob + 1e-6))

        # gather unknown matched logits and targets
        logit_list = []
        tgt_list = []
        for b in range(bs):
            sel_u = owod_indices[b][0]
            gt_i = owod_indices[b][1]
            if sel_u.dtype == torch.bool:
                qu = torch.nonzero(sel_u, as_tuple=False).squeeze(1)
            else:
                qu = sel_u
            if (qu.numel() == 0) or (gt_i.numel() == 0):
                continue
            logit_list.append(logits[b, qu])  # (nu, C)

            # labels from unknown targets
            y = owod_targets[b]["labels"][gt_i]  # usually all 80
            tgt_list.append(y)

        if len(logit_list) == 0:
            loss_nc = logits.sum() * 0.0
        else:
            L = torch.cat(logit_list, dim=0)  # (Nuk, C)
            y = torch.cat(tgt_list, dim=0)  # (Nuk,)
            C = L.shape[-1]
            y_onehot = torch.zeros((L.shape[0], C), device=device, dtype=L.dtype)
            y_onehot.scatter_(1, y.unsqueeze(1), 1.0)

            # focal loss per-sample
            per = sigmoid_focal_loss_jit(
                L, y_onehot,
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="none"
            ).sum(dim=-1)  # (Nuk,)

            # apply UFDM soft weights
            loss_nc = (per * s).mean()

        loss_total = loss_nc + self.ufdm_ae_weight * loss_ae
        return {"loss_ufdm": loss_total}

    def _weibull_pdf(self, e: torch.Tensor, k: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        # e>0
        eps = 1e-6
        e = torch.clamp(e, min=eps)
        k = torch.clamp(k, min=eps)
        lam = torch.clamp(lam, min=eps)
        x = e / lam
        return (k / lam) * torch.pow(x, k - 1.0) * torch.exp(-torch.pow(x, k))

    @torch.no_grad()
    def _fit_weibull_linear(self, e_cpu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Linear-regression fit on Weibull plot:
          y = log(-log(1-F)) = k*log(e) - k*log(lam)
        Return (k, lam)
        """
        e = e_cpu.detach().float().cpu()
        e = e[e > 1e-8]
        n = e.numel()
        if n < self.ufdm_fit_min_n:
            return None, None
        e, _ = torch.sort(e)
        # median rank
        i = torch.arange(1, n + 1, dtype=torch.float32)
        F = (i - 0.3) / (n + 0.4)
        F = torch.clamp(F, 1e-6, 1 - 1e-6)
        x = torch.log(e)
        y = torch.log(-torch.log(1 - F))
        # least squares: y = a*x + b
        x_mean = x.mean()
        y_mean = y.mean()
        a = ((x - x_mean) * (y - y_mean)).sum() / (((x - x_mean) ** 2).sum() + 1e-6)
        b = y_mean - a * x_mean
        k = torch.clamp(a, min=0.3, max=10.0)
        lam = torch.exp(-b / (k + 1e-6))
        lam = torch.clamp(lam, min=1e-3, max=1e3)
        return k.to(self.ufdm_k_fg.device), lam.to(self.ufdm_l_fg.device)

    @torch.no_grad()
    def _update_weibull_params(self, e_fg: torch.Tensor, e_bg: torch.Tensor):
        k_fg, l_fg = self._fit_weibull_linear(e_fg)
        k_bg, l_bg = self._fit_weibull_linear(e_bg)
        if (k_fg is not None) and (l_fg is not None):
            self.ufdm_k_fg = self.ufdm_param_m * self.ufdm_k_fg + (1 - self.ufdm_param_m) * k_fg
            self.ufdm_l_fg = self.ufdm_param_m * self.ufdm_l_fg + (1 - self.ufdm_param_m) * l_fg
        if (k_bg is not None) and (l_bg is not None):
            self.ufdm_k_bg = self.ufdm_param_m * self.ufdm_k_bg + (1 - self.ufdm_param_m) * k_bg
            self.ufdm_l_bg = self.ufdm_param_m * self.ufdm_l_bg + (1 - self.ufdm_param_m) * l_bg
    def _get_src_single_permutation_idx(self, indices, index):
        ## Only need the src query index selection from this function for attention feature selection
        batch_idx = [torch.full_like(src, i) for i, src in enumerate(indices)][0]
        src_idx = indices[0]
        return batch_idx, src_idx
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, num_pseudo_boxes, lvl,  owod_targets, owod_indices, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'nc_labels': self.loss_nc_labels,
            'decorr': self.loss_decorr,
            # 'obj_likelihood': self.loss_obj_likelihood
            'ufdm': self.loss_ufdm,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        if loss == 'obj_likelihood':
            return loss_map[loss](outputs, targets, indices, num_boxes, num_pseudo_boxes, lvl, owod_targets, owod_indices, **kwargs)
        elif loss == 'ufdm':
            return loss_map[loss](outputs, targets, indices, owod_targets, owod_indices)
        return loss_map[loss](outputs, targets, indices)
    def _filter_invalid(self, boxes):
        return (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
    def _subsample_labels(self, label):
        """
        Randomly sample a subset of positive and negative examples, and overwrite
        the label vector to the ignore value (-1) for all elements that are not
        included in the sample.

        Args:
            labels (Tensor): a vector of -1, 0, 1. Will be modified in-place and returned.
        """
        pos_idx, neg_idx = subsample_labels(
            label, 256, 0.5, 0
        )
        # Fill with the ignore label (-1), then set positive and negative labels
        label.fill_(-1)
        label.scatter_(0, pos_idx, 1)
        label.scatter_(0, neg_idx, 0)
        return label
    @torch.jit.unused
    @torch.no_grad()
    def label_and_sample_anchors(
            self, anchors: List[Boxes], gt_instances: List[Instances],
    ):
        """
        Args:
            anchors (list[Boxes]): anchors for each feature map.
            gt_instances: the ground-truth instances for each image.

        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across all feature maps R = sum(Hi * Wi * A).
                Label values are in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative
                class; 1 = positive class.
            list[Tensor]:
                i-th element is a Rx4 tensor. The values are the matched gt boxes for each
                anchor. Values are undefined for those anchors not labeled as 1.
        """
        anchors = Boxes.cat(anchors)
        anchors_list = [Boxes(anchor) for anchor in anchors]

        gt_boxes = [x.gt_boxes for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]
        gt_classes = [x.gt_classes for x in gt_instances]
        soft_labels = [x.gt_classes for x in gt_instances]
        if gt_instances[0].has("soft_labels"):
            soft_labels = [x.soft_labels for x in gt_instances]

        unk_idxs = []
        for gt_classes_i in gt_classes:
            unk_idxs_i = []
            for idx, gt_class in enumerate(gt_classes_i):
                if gt_class == 80:
                    unk_idxs_i.append(idx)
            unk_idxs.append(unk_idxs_i)

        gt_labels = []
        matched_gt_boxes = []
        matched_gt_classes = []
        matched_soft_labels = []
        matched_idx_list = []

        for image_size_i, gt_boxes_i, gt_classes_i, soft_labels_i, anchors in zip(image_sizes, gt_boxes, \
                                                                         gt_classes, soft_labels,anchors_list):
            """
            image_size_i: (h, w) for the i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            """
            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)
            # Matching is memory-expensive and may result in CPU tensors. But the result is small
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            del match_quality_matrix

            # if self.anchor_boundary_thresh >= 0:
            #     # Discard anchors that go out of the boundaries of the image
            #     # NOTE: This is legacy functionality that is turned off by default in Detectron2
            #     anchors_inside_image = anchors.inside_box(image_size_i, self.anchor_boundary_thresh)
            #     gt_labels_i[~anchors_inside_image] = -1

            gt_labels_i = self._subsample_labels(gt_labels_i)

            matched_gt_classes_i = gt_classes_i[matched_idxs]
            matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor

            if gt_instances[0].has("soft_labels"):
                matched_soft_labels_i = soft_labels_i[matched_idxs]

            gt_labels.append(gt_labels_i)  # N,AHW
            matched_gt_boxes.append(matched_gt_boxes_i)
            matched_gt_classes.append(matched_gt_classes_i)
            matched_idx_list.append(matched_idxs)
            if gt_instances[0].has("soft_labels"):
                matched_soft_labels.append(matched_soft_labels_i)
        return gt_labels, matched_gt_boxes, matched_gt_classes, matched_idx_list, unk_idxs, \
            matched_soft_labels

    @torch.jit.unused
    def losses_objetness(
            self,
            anchors: List[Boxes],
            pred_objectness_logits: List[torch.Tensor],
            pred_anchor_deltas: List[torch.Tensor],
            gt_labels: List[torch.Tensor],
            gt_boxes: List[torch.Tensor],
            soft_labels: List[torch.Tensor],
            gt_classes: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.

        Args:
            anchors (list[Boxes or RotatedBoxes]): anchors for each feature map, each
                has shape (Hi*Wi*A, B), where B is box dimension (4 or 5).
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, Hi*Wi*A) representing
                the predicted objectness logits for all anchors.
            gt_labels (list[Tensor]): Output of :meth:`label_and_sample_anchors`.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, Hi*Wi*A, 4 or 5) representing the predicted "deltas" used to transform anchors
                to proposals.
            gt_boxes (list[Tensor]): Output of :meth:`label_and_sample_anchors`.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """
        num_images = len(gt_labels)
        storage = get_event_storage()

        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))
        if (len(soft_labels) == 0):
            pos_mask = gt_labels == 1
        else:
            soft_labels = torch.stack(soft_labels)
            pos_mask = (gt_labels == 1) & (soft_labels == 1.0)

        # localization_loss = dense_box_regression_loss(
        #     anchors,
        #     self.box2box_transform,
        #     pred_anchor_deltas,
        #     gt_boxes,
        #     pos_mask,
        #     box_reg_loss_type='smooth_l1',
        #     smooth_l1_beta=0.0,
        #     reduction="sum"
        # )

        if (len(soft_labels) == 0):
            valid_mask = gt_labels >= 0
            objectness_loss = F.binary_cross_entropy_with_logits(
                cat(pred_objectness_logits, dim=1)[valid_mask].float(),
                gt_labels[valid_mask].float(),
                reduction="sum",
            )
        else:
            # soft_labels = torch.stack(soft_labels)
            valid_mask = gt_labels >= 0
            soft_labels[gt_labels == 0] = 1
            objectness_loss = F.binary_cross_entropy_with_logits(
                pred_objectness_logits.flatten(1)[valid_mask].float(),
                gt_labels[valid_mask].float(),
                reduction="none",
            )
            objectness_loss = (soft_labels[valid_mask] * objectness_loss).sum()

        normalizer = self.batch_size_per_image * num_images
        losses = {}

        # losses["loss_rpn_loc"] = localization_loss / normalizer
        losses["loss_rpn_cls"] = objectness_loss / normalizer

        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        return losses
    def forward(self, outputs, targets,x_boxes, gt_instances):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        self.start_count += 1
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # gt_labels, gt_boxes, gt_classes, idx_list, unk_idxs, soft_labels = \
        #     self.label_and_sample_anchors(x_boxes, gt_instances)

        # Retrieve the matching between the outputs of the last layer and the targets
        # indices, _, ow_indices, unknown_targets = self.matcher(outputs_without_aux, targets)
        indices, matched_ids, ow_indices, ow_matched_ids, unknown_targets = self.matcher(outputs_without_aux, targets)
        new_indices = []
        new_ow_indices = []
        for i in range(len(indices)):
            new_indices.append((matched_ids[i],indices[i][1]))
        for i in range(len(ow_indices)):
            new_ow_indices.append((ow_matched_ids[i],ow_indices[i][1]))

        num_boxes = sum((t["labels"] != 80).sum().item() for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device).item()
        num_boxes = int(num_boxes)
        num_pseudo_boxes = sum((t["labels"] == 80).sum().item() for t in targets)
        num_pseudo_boxes = torch.as_tensor([num_pseudo_boxes], dtype=torch.float, device=next(iter(outputs.values())).device).item()
        num_pseudo_boxes = int(num_pseudo_boxes)
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if loss == 'nc_labels':
                if self.start_count > self.start_iter:
                    losses.update(self.get_loss(loss, outputs, unknown_targets, ow_indices, num_boxes, num_pseudo_boxes, 5,  unknown_targets, new_ow_indices))
            elif loss == 'ufdm':
                # UFDM：必须用 unknown_targets + ow_indices（不能用 new_ow_indices）
                if self.start_count > self.start_iter:
                    losses.update(self.get_loss(loss, outputs, targets, indices,
                                                num_boxes, num_pseudo_boxes, 5, unknown_targets, ow_indices))
            # elif loss == 'objectness':
            #     losses.update(self.losses_objetness(x_boxes, outputs['pred_objectness'], outputs['pred_boxes'], gt_labels, gt_boxes, soft_labels, gt_classes))
            # elif loss == 'obj_likelihood':
            #     losses.update(self.get_loss(loss, outputs, targets, new_indices, num_boxes, num_pseudo_boxes, 5,  unknown_targets, new_ow_indices))
            else:
                losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, num_pseudo_boxes, 5,  unknown_targets, ow_indices))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices, matched_ids, ow_indices, ow_matched_ids, unknown_targets = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'nc_labels':
                        if self.start_count > self.start_iter:
                            l_dict = self.get_loss(loss, aux_outputs, unknown_targets, ow_indices, num_boxes, num_pseudo_boxes, 5,  unknown_targets, new_ow_indices)
                            l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                            losses.update(l_dict)
                    # elif loss == 'objectness':
                    #     l_dict = self.losses_objetness(x_boxes, aux_outputs['pred_objectness'], aux_outputs['pred_boxes'], gt_labels,
                    #                           gt_boxes, soft_labels, gt_classes)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)
                    elif loss == 'ufdm':
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, num_pseudo_boxes, 5, unknown_targets, ow_indices)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)
                    # elif loss == 'obj_likelihood':
                    #     l_dict = self.get_loss(loss, aux_outputs, targets, new_indices, num_boxes, num_pseudo_boxes, 5,
                    #                                 unknown_targets, new_ow_indices)
                    #     l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    #     losses.update(l_dict)
                    else:
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, num_pseudo_boxes, 5,  unknown_targets, new_ow_indices)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

        return losses

from scipy.optimize import linear_sum_assignment
class HungarianMatcherDynamicK(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-k (dynamic) matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cfg, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.ota_k = cfg.MODEL.OTA_K
        self.forward_k = cfg.MODEL.FORWARD_K
        self.cfg = cfg
        self.focal_loss_alpha = cfg.MODEL.ALPHA
        self.focal_loss_gamma = cfg.MODEL.GAMMA
        self.disentangled = cfg.MODEL.DISENTANGLED
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        """ simOTA for detr"""
        known_instances, unknown_instances = self.update_instance(targets)
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            # We flatten to compute the cost matrices in a batch
            if self.disentangled == 0:
                out_prob = outputs["pred_logits"].sigmoid()  # [batch_size, num_queries, num_classes]
            else:
                out_prob = torch.softmax(outputs['pred_logits'], dim=-1) * outputs['pred_objectness']
            out_bbox = outputs["pred_boxes"]  # [batch_size,  num_queries, 4]

            indices = []
            matched_ids = []
            unknown_labels = []
            ow_indices = []
            ow_matched_ids = []
            assert bs == len(targets)
            for batch_idx in range(bs):
                bz_boxes = out_bbox[batch_idx]  # [num_proposals, 4]
                bz_out_prob = out_prob[batch_idx]
                bz_tgt_ids = known_instances[batch_idx]["labels"]
                num_insts = len(bz_tgt_ids) #实例数量
                if num_insts == 0:  # empty object in key frame
                    non_valid = torch.zeros(bz_out_prob.shape[0]).to(bz_out_prob) > 0
                    indices_batchi = (non_valid, torch.arange(0, 0).to(bz_out_prob))
                    matched_qidx = torch.arange(0, 0).to(bz_out_prob)
                    indices.append(indices_batchi)
                    matched_ids.append(matched_qidx)
                else:
                    bz_gtboxs = known_instances[batch_idx]['boxes']  # [num_gt, 4] normalized (cx, xy, w, h) 归一化的真实框
                    bz_gtboxs_abs_xyxy = known_instances[batch_idx]['boxes_xyxy']#真实框的绝对坐标
                    fg_mask, is_in_boxes_and_center = self.get_in_boxes_info( # fg_mask:可能是前景的候选框   is_in_boxes_and_center:是否在候选框内且中心点在真实框内
                        box_xyxy_to_cxcywh(bz_boxes),  # absolute (cx, cy, w, h)
                        box_xyxy_to_cxcywh(bz_gtboxs_abs_xyxy),  # absolute (cx, cy, w, h)
                        expanded_strides=32
                    )

                    pair_wise_ious = ops.box_iou(bz_boxes, bz_gtboxs_abs_xyxy) # 计算候选框与 GT 框之间的 IoU

                    # Compute the classification cost.  基于 Focal Loss 计算每个候选框与 GT 之间的分类损失
                    alpha = self.focal_loss_alpha
                    gamma = self.focal_loss_gamma
                    neg_cost_class = (1 - alpha) * (bz_out_prob ** gamma) * (-(1 - bz_out_prob + 1e-8).log())
                    pos_cost_class = alpha * ((1 - bz_out_prob) ** gamma) * (-(bz_out_prob + 1e-8).log())
                    cost_class = pos_cost_class[:, bz_tgt_ids] - neg_cost_class[:, bz_tgt_ids]

                    # Compute the L1 cost between boxes
                    # image_size_out = torch.cat([v["image_size_xyxy"].unsqueeze(0) for v in targets])
                    # image_size_out = image_size_out.unsqueeze(1).repeat(1, num_queries, 1).flatten(0, 1)
                    # image_size_tgt = torch.cat([v["image_size_xyxy_tgt"] for v in targets])

                    bz_image_size_out = known_instances[batch_idx]['image_size_xyxy']
                    bz_image_size_tgt = known_instances[batch_idx]['image_size_xyxy_tgt']

                    bz_out_bbox_ = bz_boxes / bz_image_size_out  # normalize (x1, y1, x2, y2)
                    bz_tgt_bbox_ = bz_gtboxs_abs_xyxy / bz_image_size_tgt  # normalize (x1, y1, x2, y2)
                    cost_bbox = torch.cdist(bz_out_bbox_, bz_tgt_bbox_, p=1)

                    cost_giou = -generalized_box_iou(bz_boxes, bz_gtboxs_abs_xyxy)

                    # Final cost matrix
                    cost = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou + 100.0 * (
                        ~is_in_boxes_and_center) # 对于不在候选框内且中心点不在候选框内的候选框，增加一个很大的损失
                    # cost = (cost_class + 3.0 * cost_giou + 100.0 * (~is_in_boxes_and_center))  # [num_query,num_gt]
                    cost[~fg_mask] = cost[~fg_mask] + 10000.0

                    # if bz_gtboxs.shape[0]>0:  indices_batchi: 哪些预测框被匹配到了真实框的掩码  matched_qidx: 最终匹配的预测框索引
                    indices_batchi, matched_qidx = self.dynamic_k_matching(cost, pair_wise_ious, bz_gtboxs.shape[0])

                    indices.append(indices_batchi)
                    matched_ids.append(matched_qidx)

                if self.cfg.MODEL.NC:

                    bz_tgt_ids_unknown = unknown_instances[batch_idx]["labels"]
                    bz_gtboxs_unknown = unknown_instances[batch_idx]['boxes']  # [num_gt, 4] normalized (cx, xy, w, h) 归一化的真实框
                    bz_gtboxs_abs_xyxy_unknown = unknown_instances[batch_idx]['boxes_xyxy']#真实框的绝对坐标

                    fg_mask_unknown, is_in_boxes_and_center_unknown = self.get_in_boxes_info( # fg_mask:可能是前景的候选框   is_in_boxes_and_center:是否在候选框内且中心点在真实框内
                        box_xyxy_to_cxcywh(bz_boxes),  # absolute (cx, cy, w, h)
                        box_xyxy_to_cxcywh(bz_gtboxs_abs_xyxy_unknown),  # absolute (cx, cy, w, h)
                        expanded_strides=32
                    )

                    pair_wise_ious_unknown = ops.box_iou(bz_boxes, bz_gtboxs_abs_xyxy_unknown) # 计算候选框与 GT 框之间的 IoU

                    # Compute the classification cost.  基于 Focal Loss 计算每个候选框与 GT 之间的分类损失
                    alpha = self.focal_loss_alpha
                    gamma = self.focal_loss_gamma
                    neg_cost_class = (1 - alpha) * (bz_out_prob ** gamma) * (-(1 - bz_out_prob + 1e-8).log())
                    pos_cost_class = alpha * ((1 - bz_out_prob) ** gamma) * (-(bz_out_prob + 1e-8).log())
                    cost_class_unknown = pos_cost_class[:, bz_tgt_ids_unknown] - neg_cost_class[:, bz_tgt_ids_unknown]

                    bz_image_size_out_unknown = unknown_instances[batch_idx]['image_size_xyxy']
                    bz_image_size_tgt_unknown = unknown_instances[batch_idx]['image_size_xyxy_tgt']

                    bz_out_bbox_unknown = bz_boxes / bz_image_size_out_unknown  # normalize (x1, y1, x2, y2)
                    bz_tgt_bbox_unknown = bz_gtboxs_abs_xyxy_unknown / bz_image_size_tgt_unknown  # normalize (x1, y1, x2, y2)
                    cost_bbox_unknown = torch.cdist(bz_out_bbox_unknown, bz_tgt_bbox_unknown, p=1)

                    cost_giou_unknown = -generalized_box_iou(bz_boxes, bz_gtboxs_abs_xyxy_unknown)

                    # # Final cost matrix
                    cost_unknown = self.cost_bbox * cost_bbox_unknown + self.cost_class * cost_class_unknown + self.cost_giou * cost_giou_unknown + 100.0 * (
                        ~is_in_boxes_and_center_unknown) # 对于不在候选框内且中心点不在候选框内的候选框，增加一个很大的损失
                    # cost = (cost_class + 3.0 * cost_giou + 100.0 * (~is_in_boxes_and_center))  # [num_query,num_gt]
                    cost_unknown[~fg_mask_unknown] = cost_unknown[~fg_mask_unknown] + 10000.0

                    # if bz_gtboxs.shape[0]>0:  indices_batchi: 哪些预测框被匹配到了真实框的掩码  matched_qidx: 最终匹配的预测框索引
                    indices_batchi, matched_qidx = self.dynamic_k_matching(cost_unknown, pair_wise_ious_unknown, bz_gtboxs_unknown.shape[0])

                    ow_indices.append(indices_batchi)
                    ow_matched_ids.append(matched_qidx)

        return indices, matched_ids, ow_indices, ow_matched_ids, unknown_instances

    def update_instance(self, targets):
        known_instances, sam_instances = [], []
        for instance_item in targets:
            image_size_xyxy = instance_item.pop('image_size_xyxy')

            labels = instance_item['labels']
            mask = labels == 80  # 获取labels是否等于80的布尔掩码

            dict1 = {key: value[mask] for key, value in instance_item.items()}  # labels等于80的
            dict2 = {key: value[~mask] for key, value in instance_item.items()}  # labels不等于80的
            dict1['image_size_xyxy'] = image_size_xyxy
            dict2['image_size_xyxy'] = image_size_xyxy
            instance_item['image_size_xyxy'] = image_size_xyxy
            sam_instances.append(dict1)
            known_instances.append(dict2)
            # gt_classes = instance_item.gt_classes
            # known_idx = (gt_classes != 80)
            # known_instances.append(instance_item[known_idx])
            # sam_instances.append(instance_item[~known_idx])
        return known_instances, sam_instances

    def get_in_boxes_info(self, boxes, target_gts, expanded_strides):
        xy_target_gts = box_cxcywh_to_xyxy(target_gts)  # (x1, y1, x2, y2)

        anchor_center_x = boxes[:, 0].unsqueeze(1)
        anchor_center_y = boxes[:, 1].unsqueeze(1)

        # whether the center of each anchor is inside a gt box
        b_l = anchor_center_x > xy_target_gts[:, 0].unsqueeze(0)
        b_r = anchor_center_x < xy_target_gts[:, 2].unsqueeze(0)
        b_t = anchor_center_y > xy_target_gts[:, 1].unsqueeze(0)
        b_b = anchor_center_y < xy_target_gts[:, 3].unsqueeze(0)
        # (b_l.long()+b_r.long()+b_t.long()+b_b.long())==4 [300,num_gt] ,
        is_in_boxes = ((b_l.long() + b_r.long() + b_t.long() + b_b.long()) == 4)
        is_in_boxes_all = is_in_boxes.sum(1) > 0  # [num_query]  >0 表示该候选框至少有一个真实框包含在内
        # in fixed center
        center_radius = 2.5
        # Modified to self-adapted sampling --- the center size depends on the size of the gt boxes
        # https://github.com/dulucas/UVO_Challenge/blob/main/Track1/detection/mmdet/core/bbox/assigners/rpn_sim_ota_assigner.py#L212
        b_l = anchor_center_x > (
                    target_gts[:, 0] - (center_radius * (xy_target_gts[:, 2] - xy_target_gts[:, 0]))).unsqueeze(0)
        b_r = anchor_center_x < (
                    target_gts[:, 0] + (center_radius * (xy_target_gts[:, 2] - xy_target_gts[:, 0]))).unsqueeze(0)
        b_t = anchor_center_y > (
                    target_gts[:, 1] - (center_radius * (xy_target_gts[:, 3] - xy_target_gts[:, 1]))).unsqueeze(0)
        b_b = anchor_center_y < (
                    target_gts[:, 1] + (center_radius * (xy_target_gts[:, 3] - xy_target_gts[:, 1]))).unsqueeze(0)

        is_in_centers = ((b_l.long() + b_r.long() + b_t.long() + b_b.long()) == 4)
        is_in_centers_all = is_in_centers.sum(1) > 0

        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all # 判断锚框是否在 GT 边界框或中心区域内。（fg_mask）
        is_in_boxes_and_center = (is_in_boxes & is_in_centers) # 判断锚框是否同时在 GT 边界框和中心区域内。

        return is_in_boxes_anchor, is_in_boxes_and_center
    def dynamic_k_matching(self, cost, pair_wise_ious, num_gt):
        matching_matrix = torch.zeros_like(cost)  # [300,num_gt]
        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = self.ota_k

        # Take the sum of the predicted value and the top 10 iou of gt with the largest iou as dynamic_k
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1.0

        # del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(1)

        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[anchor_matching_gt > 1], dim=1)
            matching_matrix[anchor_matching_gt > 1] *= 0
            matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1

        iter = 0
        while (matching_matrix.sum(0) == 0).any():
            iter += 1
            if iter > 50:
                print("陷入死循环啦！！！！！！！！！！！")
                print("cost的shape：" + str(cost.shape))
                print("torch.sum(cost > 1000000):" + torch.sum(cost > 1000000))
            previous_matching_matrix = matching_matrix.clone()
            num_zero_gt = (matching_matrix.sum(0) == 0).sum()
            matched_query_id = matching_matrix.sum(1) > 0
            cost[matched_query_id] += 100000.0
            unmatch_id = torch.nonzero(matching_matrix.sum(0) == 0, as_tuple=False).squeeze(1)
            for gt_idx in unmatch_id:
                pos_idx = torch.argmin(cost[:, gt_idx])
                matching_matrix[:, gt_idx][pos_idx] = 1.0
            if (matching_matrix.sum(1) > 1).sum() > 0:  # If a query matches more than one gt
                _, cost_argmin = torch.min(cost[anchor_matching_gt > 1],
                                           dim=1)  # find gt for these queries with minimal cost
                matching_matrix[anchor_matching_gt > 1] *= 0  # reset mapping relationship
                matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1  # keep gt with minimal cost
            if torch.equal(previous_matching_matrix, matching_matrix):
                print("Warning: No progress in matching. Possible deadlock.")
                break
        assert not (matching_matrix.sum(0) == 0).any()
        selected_query = matching_matrix.sum(1) > 0
        if selected_query.sum() > 0:
            gt_indices = matching_matrix[selected_query].max(1)[1]  # 获取匹配的 GT 索引
        else:
            # 如果没有选中的候选框，则处理这种情况
            gt_indices = torch.tensor([], dtype=torch.long, device=matching_matrix.device)  # 返回空的张量
            print('gt_indices  is  empty!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        assert selected_query.sum() == len(gt_indices)

        cost[matching_matrix == 0] = cost[matching_matrix == 0] + float('inf')
        matched_query_id = torch.min(cost, dim=0)[1]

        return (selected_query, gt_indices), matched_query_id

    # def dynamic_k_matching(self,cost, pair_wise_ious, num_gt, ota_k):
    #     """
    #     动态 K 匹配算法
    #
    #     参数:
    #         cost (torch.Tensor): 大小为 [num_candidates, num_gt] 的代价矩阵。
    #         pair_wise_ious (torch.Tensor): 大小为 [num_candidates, num_gt] 的 IoU 矩阵。
    #         num_gt (int): GT（真实框）的数量。
    #         ota_k (int): 计算动态 K 时考虑的候选框数量。
    #
    #     返回:
    #         selected_query (torch.Tensor): 布尔张量，表示被选中的预测框。
    #         gt_indices (torch.Tensor): 每个被选中预测框对应的 GT 索引。
    #     """
    #     if num_gt == 0:
    #         print("no num_gt!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #
    #     matching_matrix = torch.zeros_like(cost)  # 初始化匹配矩阵，大小为 [num_candidates, num_gt]
    #
    #     # 步骤 1：为每个 GT 计算动态 K
    #     topk_ious, _ = torch.topk(pair_wise_ious, ota_k, dim=0)  # 计算每个 GT 与前 ota_k 个 IoU 的最大值
    #     dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)  # 动态 K 值（至少为 1）
    #
    #     # 步骤 2：根据代价矩阵进行初步匹配
    #     for gt_idx in range(num_gt):
    #         _, pos_idx = torch.topk(cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)  # 选出代价最小的 K 个候选框
    #         matching_matrix[:, gt_idx][pos_idx] = 1.0  # 标记为匹配
    #
    #     # 步骤 3：处理冲突（一个预测框被多个 GT 匹配的情况）
    #     anchor_matching_gt = matching_matrix.sum(1)  # 统计每个预测框匹配的 GT 数量
    #     if (anchor_matching_gt > 1).any():  # 如果存在冲突
    #         conflict_indices = torch.nonzero(anchor_matching_gt > 1, as_tuple=False).squeeze(1)  # 获取冲突预测框的索引
    #         for idx in conflict_indices:
    #             conflicting_gts = torch.nonzero(matching_matrix[idx], as_tuple=False).squeeze(1)  # 获取冲突的 GT 索引
    #             min_cost_idx = torch.argmin(cost[idx, conflicting_gts])  # 找到代价最小的 GT
    #             selected_gt = conflicting_gts[min_cost_idx]
    #             matching_matrix[idx] = 0  # 清除当前预测框的所有匹配
    #             matching_matrix[idx, selected_gt] = 1  # 仅保留代价最小的匹配
    #             cost[idx] += float('inf')
    #     cost[matching_matrix.sum(1) > 0] += float('inf')  # 将已分配的候选框代价增大
    #
    #
    #     # 步骤 4：处理未匹配的 GT（确保每个 GT 至少有一个预测框匹配）
    #     while (matching_matrix.sum(0) == 0).any():  # 如果存在未匹配的 GT
    #         unmatched_gts = torch.nonzero(matching_matrix.sum(0) == 0, as_tuple=False).squeeze(1)  # 获取未匹配的 GT 索引
    #         for gt_idx in unmatched_gts:
    #             pos_idx = torch.argmin(cost[:, gt_idx])  # 找到代价最小的候选框
    #             matching_matrix[:, gt_idx][pos_idx] = 1.0  # 将其匹配到 GT
    #
    #         # 再次处理新引入的冲突
    #         anchor_matching_gt = matching_matrix.sum(1)
    #         if (anchor_matching_gt > 1).any():
    #             conflict_indices = torch.nonzero(anchor_matching_gt > 1, as_tuple=False).squeeze(1)
    #             for idx in conflict_indices:
    #                 conflicting_gts = torch.nonzero(matching_matrix[idx], as_tuple=False).squeeze(1)
    #                 min_cost_idx = torch.argmin(cost[idx, conflicting_gts])
    #                 selected_gt = conflicting_gts[min_cost_idx]
    #                 matching_matrix[idx] = 0
    #                 matching_matrix[idx, selected_gt] = 1
    #                 cost[idx] += float('inf')
    #
    #     # 步骤 5：生成最终匹配结果
    #     # assert not (matching_matrix.sum(0) == 0).any(), "每个 GT 必须至少匹配一个预测框。"
    #
    #     selected_query = matching_matrix.sum(1) > 0  # 布尔张量，表示被选中的预测框
    #     # 确保 selected_query 中有选中的候选框
    #     if selected_query.sum() > 0:
    #         gt_indices = matching_matrix[selected_query].max(1)[1]  # 获取匹配的 GT 索引
    #     else:
    #         # 如果没有选中的候选框，则处理这种情况
    #         gt_indices = torch.tensor([], dtype=torch.long, device=matching_matrix.device)  # 返回空的张量
    #         print('gt_indices  is  empty!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    #
    #     assert selected_query.sum() == len(gt_indices)
    #
    #     cost[matching_matrix == 0] = cost[matching_matrix == 0] + float('inf')
    #     # matched_query_id = torch.min(cost, dim=0)[1]
    #     matched_query_id = torch.nonzero(selected_query, as_tuple=False).squeeze(1)
    #
    #     return (selected_query, gt_indices), matched_query_id

    def relaxed_matching(self, cost, pair_wise_ious, relax_factor=0.5, max_matches_per_gt=3):
        """
        Relaxed Matching with IoU and Cost Relaxation.

        Args:
            cost (torch.Tensor): Cost matrix [num_queries, num_gt].
            pair_wise_ious (torch.Tensor): IoU matrix [num_queries, num_gt].
            relax_factor (float): Minimum IoU threshold for relaxed matching.
            max_matches_per_gt (int): Maximum number of relaxed matches per GT.

        Returns:
            selected_query (torch.Tensor): Boolean mask for matched queries [num_queries].
            gt_indices (torch.Tensor): Matched GT indices for selected queries.
        """
        # 初始化匹配矩阵
        matching_matrix = torch.zeros_like(cost)  # [num_queries, num_gt]

        # 基于 relax_factor 的松弛 IoU 匹配规则
        for gt_idx in range(cost.shape[1]):  # 遍历每个 GT
            ious = pair_wise_ious[:, gt_idx]  # 当前 GT 对应的 IoU 候选值
            valid_candidates = ious > relax_factor  # 满足松弛条件的候选框
            if valid_candidates.sum() == 0:
                continue

            # 对满足松弛条件的候选框，选择 top-k 最低代价的框
            valid_cost = cost[valid_candidates, gt_idx]
            topk_indices = torch.topk(valid_cost, k=min(max_matches_per_gt, valid_candidates.sum()),
                                      largest=False).indices

            # 获取全局的候选框索引
            global_indices = torch.nonzero(valid_candidates, as_tuple=False).squeeze(1)
            selected_indices = global_indices[topk_indices]

            # 更新匹配矩阵
            matching_matrix[selected_indices, gt_idx] = 1.0

        # 避免重复匹配：同一个 query 匹配多个 GT 时保留最低 cost 的 GT
        if (matching_matrix.sum(1) > 1).any():
            _, min_cost_indices = torch.min(cost[matching_matrix.sum(1) > 1], dim=1)
            matching_matrix[matching_matrix.sum(1) > 1] = 0  # 重置重复匹配
            matching_matrix[matching_matrix.sum(1) > 1, min_cost_indices] = 1.0

        # 构造返回值
        selected_query = matching_matrix.sum(1) > 0  # 被选中的候选框
        gt_indices = matching_matrix[selected_query].max(1)[1]  # 对应的 GT 索引
        return selected_query, gt_indices

    def show_anns_hou(self,anns):
        if len(anns) == 0:
            return
        # sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        anns = anns.cpu().numpy()
        # img = np.ones((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 4))
        # img[:,:,3] = 0
        for ann in anns:
            # m = ann['segmentation']
            # color_mask = np.concatenate([np.random.random(3), [0.35]])
            # img[m] = color_mask

            # 绘制矩形框
            # bbox = ann['bbox']  # 获取边界框 [xmin, ymin, width, height]
            xmin, ymin, xmax, ymax = ann
            # 使用xmin, ymin, width, height来绘制矩形框
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r',
                                     facecolor='none')
            ax.add_patch(rect)  # 将矩形框添加到图像上

        # ax.imshow(img)