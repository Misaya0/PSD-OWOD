# Copyright (c) Facebook, Inc. and its affiliates.
from asyncio.log import logger
from turtle import pos
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
import collections
import copy
import os

import detectron2.detectron2.utils.comm as comm
from detectron2.detectron2.config import configurable
from detectron2.detectron2.layers import Conv2d, ShapeSpec, cat
from detectron2.detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.detectron2.utils.events import get_event_storage
from detectron2.detectron2.utils.memory import retry_if_cuda_oom

from detectron2.detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.detectron2.modeling.matcher import Matcher
from detectron2.detectron2.modeling.sampling import subsample_labels
from detectron2.detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.detectron2.modeling.proposal_generator.proposal_utils import find_top_rpn_proposals
from detectron2.detectron2.utils.registry import Registry
from core.distribution_fitter import get_distribution
from torchvision.ops import roi_align

RPN_HEAD_REGISTRY = Registry("RPN_HEAD")

def build_rpn_head(cfg, input_shape):
    """
    Build an RPN head defined by `cfg.MODEL.RPN.HEAD_NAME`.
    """
    name = cfg.MODEL.RPN.HEAD_NAME
    return RPN_HEAD_REGISTRY.get(name)(cfg, input_shape)


@RPN_HEAD_REGISTRY.register()
class OFFLINE_AE_RPNHead(nn.Module):

    @configurable
    def __init__(
        self, *, in_channels: int, enable_rew: bool, ae_inter: List[int], 
    ):
        super().__init__()
        num_levels = 5
        self.enable_rew = enable_rew

        # AutoEncoder Head
        if self.enable_rew:
            for level in range(num_levels):
                conv_dict_ae = collections.OrderedDict()
                conv_dict_ae["encoder-level" + str(level)] = nn.Conv2d(
                    in_channels, ae_inter[level], 
                    kernel_size=1, stride=1, 
                )
                conv_dict_ae["decoder-level" + str(level)] = nn.Conv2d(
                    ae_inter[level], in_channels,
                    kernel_size=1, stride=1
                )
                setattr(self, "aehead"+str(level), nn.Sequential(conv_dict_ae))

        # Keeping the order of weights initialization same for backwards compatiblility.
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        # RPNHead should take the same input as anchor generator
        # NOTE: it assumes that creating an anchor generator does not have unwanted side effect.
        anchor_generator = build_anchor_generator(cfg, input_shape)
        num_anchors = anchor_generator.num_anchors
        assert (
            len(set(num_anchors)) == 1
        ), "Each level must have the same number of anchors per spatial position"
        return {
            "in_channels": in_channels,
            "enable_rew": cfg.OPENSET.ENABLE_REW,
            "ae_inter": cfg.OPENSET.REW.AE_INTER, 
        }

    def forward(self, features: List[torch.Tensor]):
        #RPN Head
        recons_error_by_level = []
        recons_error_map_by_level = []
        for level, x in enumerate(features):
            if self.enable_rew:
                ae_head = getattr(self, "aehead"+str(level))#自编码器头部
                ae_output = ae_head(x.detach())#自编码器输出
                recons_error_map = ((ae_output - x.detach())**2).mean(dim = 1).unsqueeze(dim = 1)#重构误差映射
                recons_error_map = torch.sqrt(recons_error_map)#重构误差映射
                recons_error_map_by_level.append(recons_error_map)#各层级的重构误差映射
                recons_error = recons_error_map.expand(
                    [recons_error_map.shape[0], 3
                    , recons_error_map.shape[2], recons_error_map.shape[3]])#重构误差
                recons_error_by_level.append(recons_error)
        return recons_error_by_level, recons_error_map_by_level
    
    


@PROPOSAL_GENERATOR_REGISTRY.register()
class OFFLINE_AE_RPN(nn.Module):
    """
    Region Proposal Network, introduced by :paper:`Faster R-CNN`.
    """
    @configurable
    def __init__(
        self,
        *,
        in_features: List[str],
        head: nn.Module,
        anchor_generator: nn.Module,
        anchor_matcher: Matcher,
        box2box_transform: Box2BoxTransform,
        batch_size_per_image: int,
        positive_fraction: float,
        pre_nms_topk: Tuple[float, float],
        post_nms_topk: Tuple[float, float],
        nms_thresh: float = 0.7,
        min_box_size: float = 0.0,
        anchor_boundary_thresh: float = -1.0,
        loss_weight: Union[float, Dict[str, float]] = 1.0,
        box_reg_loss_type: str = "smooth_l1",
        smooth_l1_beta: float = 0.0,
        enable_rew: bool = True,
        num_samples: int = 40000, 
        sampling_iters: int = 4000, 
        update_weibull: bool = False,
        rew_inference: bool = False,
        config_name: str = "",
    ):
        
        super().__init__()
        self.in_features = in_features
        self.rpn_head = head
        self.anchor_generator = anchor_generator
        self.anchor_matcher = anchor_matcher
        self.box2box_transform = box2box_transform
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        # Map from self.training state to train/test settings
        self.pre_nms_topk = {True: pre_nms_topk[0], False: pre_nms_topk[1]}
        self.post_nms_topk = {True: post_nms_topk[0], False: post_nms_topk[1]}
        self.nms_thresh = nms_thresh
        self.min_box_size = float(min_box_size)
        self.anchor_boundary_thresh = anchor_boundary_thresh
        if isinstance(loss_weight, float):
            loss_weight = {"loss_rpn_cls": loss_weight, "loss_rpn_loc": loss_weight}
        self.loss_weight = loss_weight
        self.box_reg_loss_type = box_reg_loss_type
        self.smooth_l1_beta = smooth_l1_beta
        # self._logger = logging.getLogger(__name__)
        
        self.enable_rew = enable_rew
        self.update_weibull = update_weibull
        self.num_samples = num_samples
        self.sampling_iters = sampling_iters
        self.recons_error_levels_kn = torch.zeros([5, self.num_samples]).cuda()
        self.recons_error_levels_bg = torch.zeros([5, 2 * self.num_samples]).cuda()
        self.recons_error_levels_ptr = torch.zeros([5,2]).int().cuda()
        self.register_buffer("y_bg", torch.zeros([5,501]).cuda())
        self.register_buffer("y_kn", torch.zeros([5,501]).cuda())
        self.record_list_kn = [] 
        self.record_list_bg = []
        self.soft_label_record = []
        self.iter = 0
        self.rew_inference = rew_inference
        if self.rew_inference:
            self.training = False
        self.config_name = config_name
    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        in_features = cfg.MODEL.RPN.IN_FEATURES
        ret = {
            "in_features": in_features,
            "min_box_size": cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE,
            "nms_thresh": cfg.MODEL.RPN.NMS_THRESH,
            "batch_size_per_image": cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE,
            "positive_fraction": cfg.MODEL.RPN.POSITIVE_FRACTION,
            "loss_weight": {
                "loss_rpn_cls": cfg.MODEL.RPN.LOSS_WEIGHT,
                "loss_rpn_loc": cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT * cfg.MODEL.RPN.LOSS_WEIGHT,
            },
            "anchor_boundary_thresh": cfg.MODEL.RPN.BOUNDARY_THRESH,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS),
            "box_reg_loss_type": cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE,
            "smooth_l1_beta": cfg.MODEL.RPN.SMOOTH_L1_BETA,

            "enable_rew": cfg.OPENSET.ENABLE_REW,
            "sampling_iters": cfg.OPENSET.REW.SAMPLING_ITERS, 
            "num_samples": cfg.OPENSET.REW.NUM_SAMPLES, 
            "update_weibull": cfg.OPENSET.REW.UPDATE_WEIBULL,
            "rew_inference": cfg.OPENSET.REW_INFERENCE,
            "config_name": cfg.CONFIG_NAME
        }

        ret["pre_nms_topk"] = (cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN, cfg.MODEL.RPN.PRE_NMS_TOPK_TEST)
        ret["post_nms_topk"] = (cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN, cfg.MODEL.RPN.POST_NMS_TOPK_TEST)
        ret["anchor_generator"] = build_anchor_generator(cfg, [input_shape[f] for f in in_features])
        ret["anchor_matcher"] = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        ret["head"] = build_rpn_head(cfg, [input_shape[f] for f in in_features])

        return ret

    def _subsample_labels(self, label):
        """
        Randomly sample a subset of positive and negative examples, and overwrite
        the label vector to the ignore value (-1) for all elements that are not
        included in the sample.

        Args:
            labels (Tensor): a vector of -1, 0, 1. Will be modified in-place and returned.
        """
        pos_idx, neg_idx = subsample_labels(
            label, self.batch_size_per_image, self.positive_fraction, 0
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
        anchors = Boxes.cat(anchors)
        
        gt_boxes = [x.gt_boxes for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]
        gt_classes = [x.gt_classes for x in gt_instances]

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
        matched_idx_list = []
        for image_size_i, gt_boxes_i, gt_classes_i in zip(image_sizes, gt_boxes, \
            gt_classes):
            """
            image_size_i: (h, w) for the i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            """
            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)
            # Matching is memory-expensive and may result in CPU tensors. But the result is small
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            del match_quality_matrix

            if self.anchor_boundary_thresh >= 0:
                # Discard anchors that go out of the boundaries of the image
                # NOTE: This is legacy functionality that is turned off by default in Detectron2
                anchors_inside_image = anchors.inside_box(image_size_i, self.anchor_boundary_thresh)
                gt_labels_i[~anchors_inside_image] = -1
            
            gt_labels_i = self._subsample_labels(gt_labels_i)
            
            matched_gt_classes_i = gt_classes_i[matched_idxs]#与anchors匹配的gt类别
            matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor#与anchors匹配的gt框

            gt_labels.append(gt_labels_i)  # N,AHW
            matched_gt_boxes.append(matched_gt_boxes_i)
            matched_gt_classes.append(matched_gt_classes_i)
            matched_idx_list.append(matched_idxs)
        return gt_labels, matched_gt_boxes, matched_gt_classes, matched_idx_list, unk_idxs
            
    def asign_soft_label(self, images, gt_instances, recons_error_map_by_level):

        h_img, w_img = images.tensor.shape[2], images.tensor.shape[3]
        images = images.tensor
        for i in range(len(gt_instances)):
            level = 2
            # assert (gt_instances[i].gt_classes != 80).any()
            gt_boxes = copy.deepcopy(gt_instances[i].gt_boxes.tensor)

            soft_label_levels = []  # 软标签数据
            valid_indices = []  # 保存有效边界框的索引

            bbox_size = [0, 32 ** 2, 64 ** 2, 128 ** 2, 256 ** 2, 3000 ** 2]  # 定义了不同层级对应的边界框面积范围
            bbox_level = torch.full([gt_boxes.shape[0]], -1)  # 初始化一个与真实边界框数量相同的张量，用于记录每个真实边界框所属的层级
            bbox_w, bbox_h = gt_boxes[:, 2] - gt_boxes[:, 0], gt_boxes[:, 3] - gt_boxes[:, 1]  # 分别计算每个真实边界框的宽度和高度
            bbox_area = (bbox_w * bbox_h)  # 真实边界框面积
            gt_boxes[:, 0], gt_boxes[:, 1] = gt_boxes[:, 0] + (7 / 16) * bbox_w, gt_boxes[:, 1] + (
                        7 / 16) * bbox_h  # 对真实边界框的坐标进行调整
            gt_boxes[:, 2], gt_boxes[:, 3] = gt_boxes[:, 2] - (7 / 16) * bbox_w, gt_boxes[:, 3] - (7 / 16) * bbox_h

            self.y_bg = torch.load(os.path.join("save", self.config_name, "bg.pth"), map_location='cpu')
            self.y_kn = torch.load(os.path.join("save", self.config_name, "kn.pth"), map_location='cpu')
            foreground_left_boundaries = torch.load(
                os.path.join("save", self.config_name, "foreground_left_boundaries.pth"), map_location='cpu')
            for level in range(0, 5):
                error_map = recons_error_map_by_level[level]  # 获取当前层级的重构误差映射数据
                h_fea, w_fea = error_map.shape[2], error_map.shape[
                    3]  # 获取当前层级误差映射的高度和宽度，用于后续的空间尺度计算以及感兴趣区域对齐（ROI Align）操作等
                spatial_scale = h_fea / h_img  # 计算空间尺度因子
                error_per_img = roi_align(input=error_map, \
                                          boxes=[gt_boxes], output_size=1, spatial_scale=spatial_scale).reshape([
                                                                                                                    -1])  # 使用感兴趣区域对齐（ROI Align）操作，将每个真实边界框对应的区域从误差映射中提取出来，并统一调整输出大小为1（可能是将提取的区域进行池化等处理为固定大小的表示），然后将结果重塑为一维张量，得到每个真实边界框在当前层级误差映射上对应的误差值
                error_per_img = (error_per_img / 0.01).int()
                error_per_img[error_per_img > 500] = 500
                error_per_img = error_per_img.cpu().numpy()

                bbox_level[(bbox_area >= (bbox_size[level])) & (bbox_area < (bbox_size[level + 1]))] = level  # 根据边界框面积判断每个真实边界框所属的层级，将面积在当前层级对应的面积范围（[bbox_size[level], bbox_size[level + 1])）内的边界框标记为当前层级，更新 `bbox_level` 张量中对应边界框的层级标记
                # 根据当前层级以及处理后的误差值，从预先存储的背景样本分布和前景样本分布相关数据中提取对应位置的数据，转换为浮点数类型，用于后续计算软标签
                y_bg = self.y_bg[level, error_per_img].float()
                y_kn = self.y_kn[level, error_per_img].float()
                soft_label = (y_kn) / (y_bg + y_kn)  # 计算软标签分数
                # soft_label = error_per_img
                soft_label[gt_instances[i].gt_classes != 80] = 1.0
                soft_label_levels.append(soft_label.reshape([1, -1]))
                # 筛选当前层级的有效实例
                # foreground_left_boundary = foreground_left_boundaries[level]  # 当前层级的前景左边界
                # valid_indices_level = ((bbox_level == level) & (soft_label.cpu() >= foreground_left_boundary.cpu()))
                # valid_indices.append(valid_indices_level.reshape([1, -1]))

            # 将不同层级的软标签数据沿着维度0进行拼接，形成一个二维张量，然后根据之前确定的每个边界框所属的层级（bbox_level），从拼接后的张量中选取对应层级的软标签数据，整理成最终的软标签张量形式，确保每个边界框都获取到其对应层级计算出的软标签值
            soft_label_levels = torch.cat(soft_label_levels, dim=0)[bbox_level.long(), range(bbox_level.shape[0])]
            # print(bbox_level[torch.sort(soft_label_levels)[1]])
            # # print(bbox_level.tolist(), soft_label_levels.tolist(), gt_instances[0].gt_classes.tolist())
            # 合并所有层级的有效索引
            # valid_indices = torch.cat(valid_indices, dim=0)[
            #     bbox_level.long(), range(bbox_level.shape[0])]  # 将每层的有效索引合并为一个整体
            # 筛选有效的边界框和软标签
            # filtered_boxes = gt_boxes[valid_indices]
            # filtered_soft_labels = soft_label_levels[valid_indices].to(device=gt_instances[0].gt_boxes.tensor.device)

            ret = gt_instances[i]
            # ret.gt_boxes = Boxes(ret.gt_boxes.tensor[valid_indices])
            # ret.gt_classes = ret.gt_classes[valid_indices]  # 筛选 gt_classes# 筛选 gt_boxes
            ret.set("pred_boxes", Boxes(gt_instances[i].gt_boxes.tensor))
            ret.set("soft_labels", soft_label_levels.to(device=gt_instances[0].gt_boxes.tensor.device))
            gt_instances[i] = ret
            # assert (gt_instances[i].gt_classes != 80).any()

        
        
        
    # def asign_soft_label(self, gt_labels, gt_instances,\
    #      unk_idxs, recons_error_list, gt_classes, matched_idx_list):
        
    #     num_images = len(gt_labels)
    #     num_levels = len(recons_error_list)

    #     gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))
    #     gt_classes = torch.stack(gt_classes)  # (N, sum(Hi*Wi*Ai))
    #     matched_idx = torch.stack(matched_idx_list) # (N, sum(Hi*Wi*Ai))
    #     num_pixel_levels = [0, 0, 0, 0, 0, 0]
    #     gt_labels_by_level = []
    #     gt_classes_by_level = []
    #     matched_idx_by_level = []
        
    #     for level in range(num_levels):
    #         if level > 0:
    #             num_pixel_levels[level + 1] = num_pixel_levels[level] + recons_error_list[level].shape[1]
    #         else:
    #             num_pixel_levels[level + 1] = recons_error_list[level].shape[1]
    #         gt_labels_level = gt_labels[:, num_pixel_levels[level]:num_pixel_levels[level + 1]]
    #         gt_classes_level = gt_classes[:, num_pixel_levels[level]:num_pixel_levels[level + 1]]
    #         matched_idx_level = matched_idx[:, num_pixel_levels[level]:num_pixel_levels[level + 1]]
    #         gt_labels_by_level.append(gt_labels_level)
    #         gt_classes_by_level.append(gt_classes_level)
    #         matched_idx_by_level.append(matched_idx_level)

    #     matched_idx_unk = []
    #     recons_error_unk = []
    #     for level in range(num_levels):
    #         matched_idx_unk_level = []
    #         recons_error_unk_level = []
    #         for img in range(num_images):
    #             gt_classes_level_i = gt_classes_by_level[level][img]
    #             gt_labels_level_i = gt_labels_by_level[level][img]
    #             recons_error_unk_level_i = recons_error_list[level][img][(
    #                 gt_classes_level_i == 80) & (gt_labels_level_i == 1)].detach().cpu().numpy()
    #             matched_idx_unk_level_i = matched_idx_by_level[level][img][(
    #                 gt_classes_level_i == 80) & (gt_labels_level_i == 1)].detach().cpu().numpy()
    #             recons_error_unk_level.append(recons_error_unk_level_i)
    #             matched_idx_unk_level.append(matched_idx_unk_level_i)
    #         matched_idx_unk.append(matched_idx_unk_level)
    #         recons_error_unk.append(recons_error_unk_level)

    #     unk_idx_soft_label = []
    #     for img in range(num_images):
    #         unk_idx_soft_label_i = {}
    #         for level in range(num_levels):
    #             matched_idx_unk_level_i = matched_idx_unk[level][img]
    #             recons_error_unk_level_i = recons_error_unk[level][img]
    #             for unk_idx in unk_idxs[img]:
    #                 if(len(matched_idx_unk_level_i) and (unk_idx in matched_idx_unk_level_i)):
    #                     x = (recons_error_unk_level_i[matched_idx_unk_level_i \
    #                         == unk_idx]).mean().item() 
    #                     x = int(x//0.01)
    #                     if x > 500:
    #                         x = 500
    #                     y_bg = float(self.y_bg[level, x].item())
    #                     y_kn = float(self.y_kn[level, x].item())
    #                     soft_label = (y_kn)/(y_bg + y_kn)
    #                     self.soft_label_record.append(soft_label)
    #                     unk_idx_soft_label_i[unk_idx] = soft_label
    #         unk_idx_soft_label.append(unk_idx_soft_label_i)

    #     gt_boxes = [x.gt_boxes for x in gt_instances]
    #     image_sizes = [x.image_size for x in gt_instances]
    #     gt_classes = [x.gt_classes for x in gt_instances]
    #     for i, (gt_boxes_i, gt_classes_i, image_size_i, unk_idx_soft_label_i) in \
    #         enumerate(zip(gt_boxes, gt_classes, image_sizes, unk_idx_soft_label)):
    #         gt_boxes_i_tensor = gt_boxes_i.tensor
    #         soft_labels_i = torch.ones_like(gt_classes_i).float()
    #         for idx in range(gt_boxes_i_tensor.shape[0]):
    #             soft_label = unk_idx_soft_label_i.get(idx)
    #             if soft_label != None:
    #                 soft_labels_i[idx] = soft_label
    #         ret = gt_instances[i]
    #         ret.set("pred_boxes", Boxes(gt_boxes_i_tensor))
    #         ret.set("soft_labels", soft_labels_i)
    #         gt_instances[i] = ret

    @torch.jit.unused
    def losses(
        self,
        recons_error_by_level: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        if self.enable_rew:
            recons_error_by_level = cat(recons_error_by_level, dim = 1)
            ae_loss = (recons_error_by_level**2).mean()
        losses = {}
        if self.enable_rew:
            losses["loss_ae"] = ae_loss

        return losses
            
    def update_weibull_function(self, gt_labels, recons_error_list, gt_classes, matched_idx_list):
        # storage = get_event_storage()
        num_images = len(gt_labels)
        num_levels = len(recons_error_list)

        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))
        gt_classes = torch.stack(gt_classes)  # (N, sum(Hi*Wi*Ai))
        matched_idx = torch.stack(matched_idx_list) # (N, sum(Hi*Wi*Ai))
        num_pixel_levels = [0, 0, 0, 0, 0, 0]#像素数量
        gt_labels_by_level = []
        gt_classes_by_level = []
        matched_idx_by_level = []
        foreground_left_boundaries = []
        with torch.no_grad():
            for level in range(num_levels):
                if level > 0:
                    num_pixel_levels[level + 1] = num_pixel_levels[level] + recons_error_list[level].shape[1]
                else:
                    num_pixel_levels[level + 1] = recons_error_list[level].shape[1]
                gt_labels_level = gt_labels[:, num_pixel_levels[level]:num_pixel_levels[level + 1]]
                gt_classes_level = gt_classes[:, num_pixel_levels[level]:num_pixel_levels[level + 1]]
                matched_idx_level = matched_idx[:, num_pixel_levels[level]:num_pixel_levels[level + 1]]
                gt_labels_by_level.append(gt_labels_level)
                gt_classes_by_level.append(gt_classes_level)
                matched_idx_by_level.append(matched_idx_level)
            
            for level, re in enumerate(recons_error_list):
                
                recons_error_level_kn = re[(gt_classes_by_level[level] != 80) & (gt_labels_by_level[level] == 1)]\
                    .clone().detach()#筛选出已知类别且为正样本
                sample_idx = []
                if recons_error_level_kn.numel() > 0:
                    sample_idx = torch.randint(0, recons_error_level_kn.numel(), [40])#随机采样40个
                self.recons_error_levels_ptr[level][0] = self.record_in_queue(self.recons_error_levels_kn[level], \
                    self.recons_error_levels_ptr[level][0], recons_error_level_kn[sample_idx])#将采样的正样本加入队列

                recons_error_level_bg = re[(gt_labels_by_level[level] == 0)].clone().detach()#筛选出背景样本
                sample_idx = []
                if recons_error_level_bg.numel() > 0:
                    sample_idx = torch.randint(0, recons_error_level_bg.numel(), [80])#随机采样80个
                self.recons_error_levels_ptr[level][1] = self.record_in_queue(self.recons_error_levels_bg[level], \
                    self.recons_error_levels_ptr[level][1], recons_error_level_bg[sample_idx])#将采样的背景样本加入队列

            if self.iter % self.sampling_iters == self.sampling_iters - 1:
                self.record_list_kn, self.record_list_bg = self.gather_all_record()
                
                if comm.is_main_process():
                    for level in range(num_levels):
                        record_list_kn_level = self.record_list_kn[level].cpu().numpy()
                        record_list_bg_level = self.record_list_bg[level].cpu().numpy()
                        y_bg_level, y_kn_level, foreground_left_boundary = get_distribution(
                            record_list_bg_level, record_list_kn_level,
                            picture_name="level"+str(level)+".jpg" if comm.is_main_process() else None)#计算分布
                        self.y_bg[level] = torch.tensor(y_bg_level).cuda()
                        self.y_kn[level] = torch.tensor(y_kn_level).cuda()
                        foreground_left_boundaries.append(foreground_left_boundary)
                foreground_left_boundaries_tensor = torch.tensor(foreground_left_boundaries).view(-1,1).cuda()  # (5, 1)
                if os.path.exists(os.path.join("save",self.config_name)) == False:
                    os.mkdir(os.path.join("save",self.config_name))
                torch.save(foreground_left_boundaries_tensor, os.path.join("save",self.config_name, "foreground_left_boundaries.pth"))  # 保存为文件

                world_size = comm.get_world_size()
                if world_size > 1:
                    torch.distributed.broadcast(self.y_kn, 0)
                    torch.distributed.broadcast(self.y_bg, 0)
                if os.path.exists("save") == False:
                    os.mkdir("save")
                if os.path.exists(os.path.join("save",self.config_name)) == False:
                    os.mkdir(os.path.join("save",self.config_name))
                bg_path = os.path.join("save",self.config_name, "bg.pth")
                kn_path = os.path.join("save",self.config_name, "kn.pth")
                torch.save(self.y_bg, bg_path)
                print("save weibull function to ", bg_path)
                torch.save(self.y_kn, kn_path)
                print("save weibull function to ", kn_path)
                self.recons_error_levels_kn = torch.zeros([5, self.num_samples]).cuda()
                self.recons_error_levels_bg = torch.zeros([5, 2 * self.num_samples]).cuda()
                self.recons_error_levels_ptr = torch.zeros([5,2]).int().cuda()
        return num_images, num_levels, gt_labels_by_level, gt_classes_by_level, matched_idx_by_level

    def gather_all_record(self):
        world_size = comm.get_world_size()
        if world_size == 1:
            tensor_list_kn = [self.recons_error_levels_kn]
            tensor_list_bg = [self.recons_error_levels_bg]
            tensor_list_ptr = [self.recons_error_levels_ptr]
        else:
            tensor_list_kn = [torch.ones_like(self.recons_error_levels_kn) for _ in range(world_size)]
            tensor_list_bg = [torch.ones_like(self.recons_error_levels_bg) for _ in range(world_size)]
            tensor_list_ptr = [torch.ones_like(self.recons_error_levels_ptr) for _ in range(world_size)]
            torch.distributed.all_gather(tensor_list_kn, self.recons_error_levels_kn, async_op=False)
            torch.distributed.all_gather(tensor_list_bg, self.recons_error_levels_bg, async_op=False)
            torch.distributed.all_gather(tensor_list_ptr, self.recons_error_levels_ptr, async_op=False)
        record_list_kn, record_list_bg = [], []
        for level in range(5):
            record_list_unk_level, record_list_kn_level, record_list_bg_level = [], [], []
            for t_kn, t_bg, t_ptr in zip(tensor_list_kn, tensor_list_bg, \
                tensor_list_ptr):
                ptr_level = t_ptr[level]
                kn_record_level = t_kn[level][0:ptr_level[0]]
                bg_record_level = t_bg[level][0:ptr_level[1]]
                record_list_kn_level.append(kn_record_level)
                record_list_bg_level.append(bg_record_level)
            record_list_kn_level = torch.cat(record_list_kn_level)
            record_list_bg_level = torch.cat(record_list_bg_level)

            record_list_kn.append(record_list_kn_level)
            record_list_bg.append(record_list_bg_level)
        return record_list_kn, record_list_bg

    def record_in_queue(self, record, ptr, new_data):#将新数据加入队列
        max_len = record.numel()
        new_len = new_data.numel()
        if ptr == max_len:
            return ptr
        elif ptr + new_len <= max_len:
            record[ptr: ptr + new_len] = new_data
            ptr = ptr + new_len
            return ptr
        else:
            remain_len = max_len - ptr
            record[ptr:] = new_data[-remain_len:]
            ptr = max_len
            return ptr

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[List[Instances]] = None,
    ):

        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)

        recons_error_by_level, recons_error_map_by_level = self.rpn_head(features)#重构误差 和 重构误差映射
        
        recons_error_by_level = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            recons_error.permute(0, 2, 3, 1).flatten(1)
            for recons_error in recons_error_by_level
        ]
        
        if not self.rew_inference:
            losses = self.losses(recons_error_by_level)
            if self.update_weibull:
                self.iter = self.iter + 1
                gt_labels, gt_boxes, gt_classes, idx_list, unk_idxs = \
                    self.label_and_sample_anchors(anchors, gt_instances)
                self.update_weibull_function(gt_labels, recons_error_by_level, gt_classes, idx_list)
            return losses
        else:
            # gt_labels, gt_boxes, gt_classes, idx_list, unk_idxs = \
            #     self.label_and_sample_anchors(anchors, gt_instances)
            # self.asign_soft_label(gt_labels, gt_instances, unk_idxs, recons_error_by_level, \
            #     gt_classes, idx_list)
            self.asign_soft_label(images, gt_instances, recons_error_map_by_level)
            return gt_instances
