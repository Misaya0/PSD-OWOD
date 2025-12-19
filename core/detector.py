import math
import random
from typing import List
from collections import namedtuple
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.detectron2.layers import batched_nms
from detectron2.detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN

from detectron2.detectron2.structures import Boxes, ImageList, Instances

from .loss import SetCriterionDynamicK, HungarianMatcherDynamicK
from .head import DynamicHead
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import nested_tensor_from_tensor_list
from detectron2.detectron2.modeling.matcher import Matcher

# from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
# from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry
import cv2
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patches as patches
import time

__all__ = ["RandBox"]

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# print("Registering RandBox...")
@META_ARCH_REGISTRY.register()
class RandBox(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.num_proposals = cfg.MODEL.NUM_PROPOSALS
        self.hidden_dim = cfg.MODEL.HIDDEN_DIM
        self.num_heads = cfg.MODEL.NUM_HEADS
        self.sampling_method = cfg.MODEL.SAMPLING_METHOD
        self.disentangled = cfg.MODEL.DISENTANGLED
        # Build Backbone.
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility
        # self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())  ##使用RPN时开启
        self.proposal_generator = None
        # build diffusion
        timesteps = 1000
        sampling_timesteps = cfg.MODEL.SAMPLE_STEP
        self.objective = 'pred_x0'
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.multiple_sample = cfg.MODEL.M_STEP
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = cfg.MODEL.SNR_SCALE

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        self.x_dic = torch.rand((10000, 4))
        self.x_meta = torch.arange(start=-2, end=2, step=0.4)
        for i1 in range(10):
            for i2 in range(10):
                for i3 in range(10):
                    for i4 in range(10):
                        self.x_dic[i1 * 1000 + i2 * 100 + i3 * 10 + i4][0], \
                        self.x_dic[i1 * 1000 + i2 * 100 + i3 * 10 + i4][1], \
                        self.x_dic[i1 * 1000 + i2 * 100 + i3 * 10 + i4][2], \
                        self.x_dic[i1 * 1000 + i2 * 100 + i3 * 10 + i4][3] = self.x_meta[i1], self.x_meta[i2], \
                        self.x_meta[i3], self.x_meta[i4]
        self.x_dic = self.x_dic[torch.randperm(self.x_dic.size(0))]
        # Build Dynamic Head.
        self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())
        # Loss parameters:
        class_weight = cfg.MODEL.CLASS_WEIGHT
        giou_weight = cfg.MODEL.GIOU_WEIGHT
        l1_weight = cfg.MODEL.L1_WEIGHT
        nc_weight = cfg.MODEL.NC_WEIGHT
        no_object_weight = cfg.MODEL.NO_OBJECT_WEIGHT
        decorr_weight = cfg.MODEL.DECORR_WEIGHT
        self.deep_supervision = cfg.MODEL.DEEP_SUPERVISION
        self.use_nms = cfg.MODEL.USE_NMS

        # Build Criterion.
        matcher = HungarianMatcherDynamicK(
            cfg=cfg, cost_class=class_weight, cost_bbox=l1_weight, cost_giou=giou_weight
        )
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight,
                       "loss_nc_ce": nc_weight, "loss_decorr": decorr_weight, "loss_ufdm": 1,
                       "loss_rpn_cls":1.0, "loss_rpn_loc":1.0}
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes"]
        if cfg.MODEL.NC:
            losses += ["nc_labels"]
            losses += ["ufdm"]
            # losses += ["nc_labels", "objectness"]
            # losses += ["nc_labels", "obj_likelihood"]
        if decorr_weight > 0:
            losses += ["decorr"]

        anchor_matcher = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        self.criterion = SetCriterionDynamicK(
            cfg=cfg, num_classes=self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight,
            losses=losses, anchor_matcher=anchor_matcher)


        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

        self.count = 0

    def _strong_photometric(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (B,3,H,W) 已经过 normalizer 的张量也可以，但更推荐对未normalize做增强。
        这里用一个“不会改变几何”的强增强：亮度/对比度扰动 + 噪声（简单可控）
        """
        # 假设 img 已是 float32
        # 亮度
        if torch.rand(1, device=img.device) < 0.8:
            delta = (torch.rand((img.shape[0], 1, 1, 1), device=img.device) - 0.5) * 0.2
            img = img + delta

        # 对比度
        if torch.rand(1, device=img.device) < 0.8:
            mean = img.mean(dim=(2, 3), keepdim=True)
            factor = 1.0 + (torch.rand((img.shape[0], 1, 1, 1), device=img.device) - 0.5) * 0.5
            img = (img - mean) * factor + mean

        # 轻微高斯噪声
        if torch.rand(1, device=img.device) < 0.5:
            noise = torch.randn_like(img) * 0.03
            img = img + noise

        return img
    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, backbone_feats, images_whwh, x, t, x_self_cond=None, clip_x_start=False, sample_i=0):
        if self.sampling_method == 'Random':
            x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale)
            x_boxes = ((x_boxes / self.scale) + 1) / 2
        else:
            x_boxes = self.x_dic.to(x.device)[self.num_proposals * sample_i:self.num_proposals * (sample_i + 1), :]#获取当前采样步骤的候选框
            x_boxes = ((x_boxes / self.scale) + 1) / 2 #归一化

        x_boxes = box_cxcywh_to_xyxy(x_boxes)
        x_boxes = x_boxes * images_whwh[:, None, :]
        outputs_class, outputs_objectness, outputs_coord, outputs_feat = self.head(backbone_feats, x_boxes, t, None)

        x_start = outputs_coord[-1]  # (batch, num_proposals, 4) predict boxes: absolute coordinates (x1, y1, x2, y2)
        x_start = x_start / images_whwh[:, None, :]
        x_start = box_xyxy_to_cxcywh(x_start)
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start), outputs_class, outputs_objectness, outputs_coord

    @torch.no_grad()
    def ddim_sample(self, batched_inputs, backbone_feats, images_whwh, images, clip_denoised=True, do_postprocess=True):
        batch = images_whwh.shape[0]
        shape = (batch, self.num_proposals, 4)
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=self.device)

        x_start = None
        if self.sampling_method == 'Random':
            for time, time_next in time_pairs:
                time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
                self_cond = x_start if self.self_condition else None

                preds, class_cat, objectness_cat, coord_cat = self.model_predictions(backbone_feats, images_whwh, img, time_cond,
                                                                     self_cond, clip_x_start=clip_denoised)
                pred_noise, x_start = preds.pred_noise, preds.pred_x_start
        else:
            for sample_step in range(self.multiple_sample):
                for time, time_next in time_pairs:
                    time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
                    self_cond = x_start if self.self_condition else None

                    preds, outputs_class, outputs_objectness, outputs_coord = self.model_predictions(backbone_feats, images_whwh, img,
                                                                                 time_cond,
                                                                                 self_cond, clip_x_start=clip_denoised,
                                                                                 sample_i=sample_step)
                if sample_step == 0:
                    class_cat = outputs_class
                    objectness_cat = outputs_objectness
                    coord_cat = outputs_coord
                else:
                    class_cat = torch.cat((class_cat, outputs_class), 2)
                    objectness_cat = torch.cat((objectness_cat, outputs_objectness), 2)
                    coord_cat = torch.cat((coord_cat, outputs_coord), 2)

        results = self.inference(class_cat[-1], objectness_cat[-1], coord_cat[-1], images.image_sizes)

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise # x_t = (sqrt(α) * x_(t-1) + sqrt(1-α) * ε_t)

    def forward(self, batched_inputs, do_postprocess=True):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        # for batched_input in batched_inputs:
        #     if batched_input['image_id'] == "2010_005425":
        #         print("111")
        images, images_whwh = self.preprocess_image(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)
        # boxes = [x["instances"].gt_boxes for x in batched_inputs]
        # plt.figure(figsize=(20, 20))
        # # image = images.transpose(0, 2, 1)
        # plt.imshow(images[0].cpu().numpy().transpose(1, 2, 0))
        # self.show_anns_hou(boxes[0])
        # plt.axis('off')
        # plt.show()
        # for input in batched_inputs:
        #     image = input['image']
        #     image = image.detach().numpy()
        #     image = image.transpose(0, 2, 1)
        #     bboxes_tensor = self.generate_mask(image.T)

        # for image in images:
        #     image = image.detach().cpu().numpy()
        #     image = image.transpose(0, 2, 1)
        #     bboxes_tensor = self.generate_mask(image.T)

        # image = tmp_img.detach().cpu().numpy()
        # image = image.transpose(0, 2, 1)
        # start_time = time.time()
        # bboxes_tensor = self.generate_mask(image.T)
        # print("Time used for generating mask: ", time.time() - start_time)

        # Feature Extraction.
        src = self.backbone(images.tensor)
        features = list()
        for f in self.in_features:
            feature = src[f]
            features.append(feature)

        # Prepare Proposals.
        # if not self.training:#推理阶段
        #     return self.inference_rpn(batched_inputs)
        if not self.training:#推理阶段
            results = self.ddim_sample(batched_inputs, features, images_whwh, images, do_postprocess=do_postprocess)
            return results
        # if self.training:#训练阶段
        #     if "instances" in batched_inputs[0]:
        #         gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        #     else:
        #         gt_instances = None
        #     if self.proposal_generator is not None:
        #         proposals, proposal_losses = self.proposal_generator(images, src, gt_instances)  # RPN生成候选框
        #     else:
        #         assert "proposals" in batched_inputs[0]
        #         proposals = [x["proposals"].to(self.device) for x in batched_inputs]
        #         proposal_losses = {}
        #     # known_instances, unknown_instances = self.update_instance(gt_instances)
        #     gt_instances = self.processing_soft_labels(gt_instances)
        #     targets, x_boxes, noises, t = self.prepare_targets(gt_instances)
        #
        #     outputs_class, output_objectness, outputs_coord = self.head(features, proposals, None)
        #     output = {'pred_logits': outputs_class[-1], 'pred_objectness': output_objectness[-1], 'pred_boxes': outputs_coord[-1]}
        #     # 将字典保存到 .txt 文件
        #     if self.count % 100 == 0:
        #         with open('data.txt', 'a') as f:
        #             for key, value in output.items():
        #                 # 将张量移动到 CPU 并转换为 NumPy 数组
        #                 value_cpu = value[:,:5,:].detach().cpu().numpy()
        #                 # 写入键
        #                 f.write(f"{key}:\n")
        #                 # 写入值
        #                 # np.savetxt(f, value_cpu.squeeze(), fmt='%.6f')
        #                 tensor_str =np.array2string(value_cpu.squeeze(), precision=6, separator=', ')
        #                 f.write(tensor_str)
        #                 f.write("\n")  # 添加空行分隔
        #     self.count += 1
        #     if self.deep_supervision:
        #         output['aux_outputs'] = [{'pred_logits': a, 'pred_objectness': b, 'pred_boxes': c}
        #                                  for a, b, c in zip(outputs_class[:-1], output_objectness[:-1], outputs_coord[:-1])]
        #     loss_dict = self.criterion(output, targets,proposals, gt_instances)
        #     loss_dict.update(proposal_losses)
        #     weight_dict = self.criterion.weight_dict
        #     for k in loss_dict.keys():
        #         if k in weight_dict:
        #             loss_dict[k] *= weight_dict[k]
        #     return loss_dict
        if self.training:#训练阶段
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            # known_instances, unknown_instances = self.update_instance(gt_instances)
            # gt_instances = self.processing_soft_labels(gt_instances)

            targets, x_boxes, noises, t = self.prepare_targets(gt_instances)

            t = t.squeeze(-1)
            x_boxes = x_boxes * images_whwh[:, None, :]#将边界框坐标修正为绝对坐标

            # outputs_class, output_objectness, outputs_coord, outputs_feat = self.head(features, x_boxes, t, None)
            # output = {
            #     'pred_logits': outputs_class[-1],
            #     'pred_objectness': output_objectness[-1],
            #     'pred_boxes': outputs_coord[-1],
            #     'pred_features': outputs_feat[-1]
            # }
            # -------- weak view (original) --------
            outputs_class, output_objectness, outputs_coord, outputs_feat = self.head(features, x_boxes, t, None)
            output = {
                'pred_logits': outputs_class[-1],
                'pred_objectness': output_objectness[-1],
                'pred_boxes': outputs_coord[-1],
                'pred_features': outputs_feat[-1]
            }

            # -------- strong view (photometric only; T is identity) --------
            images_s = self._strong_photometric(images.tensor)
            with torch.no_grad():
                src_s = self.backbone(images_s)
                features_s = []
                for f in self.in_features:
                    features_s.append(src_s[f])

                outputs_class_s, output_objectness_s, outputs_coord_s, outputs_feat_s = self.head(features_s, x_boxes, t,
                                                                                                  None)

                output["scv"] = {
                    "pred_logits": outputs_class_s[-1],
                    "pred_objectness": output_objectness_s[-1],
                    "pred_boxes": outputs_coord_s[-1],
                    "pred_features": outputs_feat_s[-1],
                }

            if self.deep_supervision:
                output['aux_outputs'] = [{'pred_logits': a, 'pred_objectness': b, 'pred_boxes': c, 'pred_features': d}
                                         for a, b, c, d in zip(outputs_class[:-1], output_objectness[:-1], outputs_coord[:-1], outputs_feat[:-1])]
            loss_dict = self.criterion(output, targets,x_boxes, gt_instances)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict

    def update_instance(self, targets: List[Instances]):
        known_instances, sam_instances = [], []
        for instance_item in targets:
            gt_classes = instance_item.gt_classes
            known_idx = (gt_classes != 80)
            known_instances.append(instance_item[known_idx])
            sam_instances.append(instance_item[~known_idx])
        return known_instances, sam_instances
    def show_anns_hou(self,anns):
        if len(anns) == 0:
            return
        # sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        anns = anns
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
      #生成未知实例伪标签
    def prepare_unknown_targets(self, targets, images):

        return
    def prepare_diffusion_concat(self, gt_boxes):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()#随机采样一个时间步
        noise = torch.randn(self.num_proposals, 4, device=self.device)#随机生成噪声

        num_gt = gt_boxes.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device=self.device)
            num_gt = 1

        box_placeholder = torch.randn(self.num_proposals - num_gt, 4,
                                      device=self.device) / 6. + 0.5  # 3sigma = 1/2 --> sigma: 1/6  随机边界框占位符
        box_placeholder[:, 2:] = torch.clip(box_placeholder[:, 2:], min=1e-4)#避免出现无效框
        x_start = torch.randn(self.num_proposals, 4, device=self.device) # 生成随机的边界框

        x_start = (x_start * 2. - 1.) * self.scale

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        diff_boxes = box_cxcywh_to_xyxy(x)

        return diff_boxes, noise, t

    def prepare_targets(self, targets):

        new_targets = []
        diffused_boxes = []
        noises = []
        ts = []
        for targets_per_image in targets:
            target = {}
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            d_boxes, d_noise, d_t = self.prepare_diffusion_concat(gt_boxes)#生成diffusion边界框(加上噪声后的)
            diffused_boxes.append(d_boxes)
            noises.append(d_noise)
            ts.append(d_t)
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)

            if targets_per_image.has("gt_scores"):
                target["scores"] = targets_per_image.gt_scores.to(self.device)
            else:
                target["scores"] = torch.ones((len(gt_boxes),), dtype=torch.float32, device=self.device)

            new_targets.append(target)

        return new_targets, torch.stack(diffused_boxes), torch.stack(noises), torch.stack(ts)

    # def prepare_targets(self, targets, images):
    #     new_targets = []
    #     sam_boxes = []
    #     noises = []
    #     ts = []
    #     for targets_per_image, image in zip(targets, images):
    #         target = {}
    #         h, w = targets_per_image.image_size
    #         image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
    #         gt_classes = targets_per_image.gt_classes
    #         gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
    #         gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
    #         # d_boxes, d_noise, d_t = self.prepare_diffusion_concat(gt_boxes)
    #         target["labels"] = gt_classes.to(self.device)
    #         target["boxes"] = gt_boxes.to(self.device)
    #         target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
    #         target["image_size_xyxy"] = image_size_xyxy.to(self.device)
    #         image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
    #         target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
    #         target["area"] = targets_per_image.gt_boxes.area().to(self.device)
    #         new_targets.append(target)
    #         image = image.detach().cpu().numpy()
    #         image = image.transpose(0, 2, 1)
    #         # image = cv2.cvtColor(image.T, cv2.COLOR_BGR2RGB)
    #         bboxes_tensor = self.generate_mask(image.T)
    #         sam_boxes.append(bboxes_tensor)
    #
    #
    #     return new_targets, torch.stack(sam_boxes), torch.stack(noises), torch.stack(ts)

    def show_anns(self,anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask

            # 绘制矩形框
            bbox = ann['bbox']  # 获取边界框 [xmin, ymin, width, height]
            xmin, ymin, width, height = bbox
            # 使用xmin, ymin, width, height来绘制矩形框
            rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)  # 将矩形框添加到图像上

        ax.imshow(img)

    def processing_soft_labels(self, gt_instances):

        if gt_instances[0].has("soft_labels"):
            for i in range(len(gt_instances)):
                g = gt_instances[i]
                gt_boxes = g.gt_boxes
                gt_classes = g.gt_classes
                soft_labels = (g.soft_labels) ** 4
                # soft_labels = ((g.soft_labels)**self.ae_gamma)
                soft_labels[soft_labels > 1] = 1
                g_new = Instances(g.image_size)
                g_new.gt_boxes = gt_boxes
                g_new.gt_classes = gt_classes
                g_new.soft_labels = soft_labels
                gt_instances[i] = g_new

        return gt_instances

    def inference(self, box_cls, box_objectness, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).    每个候选框的分类概率
                The tensor predicts the classification probability for each proposal.
            box_objectness (Tensor): tensors of shape (batch_size, num_proposals, 1).   候选框是否包含物体的概率
                The tensor predicts the objectness for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).   每个候选框对应的边界框回归预测
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # 计算最终得分
        if self.disentangled == 0:
            scores = torch.sigmoid(box_cls)
        else:
            scores = torch.softmax(box_cls, dim=-1) * box_objectness  # 结合分类得分和目标得分

        # 生成类别标签
        labels = torch.arange(self.num_classes, device=self.device). \
            unsqueeze(0).repeat(self.num_proposals * self.multiple_sample, 1).flatten(0, 1)  # 创建类别标签张量

        # 对每张图像处理
        for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
                scores, box_pred, image_sizes
        )):
            # 获取得分最高的前k个候选框
            scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(
                self.num_proposals * self.multiple_sample, sorted=False)  # 保留前 num_proposals 个候选框
            labels_per_image = labels[topk_indices]
            box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
            box_pred_per_image = box_pred_per_image[topk_indices]

            # 非极大值抑制 (NMS)
            if self.use_nms:
                keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.6)
                box_pred_per_image = box_pred_per_image[keep]
                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]
            # print("num_classes:", self.num_classes)
            # print("scores shape:", scores_per_image.shape)
            # print("labels shape:", labels_per_image.shape)
            # print("unique labels:", labels_per_image.unique()[:10])
            # 调整未知类得分（如果 disentangled == 2）
            if self.disentangled == 2:
                scores_per_image[labels_per_image != self.num_classes - 1] *= 0.75
                scores_per_image[labels_per_image == self.num_classes - 1] *= 2

            # 保存结果
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_pred_per_image)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)

        return results
    # def inference(self, box_cls, box_objectness, box_pred, image_sizes):
    #     """
    #     Arguments:
    #         box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).    每个候选框的分类概率
    #             The tensor predicts the classification probability for each proposal.
    #         box_objectness (Tensor): tensors of shape (batch_size, num_proposals, 1).   候选框是否包含物体的概率
    #             The tensor predicts the objectness for each proposal.
    #         box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).   每个候选框对应的边界框回归预测
    #             The tensor predicts 4-vector (x,y,w,h) box
    #             regression values for every proposal
    #         image_sizes (List[torch.Size]): the input image sizes
    #
    #     Returns:
    #         results (List[Instances]): a list of #images elements.
    #     """
    #     assert len(box_cls) == len(image_sizes)
    #     results = []
    #
    #     if self.sampling_method == 'Random':
    #         multiple_sample = 1
    #     else:
    #         multiple_sample = self.multiple_sample
    #
    #     if self.disentangled == 0:
    #         scores = torch.sigmoid(box_cls)
    #     else:
    #         scores = torch.softmax(box_cls, dim=-1) * box_objectness # 计算分类得分
    #     labels = torch.arange(self.num_classes, device=self.device). \
    #         unsqueeze(0).repeat(self.num_proposals * multiple_sample, 1).flatten(0, 1) #创建一个从 0 到 80 的张量，表示所有类别的标签 并使用 repeat 生成与每个 proposal 对应的类别标签
    #
    #     for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
    #             scores, box_pred, image_sizes
    #     )):
    #         scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(
    #             self.num_proposals * multiple_sample, sorted=False)  # 得到得分最高的前k个候选框的索引
    #         labels_per_image = labels[topk_indices]
    #         box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
    #         box_pred_per_image = box_pred_per_image[topk_indices]
    #
    #         if self.use_nms: # 对每张图像的预测结果进行非极大值抑制，去除冗余的边界框
    #             keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.6)
    #             box_pred_per_image = box_pred_per_image[keep]
    #             scores_per_image = scores_per_image[keep]
    #             labels_per_image = labels_per_image[keep]
    #
    #         # rescale scores to accommodate score threshold
    #         if self.disentangled == 2: # 调整未知类得分
    #             scores_per_image[labels_per_image != self.num_classes-1] *= 0.75
    #             scores_per_image[labels_per_image == self.num_classes-1] *= 2
    #         # rescale scores to accommodate score threshold
    #         # if self.disentangled == 2: # 调整未知类得分
    #         #     scores_per_image[labels_per_image != self.num_classes-1] *= 0.75
    #         #     scores_per_image[labels_per_image == self.num_classes-1] *= 0.5
    #
    #         result = Instances(image_size)
    #         result.pred_boxes = Boxes(box_pred_per_image)
    #         result.scores = scores_per_image
    #         result.pred_classes = labels_per_image
    #         results.append(result)
    #
    #     return results
    def inference_rpn(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images, images_whwh = self.preprocess_image(batched_inputs)
        src = self.backbone(images.tensor)
        features = list()
        for f in self.in_features:
            feature = src[f]
            features.append(feature)
        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, src, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results = self.rpn_sample(batched_inputs, features, proposals, images)
            return results
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        # if do_postprocess:
        #     assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
        #     return self._postprocess(results, batched_inputs, images.image_sizes)
        # else:
        #     return results
    @torch.no_grad()
    def rpn_sample(self, batched_inputs, backbone_feats, proposals, images, clip_denoised=True, do_postprocess=True):
        # 1. RPN生成候选框
        # proposals, _ = self.rpn(images, backbone_feats)  # 使用RPN生成候选框
        # init_bboxes = [{'proposal_boxes': Boxes(p)} for p in proposals]
        # init_bboxes = [proposal.get("proposal_boxes").tensor for proposal in proposals]
        init_bboxes = proposals
        # 2. 动态检测头处理
        class_logits, objectness, pred_bboxes = self.head(backbone_feats, init_bboxes, init_features=None)

        # 3. 后处理
        results = self.inference(class_logits[-1], objectness[-1], pred_bboxes[-1], images.image_sizes)

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh
    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
