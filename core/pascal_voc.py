import json

import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union
from fvcore.common.file_io import PathManager
import itertools
import logging
import torch
from tqdm import tqdm
from torchvision.ops import nms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from detectron2.detectron2 import model_zoo
from detectron2.detectron2.engine import DefaultPredictor
from detectron2.detectron2.config import get_cfg
from detectron2.detectron2.utils.visualizer import Visualizer


from detectron2.detectron2.structures import BoxMode, Boxes, pairwise_iou
from detectron2.detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.detectron2.structures import BoxMode

__all__ = ["load_voc_instances", "register_pascal_voc"]

VOC_CLASS_NAMES_COCOFIED = [
    "airplane", "dining table", "motorcycle",
    "potted plant", "couch", "tv"
]

BASE_VOC_CLASS_NAMES = [
    "aeroplane", "diningtable", "motorbike",
    "pottedplant", "sofa", "tvmonitor"
]

UNK_CLASS = ["unknown"]

VOC_COCO_CLASS_NAMES = {}

VOC_CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

T2_CLASS_NAMES = [
    "truck", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "microwave", "oven", "toaster", "sink", "refrigerator"
]

T3_CLASS_NAMES = [
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake"
]

T4_CLASS_NAMES = [
    "bed", "toilet", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl"
]

VOC_COCO_CLASS_NAMES["IOD"] = tuple(itertools.chain(VOC_CLASS_NAMES, UNK_CLASS))
VOC_COCO_CLASS_NAMES["M-OWODB"] = tuple(
    itertools.chain(VOC_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, T4_CLASS_NAMES, UNK_CLASS))

T1_CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bus", "car",
    "cat", "cow", "dog", "horse", "motorbike", "sheep", "train",
    "elephant", "bear", "zebra", "giraffe", "truck", "person"
]

T2_CLASS_NAMES = [
    "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "chair", "diningtable",
    "pottedplant", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "microwave", "oven", "toaster", "sink",
    "refrigerator", "bed", "toilet", "sofa"
]

T3_CLASS_NAMES = [
    "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"
]

T4_CLASS_NAMES = [
    "laptop", "mouse", "remote", "keyboard", "cell phone", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "tvmonitor", "bottle"
]

VOC_COCO_CLASS_NAMES["S-OWODB"] = tuple(
    itertools.chain(T1_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, T4_CLASS_NAMES, UNK_CLASS))

def show_anns_hou_numpy(anns, path):
    if len(anns) == 0:
        return
    # sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
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
    plt.axis('off')
    plt.show()
    # ax.imshow(img)

def load_one_sam_data(num, instances, sam_file_root, image_id, tr_sz=5, tr_iou=0.9):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gt_boxes = Boxes(torch.tensor([item['bbox'] for item in instances], device=device))
    with open(os.path.join(sam_file_root, f'{image_id}.json'), 'r') as f:
        sam_data = json.load(f)['box_result']
    sam_data = [item for item in sam_data if item['bbox'][2]>= tr_sz and item['bbox'][3]>= tr_sz]#过滤掉小目标
    sam_boxes_list = [[
                        item['bbox'][0],item['bbox'][1],
                        item['bbox'][0]+item['bbox'][2],
                        item['bbox'][1]+item['bbox'][3]]
                        for item in sam_data]  # xyxy format
    sam_score_list = [item['score'] for item in sam_data]
    sam_boxes = torch.tensor(sam_boxes_list, device=device, dtype=torch.float)
    sam_scores = torch.tensor(sam_score_list, device=device, dtype=torch.float)

    # image = cv2.imread("/2T/gzj/OrthogonalDet-main/datasets/JPEGImages/" + image_id + ".jpg")
    # plt.figure(figsize=(20, 20))
    # plt.imshow(image)
    # show_anns_hou_numpy(sam_boxes)
    # plt.axis('off')
    # plt.show()

    # 筛选低得分框
    valid_indices = sam_scores >= 0.968
    sam_boxes = sam_boxes[valid_indices]
    sam_scores = sam_scores[valid_indices]
    if sam_boxes.numel() != 0:
        # 对 SAM 生成的框进行 NMS 去冗余
        keep_indices = nms(sam_boxes, sam_scores, 0.5)  # 只保留非冗余框
        sam_boxes = sam_boxes[keep_indices]
        sam_scores = sam_scores[keep_indices]

        if sam_scores.shape[0] > 40:  # 进行再一次筛选
            valid_indices = sam_scores >= 0.98
            sam_boxes = sam_boxes[valid_indices]
            sam_scores = sam_scores[valid_indices]

    sam_boxes_obj = Boxes(sam_boxes)

    # plt.figure(figsize=(20, 20))
    # plt.imshow(image)
    # show_anns_hou_numpy(sam_boxes)
    # plt.axis('off')
    # plt.show()
    if sam_boxes.numel() != 0 and len(instances) != 0:
        ious, _ = pairwise_iou(sam_boxes_obj, gt_boxes).max(dim=1)
        for boxe, iou, score in zip(sam_boxes.tolist(), ious, sam_scores.tolist()):
            if(iou >= tr_iou):
                continue
            instances.append({
                "category_id": 80,
                "bbox": [boxe[0], boxe[1],
                        boxe[2], boxe[3]],  # xyxy
                "bbox_mode": BoxMode.XYXY_ABS,
                # 'soft_labels': score
                'score': score
            })

    # num_instances = 0
    # for instance in instances:
    #     if instance['category_id'] == 80:
    #         num_instances += 1
    # if num_instances == 0:
    #     instances.append({
    #         "category_id": 80,
    #         "bbox": [sam_boxes_list[0][0], sam_boxes_list[0][1],
    #                 sam_boxes_list[0][2], sam_boxes_list[0][3]],  # xyxy
    #         "bbox_mode": BoxMode.XYXY_ABS,
    #         'score': sam_score_list[0]
    #     })
    # num_unkown_instances = 0
    # for instance in instances:
    #     if instance['category_id'] == 80:
    #         num_unkown_instances += 1
    # print(image_id + ".json文件经过过滤后还有" + str(num_unkown_instances) + "个未知物体")
    # num.append(num_unkown_instances)
    # if num_unkown_instances > 80 or image_id == '000000076654':
    #     print(f"num_unkown_instances: {num_unkown_instances}"+"!!!!!!!!!!")
    #     image = cv2.imread("/2T/gzj/OrthogonalDet-main/datasets/JPEGImages/" + image_id + ".jpg")
    #
    #     plt.figure(figsize=(20, 20))
    #     plt.imshow(image)
    #     show_anns_hou_numpy(sam_boxes)
    #     plt.axis('off')
    #     plt.show()
    #     print(f"num_unkown_instances: {num_unkown_instances}"+"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    return num, instances

def load_voc_instances(dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]], cfg):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str_)

    # Needs to read many small annotation files. Makes sense at local
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    dicts = []
    num = []
    # PROB and CAT convert image id to int before iterating over image ids
    # RandBox uses COCO's loader, which implicitly converts image id to int
    ids = []
    id2fileids = {}
    for fileid in fileids:
        id = int(fileid.split('.')[0])
        ids.append(id)
        id2fileids[id] = fileid

    # filter instances
    if len(cfg.DATASETS.TEST) != 0:
        if cfg.TEST.MASK == 1:
            allowed_class = list(range(0, cfg.TEST.PREV_INTRODUCED_CLS+cfg.TEST.CUR_INTRODUCED_CLS))
        else:
            allowed_class = list(range(cfg.TEST.PREV_INTRODUCED_CLS, cfg.TEST.PREV_INTRODUCED_CLS+cfg.TEST.CUR_INTRODUCED_CLS))
    else:
        allowed_class = list(range(0, len(class_names)))
    for id in tqdm(ids):
        fileid = id2fileids[id]
        # if fileid == '2008_002709':
        #     print("hahhahah")
    # for fileid in id2fileids.values():
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        try:
            with PathManager.open(anno_file) as f:
                tree = ET.parse(f)
        except:
            logger = logging.getLogger(__name__)
            logger.info('Not able to load: ' + anno_file + '. Continuing without aboarting...')
            continue

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if cls in VOC_CLASS_NAMES_COCOFIED:
                cls = BASE_VOC_CLASS_NAMES[VOC_CLASS_NAMES_COCOFIED.index(cls)]
            if cfg.TEST.MASK and ('test' not in split):
                if class_names.index(cls) not in allowed_class:
                    continue
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instances.append(
                {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS, "soft_label": 1.0}
            )
        #使用SAM作为伪标签！！！！！
        if 'ft' in split:# t1_ft t2_ft t3_ft t4_ft
            # instances = remove_unseen(instances, cur_seen)
            num, instances = load_one_sam_data(num, instances, os.path.join(dirname, 'SAM_H'), fileid, tr_sz=25, tr_iou=0.9)
        elif 'test' in split:
            # instances = rename_unseen_to_unknown(instances, cur_seen)
            # print('test!!!')
            pass
        elif 'merged_and_deduplicated' in split:
            pass
        elif 'rew' in split:
            pass
        else: #t1 t2 t3 t4
            # instances = remove_unseen(instances, cur_seen)
            # instances = remove_previous(instances, prev_num)
            num, instances = load_one_sam_data(num, instances, os.path.join(dirname, 'SAM_H'), fileid, tr_sz=25, tr_iou=0.9)
        r["annotations"] = instances
        dicts.append(r)
    # for dict in dicts:
    #     a = 0
    #     for instance in dict['annotations']:
    #         if instance['category_id'] == 80:
    #             a += 1
    #     if a == 0:
    #         print(dict['image_id'])
    return dicts


def register_pascal_voc(name, dirname, super_split, split, cfg, year=2007):
    # if "voc_coco" in name:
    #     class_names = VOC_COCO_CLASS_NAMES
    # else:
    #     class_names = tuple(VOC_CLASS_NAMES)
    class_names = VOC_COCO_CLASS_NAMES[super_split]
    DatasetCatalog.register(name, lambda: load_voc_instances(dirname, split, class_names, cfg))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )
