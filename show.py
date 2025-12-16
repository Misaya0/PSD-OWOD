import json

import cv2
import torch
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import matplotlib.patches as patches
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from detectron2.detectron2.structures import BoxMode, Boxes, pairwise_iou
from fvcore.common.file_io import PathManager
import xml.etree.ElementTree as ET

def show_anns_qian(anns):
    if len(anns) == 0:
        return
    # sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    # img = np.ones((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 4))
    # img[:,:,3] = 0
    for ann in anns:
        # m = ann['segmentation']
        # color_mask = np.concatenate([np.random.random(3), [0.35]])
        # img[m] = color_mask

        # 绘制矩形框
        bbox = ann['bbox']  # 获取边界框 [xmin, ymin, width, height]
        xmin, ymin, weight, hight = bbox
        # 使用xmin, ymin, width, height来绘制矩形框
        rect = patches.Rectangle((xmin, ymin), weight, hight, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)  # 将矩形框添加到图像上

    # ax.imshow(img)
def show_anns_hou_numpy(anns):
    if len(anns) == 0:
        return
    # sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    anns = anns.tensor.cpu().numpy()
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
def show_anns_hou(anns):
    if len(anns) == 0:
        return
    # sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    # img = np.ones((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 4))
    # img[:,:,3] = 0
    for ann in anns:
        # m = ann['segmentation']
        # color_mask = np.concatenate([np.random.random(3), [0.35]])
        # img[m] = color_mask

        # 绘制矩形框
        bbox = ann['bbox']  # 获取边界框 [xmin, ymin, width, height]
        xmin, ymin, xmax, ymax = bbox
        # 使用xmin, ymin, width, height来绘制矩形框
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)  # 将矩形框添加到图像上

    # ax.imshow(img)




def generate_json(image_id, save_dir, box_results):
    """
    生成 JSON 文件，保存目标检测结果。

    Args:
        image_id (str): 图像的唯一标识符（文件名部分）。
        save_dir (str): JSON 文件的保存目录。
        box_results (list of dict): 检测框的列表，每个框包含 bbox 和 score。
            - bbox: [x, y, width, height]
            - score: float

    Example of box_results:
        [
            {"bbox": [50.0, 75.0, 120.0, 80.0], "score": 0.95},
            {"bbox": [200.0, 150.0, 60.0, 40.0], "score": 0.85}
        ]
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 构造 JSON 数据
    json_data = {
        "box_result": box_results
    }

    # 保存文件
    json_path = os.path.join(save_dir, f"{image_id}.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    print(f"JSON file saved: {json_path}")


model_type = "vit_h"
sam_checkpoint = "segment-anything/sam_vit_h_4b8939.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
sam.eval()
mask_generator = SamAutomaticMaskGenerator(sam)



imageset_dir = "/2T/gzj/OrthogonalDet-main/datasets/ImageSets/Main/M-OWODB/t4.txt"
with PathManager.open(imageset_dir) as f:
    fileids = [file_name.strip() for file_name in f.readlines()]

annotation_dirname = "/2T/gzj/OrthogonalDet-main/datasets/Annotations"
jpeg_dirname = "/2T/gzj/OrthogonalDet-main/datasets/JPEGImages"

sam_dir = "/2T/gzj/OrthogonalDet-main/datasets/SAM_H"

tr_sz = 5
tr_iou = 0.9
for image_id in fileids:
    # image = cv2.imread('/2T/gzj/OrthogonalDet-main/datasets/JPEGImages/' + image_id + '.jpg')
    image = cv2.imread('/2T/gzj/OrthogonalDet-main/datasets/JPEGImages/000289.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    start_time = time.time()
    masks = mask_generator.generate(image)
    print("--- %s seconds ---" % (time.time() - start_time))
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns_qian(masks)
    plt.axis('off')
    plt.show()

    # print(masks)
    # anno_file = os.path.join(annotation_dirname, image_id + ".xml")
    # jpeg_file = os.path.join(jpeg_dirname, image_id + ".jpg")
    anno_file = os.path.join(annotation_dirname, "000289.xml")
    jpeg_file = os.path.join(jpeg_dirname, "000289.jpg")
    with PathManager.open(anno_file) as f:
        tree = ET.parse(f)

    r = {
        "file_name": jpeg_file,
        "image_id": image_id,
        "height": int(tree.findall("./size/height")[0].text),
        "width": int(tree.findall("./size/width")[0].text),
    }
    instances = []

    for obj in tree.findall("object"):
        cls_name = obj.find("name").text
        bbox = obj.find("bndbox")
        bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
        bbox[0] -= 1.0
        bbox[1] -= 1.0

        instances.append(
            {"category_id": 2, "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
        )

    box_results = []
    for mask in masks:
        box_results.append({"bbox": mask['bbox'], "score": mask['stability_score']})


    gt_boxes = Boxes(torch.tensor([item['bbox'] for item in instances], device=device))

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns_hou_numpy(gt_boxes)
    plt.axis('off')
    plt.show()


    sam_data = box_results
    sam_data = [item for item in sam_data if item['bbox'][2] >= tr_sz and item['bbox'][3] >= tr_sz]
    sam_boxes_list = [[
        item['bbox'][0], item['bbox'][1],
        item['bbox'][0] + item['bbox'][2],
        item['bbox'][1] + item['bbox'][3]]
        for item in sam_data]  # xyxy format
    sam_score_list = [item['score'] for item in sam_data]
    sam_boxes = Boxes(torch.tensor(sam_boxes_list, device=device))
    ious, _ = pairwise_iou(sam_boxes, gt_boxes).max(dim=1)
    boxes_results =[]
    for boxe, iou, score in zip(sam_boxes_list, ious, sam_score_list):
        if (iou >= tr_iou):
            continue
        instances.append({
            "category_id": 80,
            "bbox": [boxe[0], boxe[1],
                     boxe[2], boxe[3]],  # xyxy
            "bbox_mode": BoxMode.XYXY_ABS,
            'score': score
        })
        boxes_results.append({"bbox": [boxe[0], boxe[1], boxe[2], boxe[3]], "score": score})

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns_hou(boxes_results)
    plt.axis('off')
    plt.show()









