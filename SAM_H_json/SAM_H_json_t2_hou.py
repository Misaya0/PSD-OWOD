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

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        # m = ann['segmentation']
        # color_mask = np.concatenate([np.random.random(3), [0.35]])
        # img[m] = color_mask

        # 绘制矩形框
        bbox = ann['bbox']  # 获取边界框 [xmin, ymin, width, height]
        xmin, ymin, width, height = bbox
        # 使用xmin, ymin, width, height来绘制矩形框
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)  # 将矩形框添加到图像上

    ax.imshow(img)
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

from fvcore.common.file_io import PathManager
import xml.etree.ElementTree as ET
from detectron2.detectron2.structures import BoxMode

imageset_dir = "/2T/gzj/OrthogonalDet-main/datasets/ImageSets/Main/t2_hou.txt"
with PathManager.open(imageset_dir) as f:
    fileids = [file_name.strip() for file_name in f.readlines()]

annotation_dirname = "/2T/gzj/OrthogonalDet-main/datasets/Annotations"
jpeg_dirname = "/2T/gzj/OrthogonalDet-main/datasets/JPEGImages"

sam_dir = "/2T/gzj/OrthogonalDet-main/datasets/SAM_H"
for image_id in fileids:
    image = cv2.imread('/2T/gzj/OrthogonalDet-main/datasets/JPEGImages/' + image_id + '.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    start_time = time.time()
    masks = mask_generator.generate(image)
    print("--- %s seconds ---" % (time.time() - start_time))
    # plt.figure(figsize=(20, 20))
    # plt.imshow(image)
    # show_anns(masks)
    # plt.axis('off')
    # plt.show()

    # print(masks)
    anno_file = os.path.join(annotation_dirname, image_id + ".xml")
    jpeg_file = os.path.join(jpeg_dirname, image_id + ".jpg")

    box_results = []
    for mask in masks:
        box_results.append({"bbox": mask['bbox'], "score": mask['stability_score']})
    generate_json(image_id, sam_dir, box_results)








