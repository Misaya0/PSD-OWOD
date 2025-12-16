import json

import cv2
import torch
# from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import matplotlib.patches as patches
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry,SamPredictor
from segment_anything.build_sam import build_sam_vit_h
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


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


def read_yolo_annotations(file_path):
    boxes = []
    with open(file_path, 'r') as file:
        for line in file:
            # 分割每一行的数据
            parts = line.strip().split()

            # 提取边界框坐标
            x_min = float(parts[1])
            y_min = float(parts[2])
            x_max = float(parts[3])
            y_max = float(parts[4])

            # 将信息存储在字典中
            box_info = [x_min, y_min, x_max, y_max]
            # 将字典添加到boxes列表中
            boxes.append(box_info)
    return torch.tensor(boxes)

image_name = "097_172830453"
image_path = "/home/hebut-02/common/gzj/ultralytics-main/datasets/data/train2017/"+ image_name +".jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
annotations_path = "/home/hebut-02/common/gzj/ultralytics-main/runs/detect/predict5/labels/"+ image_name + ".txt"
boxes = read_yolo_annotations(annotations_path)
# annotations_path = "/2T/gzj/OrthogonalDet-main/road/annotations/"+ image_name +".xml"
# with PathManager.open(annotations_path) as f:
#     tree = ET.parse(f)

# instances = []
# boxes = []
# for obj in tree.findall("object"):
#     cls_name = obj.find("name").text
#     bbox = obj.find("bndbox")
#     bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
#     bbox[0] -= 1.0
#     bbox[1] -= 1.0
#
#     instances.append(
#         {"category_id": 2, "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
#     )
# boxes = torch.tensor([item['bbox'] for item in instances])
# boxes = np.array(boxes,dtype=np.int32)
# box = np.array(instances[0]['bbox'])

model_type = "vit_h"
sam_checkpoint = "segment-anything/sam_vit_h_4b8939.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

sam_model = build_sam_vit_h(checkpoint=sam_checkpoint)
predictor = SamPredictor(sam_model)

start_time = time.time()
predictor.set_image(image)
transformed_boxes = predictor.transform.apply_boxes_torch(boxes, image.shape[:2])
masks, scores, logits = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False
)
end_time = time.time()
print(end_time - start_time)
print(masks.shape)

plt.figure(figsize=(10, 10))
plt.imshow(image)
for mask in masks:
    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
for box in boxes:
    show_box(box.cpu().numpy(), plt.gca())
plt.axis('off')
plt.show()



# imageset_dir = "/2T/gzj/OrthogonalDet-main/datasets/ImageSets/Main/M-OWODB/t4.txt"
# with PathManager.open(imageset_dir) as f:
#     fileids = [file_name.strip() for file_name in f.readlines()]
#
# annotation_dirname = "/2T/gzj/OrthogonalDet-main/datasets/Annotations"
# jpeg_dirname = "/2T/gzj/OrthogonalDet-main/datasets/JPEGImages"
#
# sam_dir = "/2T/gzj/OrthogonalDet-main/datasets/SAM_H"
#
# tr_sz = 5
# tr_iou = 0.9
# for image_id in fileids:
#     # image = cv2.imread('/2T/gzj/OrthogonalDet-main/datasets/JPEGImages/' + image_id + '.jpg')
#     image = cv2.imread('/2T/gzj/OrthogonalDet-main/datasets/JPEGImages/000289.jpg')
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     start_time = time.time()
#     masks = mask_generator.generate(image)
#     print("--- %s seconds ---" % (time.time() - start_time))
#     plt.figure(figsize=(20, 20))
#     plt.imshow(image)
#     show_anns_qian(masks)
#     plt.axis('off')
#     plt.show()
#
#     # print(masks)
#     # anno_file = os.path.join(annotation_dirname, image_id + ".xml")
#     # jpeg_file = os.path.join(jpeg_dirname, image_id + ".jpg")
#     anno_file = os.path.join(annotation_dirname, "000289.xml")
#     jpeg_file = os.path.join(jpeg_dirname, "000289.jpg")
#     with PathManager.open(anno_file) as f:
#         tree = ET.parse(f)
#
#     r = {
#         "file_name": jpeg_file,
#         "image_id": image_id,
#         "height": int(tree.findall("./size/height")[0].text),
#         "width": int(tree.findall("./size/width")[0].text),
#     }
#     instances = []
#
#     for obj in tree.findall("object"):
#         cls_name = obj.find("name").text
#         bbox = obj.find("bndbox")
#         bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
#         bbox[0] -= 1.0
#         bbox[1] -= 1.0
#
#         instances.append(
#             {"category_id": 2, "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
#         )
#
#     box_results = []
#     for mask in masks:
#         box_results.append({"bbox": mask['bbox'], "score": mask['stability_score']})
#
#
#     gt_boxes = Boxes(torch.tensor([item['bbox'] for item in instances], device=device))
#
#     plt.figure(figsize=(20, 20))
#     plt.imshow(image)
#     show_anns_hou_numpy(gt_boxes)
#     plt.axis('off')
#     plt.show()
#
#
#     sam_data = box_results
#     sam_data = [item for item in sam_data if item['bbox'][2] >= tr_sz and item['bbox'][3] >= tr_sz]
#     sam_boxes_list = [[
#         item['bbox'][0], item['bbox'][1],
#         item['bbox'][0] + item['bbox'][2],
#         item['bbox'][1] + item['bbox'][3]]
#         for item in sam_data]  # xyxy format
#     sam_score_list = [item['score'] for item in sam_data]
#     sam_boxes = Boxes(torch.tensor(sam_boxes_list, device=device))
#     ious, _ = pairwise_iou(sam_boxes, gt_boxes).max(dim=1)
#     boxes_results =[]
#     for boxe, iou, score in zip(sam_boxes_list, ious, sam_score_list):
#         if (iou >= tr_iou):
#             continue
#         instances.append({
#             "category_id": 80,
#             "bbox": [boxe[0], boxe[1],
#                      boxe[2], boxe[3]],  # xyxy
#             "bbox_mode": BoxMode.XYXY_ABS,
#             'score': score
#         })
#         boxes_results.append({"bbox": [boxe[0], boxe[1], boxe[2], boxe[3]], "score": score})
#
#     plt.figure(figsize=(20, 20))
#     plt.imshow(image)
#     show_anns_hou(boxes_results)
#     plt.axis('off')
#     plt.show()









