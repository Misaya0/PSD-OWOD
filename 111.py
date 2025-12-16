# from segment_anything import SamPredictor, sam_model_registry
# import cv2
# import numpy as np
# model_type = 'vit_h'
# sam = sam_model_registry[model_type](checkpoint="segment-anything/sam_vit_h_4b8939.pth")
# predictor = SamPredictor(sam)
#
# image_path = "segment-anything/123456.jpg"
# image_np = cv2.imread(image_path)
# image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
#
# predictor.set_image(image_np)
# masks, _, _ = predictor.predict()
#
# print("masks:", masks)

# from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
# import cv2
# model_type = 'vit_h'
# sam = sam_model_registry[model_type](checkpoint="segment-anything/sam_vit_h_4b8939.pth")
# mask_generator = SamAutomaticMaskGenerator(sam)
# image_path = "segment-anything/123456.jpg"
# image_np = cv2.imread(image_path)
# image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
#
# masks = mask_generator.generate(image_np)
# print("Done")


# import torch
# from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
# import cv2
# import time
# import numpy as np
# from matplotlib import pyplot as plt
# model_type = "vit_t"
# sam_checkpoint = "MobileSAM-master/weights/mobile_sam.pt"
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# mobile_sam.to(device=device)
# mobile_sam.eval()
#
# predictor = SamPredictor(mobile_sam)
#
# image_path = "MobileSAM-master/123456.jpg"
# image_np = cv2.imread(image_path)
# image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
# predictor.set_image(image_np)
# start_time = time.time()
# masks, _, _ = predictor.predict()
# print("--- %s seconds ---" % (time.time() - start_time))
#
# # 创建一个示例的 [3, w, h] 形状的布尔掩码数组
# w, h = image_np.shape[0], image_np.shape[1]  # 图像的宽度和高度
#
# # 创建一个空白的 RGB 图像
# rgb_mask = np.zeros((w, h, 3), dtype=np.uint8)
#
# # 将每个通道映射到不同的颜色
# rgb_mask[masks[0]] = [255, 0, 0]  # 类别 1 显示为红色
# rgb_mask[masks[1]] = [0, 255, 0]  # 类别 2 显示为绿色
# rgb_mask[masks[2]] = [0, 0, 255]  # 类别 3 显示为蓝色
#
# # 可视化结果
# plt.imshow(rgb_mask)
# plt.axis('off')  # 不显示坐标轴
# plt.show()
# print("Done")



import cv2
import torch
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import matplotlib.patches as patches
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
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


model_type = "vit_t"
sam_checkpoint = "MobileSAM-master/weights/mobile_sam.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()
mask_generator = SamAutomaticMaskGenerator(mobile_sam)

image = cv2.imread('MobileSAM-master/654321.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

start_time = time.time()
masks = mask_generator.generate(image)
print("--- %s seconds ---" % (time.time() - start_time))
# plt.figure(figsize=(20,20))
# plt.imshow(image)
# show_anns(masks)
# plt.axis('off')
# plt.show()

print(masks)

