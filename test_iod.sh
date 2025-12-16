#!/bin/bash

SETUP=${SETUP:-"10_10"}  # 10_10 or 15_5 or 19_1
PORT=${PORT:-"50210"}

#python train_net.py --num-gpus 2 --dist-url tcp://127.0.0.1:${PORT} --task IOD/trainval --config-file configs/IOD/${SETUP}_0.yaml --eval-only MODEL.WEIGHTS output/${BENCHMARK}/model_0017999.pth

python train_net.py --num-gpus 2 --dist-url tcp://127.0.0.1:${PORT} --task IOD/trainval --config-file configs/IOD/${SETUP}_1.yaml --eval-only MODEL.WEIGHTS output/IOD_${SETUP}/model_0019999.pth

#python train_net.py --num-gpus 2 --dist-url tcp://127.0.0.1:${PORT} --task IOD/trainval --config-file configs/IOD/${SETUP}_ft.yaml --eval-only MODEL.WEIGHTS output/IOD_${SETUP}/model_0019999.pth