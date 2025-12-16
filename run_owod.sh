#!/bin/bash

BENCHMARK=${BENCHMARK:-"M-OWODB"}  # M-OWODB or S-OWODB
PORT=${PORT:-"50210"}

#if [ $BENCHMARK == "M-OWODB" ]; then
#  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:50210 --task M-OWODB/t1 --config-file configs/M-OWODB/t1.yaml

#  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:50210 --task M-OWODB/t2 --config-file configs/M-OWODB/t2.yaml --resume MODEL.WEIGHTS output/M-OWODB/model_0019999.pth
#
#  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:50210 --task M-OWODB/t2_ft --config-file configs/M-OWODB/t2_ft.yaml --resume MODEL.WEIGHTS output/M-OWODB/model_0034999.pth

  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:50210 --task M-OWODB/t3 --config-file configs/M-OWODB/t3.yaml --resume MODEL.WEIGHTS output/M-OWODB/model_0049999.pth

  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:50210 --task M-OWODB/t3_ft --config-file configs/M-OWODB/t3_ft.yaml --resume MODEL.WEIGHTS output/M-OWODB/model_0064999.pth

  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:50210 --task M-OWODB/t4 --config-file configs/M-OWODB/t4.yaml --resume MODEL.WEIGHTS output/M-OWODB/model_0079999.pth

  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:50210 --task M-OWODB/t4_ft --config-file configs/M-OWODB/t4_ft.yaml --resume MODEL.WEIGHTS output/M-OWODB/model_0094999.pth
#else
#  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:50210 --task M-OWODB/t1 --config-file configs/M-OWODB/t1.yaml
#
#  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:50210 --task M-OWODB/t2 --config-file configs/M-OWODB/t2.yaml --resume MODEL.WEIGHTS output/M-OWODB/model_0039999.pth
#
#  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:50210 --task M-OWODB/t2_ft --config-file configs/M-OWODB/t2_ft.yaml --resume MODEL.WEIGHTS output/M-OWODB/model_0054999.pth
#
#  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:50210 --task M-OWODB/t3 --config-file configs/M-OWODB/t3.yaml --resume MODEL.WEIGHTS output/M-OWODB/model_0069999.pth
#
#  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:50210 --task M-OWODB/t3_ft --config-file configs/M-OWODB/t3_ft.yaml --resume MODEL.WEIGHTS output/M-OWODB/model_0084999.pth
#
#  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:50210 --task M-OWODB/t4 --config-file configs/M-OWODB/t4.yaml --resume MODEL.WEIGHTS output/M-OWODB/model_0099999.pth
#
#  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:50210 --task M-OWODB/t4_ft --config-file configs/M-OWODB/t4_ft.yaml --resume MODEL.WEIGHTS output/M-OWODB/model_00114999.pth
#fi