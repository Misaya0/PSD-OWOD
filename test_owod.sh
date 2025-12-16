#!/bin/bash

BENCHMARK=${BENCHMARK:-"M-OWODB"}  # M-OWODB or S-OWODB
PORT=${PORT:-"50210"}

# if raise error, change num_gpus to 1
#if [ $BENCHMARK == "M-OWODB" ]; then
#  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:50210 --task M-OWODB/t1 --config-file configs/M-OWODB/t1.yaml --eval-only MODEL.WEIGHTS output/M-OWODB/model_0019999.pth
#
#  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:50210 --task M-OWODB/t2_ft --config-file configs/M-OWODB/t2_ft.yaml --eval-only MODEL.WEIGHTS output/M-OWODB/model_0049999.pth

  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:50210 --task M-OWODB/t3_ft --config-file configs/M-OWODB/t3_ft.yaml --eval-only MODEL.WEIGHTS output/M-OWODB/model_0079999.pth

  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:50210 --task M-OWODB/t4_ft --config-file configs/M-OWODB/t4_ft.yaml --eval-only MODEL.WEIGHTS output/M-OWODB/model_0109999.pth
#else
#  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:50210 --task M-OWODB/t1 --config-file configs/M-OWODB/t1.yaml --eval-only MODEL.WEIGHTS output/M-OWODB/model_0039999.pth
#
#  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:50210 --task M-OWODB/t2_ft --config-file configs/M-OWODB/t2_ft.yaml --eval-only MODEL.WEIGHTS output/M-OWODB/model_0069999.pth
#
#  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:50210 --task M-OWODB/t3_ft --config-file configs/M-OWODB/t3_ft.yaml --eval-only MODEL.WEIGHTS output/M-OWODB/model_0099999.pth
#
#  python train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:50210 --task M-OWODB/t4_ft --config-file configs/M-OWODB/t4_ft.yaml --eval-only MODEL.WEIGHTS output/M-OWODB/model_0129999.pth
#fi