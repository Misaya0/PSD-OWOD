# unsupervised pretraining of REW using COCO train2017
# pretrained SoCo backbone can be downloaded in https://github.com/hologerry/SoCo
PORT=${PORT:-"50210"}
python train_net.py --resume --dist-url tcp://127.0.0.1:${PORT} --num-gpus 4 --config-file configs/REW/rew_pretrain.yaml \
	OUTPUT_DIR output/rew \
