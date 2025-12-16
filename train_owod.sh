# Task 1

#update Weibull models of REW using known object labels in Task 1
python train_net.py --dist-url auto --num-gpus 4 --config-file configs/REW/rew_t1_mowod.yaml \
--task M-OWODB/rew_t1	OUTPUT_DIR output/rew/mowod_t1/weibull  MODEL.WEIGHTS output/rew/model_final.pth  \

#assign REW scores for unknown pseudo labels 
python train_net.py --rew-inference --resume --dist-url auto --num-gpus 4 \
--rew-config-file configs/REW/rew_t1_mowod.yaml --config-file configs/M-OWODB/t1.yaml \
--task M-OWODB/t1 OUTPUT_DIR output/rew/mowod_t1 WEIGHTS_DIR output/rew/mowod_t1/weibull MODEL.WEIGHTS output/rew/mowod_t1/model_0019999.pth


# Task 2

python train_net.py --dist-url auto --num-gpus 4 --config config/REW/rew_t2_sowod.yaml \
	OUTPUT_DIR training_dir/rew/sowod_t2  MODEL.WEIGHTS training_dir/rew/model_final.pth  \

python train_net.py --resume --eval-only --dist-url auto --num-gpus 4 --config config/MEPU-SOWOD/t2/train.yaml \
	DATASETS.TEST \(\"sowod_t2_train_and_ft\",\) OUTPUT_DIR training_dir/mepu-sowod/fs-t2-train OPENSET.OLN_INFERENCE True \
	OPENSET.INFERENCE_SELT_TRAIN True MODEL.WEIGHTS training_dir/mepu-sowod/fs-t1-self-train/model_final.pth  \

python tools/gen_pseudo_label_new.py --proposal_path training_dir/mepu-sowod/fs-t2-train/inference/inference_results.json \
	--data_path datasets/sowod --save_path datasets/sowod/Annotations/pseudo_label_st.json \
	--keep_type percent --percent_keep 0.3 --known_cls_num 40 --data_split t2_train_and_ft  \

python train_net.py --eval-only --inference-rew --resume --dist-url auto --num-gpus 4 \
	--config config/REW/rew_t2_sowod.yaml OUTPUT_DIR training_dir/rew/sowod_t2 \
	DATASETS.TEST \(\"sowod_train_t2_st\",\) \
  	OPENSET.OUTPUT_PATH_REW datasets/sowod/Annotations/pseudo_label_st.json \

python train_net.py --resume --dist-url auto --num-gpus 4 --config config/MEPU-SOWOD/t2/train.yaml \
	OUTPUT_DIR training_dir/mepu-sowod/fs-t2-train OPENSET.REW.GAMMA 4.0 \
	MODEL.WEIGHTS training_dir/mepu-sowod/fs-t1-self-train/model_final.pth \

python train_net.py --resume --dist-url auto --num-gpus 4 --config config/MEPU-SOWOD/t2/ft.yaml \
	OUTPUT_DIR training_dir/mepu-sowod/fs-t2-ft OPENSET.REW.GAMMA 4.0 \
	MODEL.WEIGHTS training_dir/mepu-sowod/fs-t2-train/model_final.pth \

# Task 3

python train_net.py --dist-url auto --num-gpus 4 --config config/REW/rew_t3_sowod.yaml \
	OUTPUT_DIR training_dir/rew/sowod_t3  MODEL.WEIGHTS training_dir/rew/model_final.pth  \

python train_net.py --resume --eval-only --dist-url auto --num-gpus 4 --config config/MEPU-SOWOD/t3/train.yaml \
	DATASETS.TEST \(\"sowod_t3_train_and_ft\",\) OUTPUT_DIR training_dir/mepu-sowod/fs-t3-train OPENSET.OLN_INFERENCE True \
	OPENSET.INFERENCE_SELT_TRAIN True MODEL.WEIGHTS training_dir/mepu-sowod/fs-t2-ft/model_final.pth  \

python tools/gen_pseudo_label_new.py --proposal_path training_dir/mepu-sowod/fs-t3-train/inference/inference_results.json \
    --data_path datasets/sowod --save_path datasets/sowod/Annotations/pseudo_label_st.json \
	--keep_type percent --percent_keep 0.3 --known_cls_num 60 --data_split t3_train_and_ft \

python train_net.py --eval-only --inference-rew --resume --dist-url auto --num-gpus 4 \
	--config config/REW/rew_t3_sowod.yaml OUTPUT_DIR training_dir/rew/sowod_t3 \
	DATASETS.TEST \(\"sowod_train_t3_st\",\) \
  	OPENSET.OUTPUT_PATH_REW datasets/sowod/Annotations/pseudo_label_st.json \

python train_net.py --resume --dist-url auto --num-gpus 4 --config config/MEPU-SOWOD/t3/train.yaml \
	OUTPUT_DIR training_dir/mepu-sowod/fs-t3-train OPENSET.REW.GAMMA 4.0 \
	MODEL.WEIGHTS training_dir/mepu-sowod/fs-t2-ft/model_final.pth \

python train_net.py --resume --dist-url auto --num-gpus 4 --config config/MEPU-SOWOD/t3/ft.yaml \
	OUTPUT_DIR training_dir/mepu-sowod/fs-t3-ft OPENSET.REW.GAMMA 4.0 \
	MODEL.WEIGHTS training_dir/mepu-sowod/fs-t3-train/model_final.pth \

# Task 4

python train_net.py --resume --dist-url auto --num-gpus 4 --config config/MEPU-SOWOD/t4/train.yaml \
	OUTPUT_DIR training_dir/mepu-sowod/fs-t4-train \
	MODEL.WEIGHTS training_dir/mepu-sowod/fs-t3-ft/model_final.pth \

python train_net.py --resume --dist-url auto --num-gpus 4 --config config/MEPU-SOWOD/t4/ft.yaml \
	OUTPUT_DIR training_dir/mepu-sowod/fs-t4-ft \
	MODEL.WEIGHTS training_dir/mepu-sowod/fs-t4-train/model_final.pth \







