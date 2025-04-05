
source activate mmrot

# 1. Split original images into multi-resolution groups
# python tools/split_code.py

# 2. Crop each group into patches (see tools/data/README.md for details)
# python tools/data/dota/split/img_split.py --base-json xxx

# 3. Start distributed training (2 GPUs)
CONFIG="configs/bridge_benchmark/oriented_rcnn_r50_fpn_2x_ImgFPN_oc.py"
# 指定使用 2 张 GPU 进行训练（分布式训练）。
GPUS=2
# 设定 训练过程的日志和模型存放目录。
WORK_DIR="bridge_train/oriented_rcnn_r50_fpn_2x_ImgFPN_dist"

# --work-dir $WORK_DIR 制定训练日志、模型checkpoint的路径
tools/dist_train.sh $CONFIG $GPUS --work-dir $WORK_DIR

# 4. (Optional) Multi-stage training with pre-trained backbone
# python tools/loadckpt_backbone.py, than change the 'load_from=' in your config file
# tools/dist_train.sh $CONFIG $GPUS --work-dir $WORK_DIR_loadfrom_xx