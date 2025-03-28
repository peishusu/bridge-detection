
source activate mmrot

## 1. refer to tools\split_code.py, process the original large-size images to multi-resolution groups
## 2. refer to tools\data\README.md, crop images into patches of each group
## 3. begin train:

tools/dist_train.sh /scratch/luojunwei/WorldBridge/Code/mmrotate_bridge/configs/bridge_benchmark/oriented_rcnn_r50_fpn_2x_ImgFPN_oc.py 2 \
                    --work-dir bridge_train/oriented_rcnn_r50_fpn_2x_ImgFPN_dist

## 4. tools\loadckpt_backbone.py for multi-ImgFPN-stage train (depends on your dataset), continue train
# tools/dist_train.sh /scratch/luojunwei/WorldBridge/Code/mmrotate_bridge/configs/bridge_benchmark/oriented_rcnn_r50_fpn_2x_ImgFPN_oc.py 2 \
#                     --work-dir bridge_train/oriented_rcnn_r50_fpn_2x_ImgFPN_dist_load-d2-backbone


# 1. Split original images into multi-resolution groups
# python tools/split_code.py

# 2. Crop each group into patches (see tools/data/README.md for details)
# python tools/data/dota/split/img_split.py --base-json xxx

# 3. Start distributed training (2 GPUs)
CONFIG="configs/bridge_benchmark/oriented_rcnn_r50_fpn_2x_ImgFPN_oc.py"
GPUS=2
WORK_DIR="bridge_train/oriented_rcnn_r50_fpn_2x_ImgFPN_dist"
tools/dist_train.sh $CONFIG $GPUS --work-dir $WORK_DIR

# 4. (Optional) Multi-stage training with pre-trained backbone
# python tools/loadckpt_backbone.py, than change the 'load_from=' in your config file
# tools/dist_train.sh $CONFIG $GPUS --work-dir $WORK_DIR_loadfrom_xx --resume-from [新检查点路径]