
source activate mmrot

## please refer to  tools\split_code.py and tools\loadckpt_backbone.py for multi-ImgFPN-stage train (depends on your dataset)

tools/dist_train.sh /scratch/luojunwei/WorldBridge/Code/mmrotate_bridge/configs/bridge_benchmark/oriented_rcnn_r50_fpn_2x_ImgFPN_oc.py 2 \
                    --work-dir bridge_train/oriented_rcnn_r50_fpn_2x_ImgFPN_dist

