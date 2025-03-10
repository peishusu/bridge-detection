
source activate mmrot


python tools/test.py --config configs/bridge_benchmark/oriented_rcnn_r50_fpn_2x_ImgFPN_oc.py\
                     --checkpoint xxx\
                     --work-dir test_result/World_bridge_new_full_test/oriented_rcnn_r50_fpn_2x_ImgFPN\
                     --eval mAP