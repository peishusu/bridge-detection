#!/usr/bin/bash
set -x
CONFIG=$1
GPUS=$2
# ：后面表示默认，代表 单机多卡 训练
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# 把当前脚本的上一级目录加入 PYTHONPATH，确保 Python 可以找到相关模块
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

# 运行 torch.distributed.launch 进行分布式训练。什么是分布式训练？？？
# 分布式训练是一种 并行化深度学习训练 的方法，它将 大规模的神经网络训练任务拆分到多个 GPU 或 计算节点，以加快训练速度、提升计算效率，并支持更大规模的数据和模型。
# torch.distributed.launch 默认采用数据并行
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    # --nproc_per_node=$GPUS 指定每个计算节点的 GPU 数量
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    # 去执行 train_dist.py $CONFIG --seed 0 --work-dir $WORK_DIR
    $(dirname "$0")/train_dist.py \
    $CONFIG \
    # 设定随机种子，保证实验可复现。？？？
      # 作用是让随机数生成器（RNG, Random Number Generator）在不同运行之间产生相同的随机数序列，确保数据加载、权重初始化等涉及随机性的部分，每次运行都相同。
    --seed 0 \
    # --launcher pytorch 指定 分布式训练的启动方式，这里使用 pytorch 的 torch.distributed.launch 方式
    # ${@:3} Shell 脚本中的第 3 个及之后的所有参数，
    --launcher pytorch ${@:3}
