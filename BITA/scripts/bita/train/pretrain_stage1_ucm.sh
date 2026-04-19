#!/bin/bash
# UCM Stage 1 Pretraining Script
# Port: 29521 (unique for UCM Stage 1)

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=29521 \
    train.py \
    --cfg-path /home/lzc/code/BITA-main/BITA/BITA/project/bita/train/pretrain_stage1_ucm.yaml
