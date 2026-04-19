#!/bin/bash
# RSICD Stage 1 Pretraining Script
# Port: 29501 (unique for RSICD Stage 1)

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=29501 \
    train.py \
    --cfg-path /home/wj/code/BITA-main/BITA/BITA/project/bita/train/pretrain_stage1_rsicd.yaml
