#!/bin/bash
# RSICD Stage 2 Pretraining Script
# Port: 29502 (unique for RSICD Stage 2)
# IMPORTANT: Make sure Stage 1 has completed before running this script

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=29502 \
    train.py \
    --cfg-path /home/wj/code/BITA-main/BITA/BITA/project/bita/train/pretrain_stage2_rsicd.yaml
