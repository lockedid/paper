#!/bin/bash
# NWPU Stage 2 Pretraining Script
# Port: 29512 (unique for NWPU Stage 2)
# IMPORTANT: Make sure Stage 1 has completed before running this script

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=29512 \
    train.py \
    --cfg-path /home/lzc/code/BITA-main/BITA/BITA/project/bita/train/pretrain_stage2_nwpu.yaml
