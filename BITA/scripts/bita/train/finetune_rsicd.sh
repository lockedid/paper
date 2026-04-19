#!/bin/bash
# RSICD Fine-tuning Script
# Port: 29503 (unique for RSICD Fine-tuning)
# IMPORTANT: Make sure Stage 2 has completed and model config is updated before running this script

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=29503 \
    train.py \
    --cfg-path /home/wj/code/BITA-main/BITA/BITA/project/bita/train/caption_rsicd_ft.yaml
