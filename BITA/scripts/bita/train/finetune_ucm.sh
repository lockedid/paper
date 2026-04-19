#!/bin/bash
# UCM Fine-tuning Script
# Port: 29523 (unique for UCM Fine-tuning)
# IMPORTANT: Make sure Stage 2 has completed and model config is updated before running this script

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=29523 \
    train.py \
    --cfg-path /home/lzc/code/BITA-main/BITA/BITA/project/bita/train/caption_ucm_ft.yaml
