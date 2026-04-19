#!/bin/bash
# BITA 模型优化后快速启动脚本
# 用途：清理旧输出并开始新训练

set -e  # 遇到错误立即退出

echo "======================================================================"
echo "BITA 模型架构优化 - 快速启动脚本"
echo "======================================================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 确认是否继续
echo -e "${YELLOW}⚠️  警告：此脚本将删除以下目录的所有输出文件：${NC}"
echo "  - output/pretrain_stage1_rsicd/"
echo "  - output/pretrain_stage2_rsicd/"
echo "  - output/caption_rsicd_ft/"
echo ""
read -p "是否继续？(y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}❌ 操作已取消${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}📦 步骤 1: 清理旧输出文件${NC}"
echo "----------------------------------------------------------------------"

# 清理 Stage 1
if [ -d "output/pretrain_stage1_rsicd" ]; then
    rm -rf output/pretrain_stage1_rsicd
    echo "✅ 已删除: output/pretrain_stage1_rsicd/"
else
    echo "ℹ️  不存在: output/pretrain_stage1_rsicd/ (跳过)"
fi

# 清理 Stage 2
if [ -d "output/pretrain_stage2_rsicd" ]; then
    rm -rf output/pretrain_stage2_rsicd
    echo "✅ 已删除: output/pretrain_stage2_rsicd/"
else
    echo "ℹ️  不存在: output/pretrain_stage2_rsicd/ (跳过)"
fi

# 清理 Fine-tuning
if [ -d "output/caption_rsicd_ft" ]; then
    rm -rf output/caption_rsicd_ft
    echo "✅ 已删除: output/caption_rsicd_ft/"
else
    echo "ℹ️  不存在: output/caption_rsicd_ft/ (跳过)"
fi

echo ""
echo -e "${GREEN}✅ 清理完成！${NC}"
echo ""

# 运行验证脚本
echo -e "${GREEN}🔍 步骤 2: 验证代码修改${NC}"
echo "----------------------------------------------------------------------"
python verify_model_changes.py
echo ""

# 询问是否开始训练
echo -e "${YELLOW}📋 准备开始训练${NC}"
echo "----------------------------------------------------------------------"
echo "配置信息:"
echo "  - 数据集: RSICD"
echo "  - GPU: 4 卡 (CUDA_VISIBLE_DEVICES=0,1,2,3)"
echo "  - agent_num: 64"
echo "  - gate_bias: -1.0 (分层自适应)"
echo "  - residual_weight: 0.8 (可学习)"
echo ""
read -p "是否立即开始 Stage 1 预训练？(y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${GREEN}🚀 开始 Stage 1 预训练...${NC}"
    echo "日志将输出到: output/pretrain_stage1_rsicd/"
    echo ""
    
    # 开始训练
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    torchrun --nproc_per_node=4 --master_port=29501 \
    train.py --cfg-path project/bita/train/pretrain_stage1_rsicd.yaml
else
    echo ""
    echo -e "${YELLOW}⏸️  训练已取消${NC}"
    echo ""
    echo "手动启动命令:"
    echo "  CUDA_VISIBLE_DEVICES=0,1,2,3 \\"
    echo "  torchrun --nproc_per_node=4 --master_port=29501 \\"
    echo "  train.py --cfg-path project/bita/train/pretrain_stage1_rsicd.yaml"
    echo ""
fi

echo ""
echo "======================================================================"
echo "脚本执行完成"
echo "======================================================================"


