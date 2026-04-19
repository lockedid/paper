# BITA 分数据集独立训练指南

## 📋 概述

本指南说明如何使用BITA项目进行**分数据集独立训练**。每个数据集（RSICD/NWPU/UCM）都有完整的三阶段训练流程（Stage 1 → Stage 2 → Fine-tune），互不干扰。

---

## 🗂️ 文件结构

### 配置文件位置
```
BITA/project/bita/train/
├── pretrain_stage1_rsicd.yaml    # RSICD Stage 1
├── pretrain_stage2_rsicd.yaml    # RSICD Stage 2
├── caption_rsicd_ft.yaml         # RSICD Fine-tune
├── pretrain_stage1_nwpu.yaml     # NWPU Stage 1
├── pretrain_stage2_nwpu.yaml     # NWPU Stage 2
├── caption_nwpu_ft.yaml          # NWPU Fine-tune
├── pretrain_stage1_ucm.yaml      # UCM Stage 1
├── pretrain_stage2_ucm.yaml      # UCM Stage 2
└── caption_ucm_ft.yaml           # UCM Fine-tune
```

### 训练脚本位置
```
scripts/bita/train/
├── pretrain_stage1_rsicd.sh
├── pretrain_stage2_rsicd.sh
├── finetune_rsicd.sh
├── pretrain_stage1_nwpu.sh
├── pretrain_stage2_nwpu.sh
├── finetune_nwpu.sh
├── pretrain_stage1_ucm.sh
├── pretrain_stage2_ucm.sh
└── finetune_ucm.sh
```

### 输出目录结构
```
output/
├── pretrain_stage1_rsicd/    ← RSICD Stage 1 检查点
├── pretrain_stage2_rsicd/    ← RSICD Stage 2 检查点
├── finetune_rsicd/           ← RSICD 微调结果
├── pretrain_stage1_nwpu/     ← NWPU Stage 1 检查点
├── pretrain_stage2_nwpu/     ← NWPU Stage 2 检查点
├── finetune_nwpu/            ← NWPU 微调结果
├── pretrain_stage1_ucm/      ← UCM Stage 1 检查点
├── pretrain_stage2_ucm/      ← UCM Stage 2 检查点
└── finetune_ucm/             ← UCM 微调结果
```

---

## 🚀 快速开始

### RSICD 数据集完整流程

```bash
cd /home/wj/code/BITA-main/BITA

# 1. Stage 1 预训练（从头训练）
bash scripts/bita/train/pretrain_stage1_rsicd.sh

# 2. Stage 2 预训练（加载 Stage 1 权重）
bash scripts/bita/train/pretrain_stage2_rsicd.sh

# 3. 更新模型配置文件（重要！）
# 编辑 configs/models/bita/bita_caption_opt2.7b.yaml
# 将 finetuned 字段修改为：
# finetuned: "/home/wj/code/BITA-main/BITA/BITA/output/pretrain_stage2_rsicd/checkpoint_best.pth"

# 4. 微调
bash scripts/bita/train/finetune_rsicd.sh
```

### NWPU 数据集完整流程

```bash
cd /home/wj/code/BITA-main/BITA

# 1. Stage 1 预训练
bash scripts/bita/train/pretrain_stage1_nwpu.sh

# 2. Stage 2 预训练
bash scripts/bita/train/pretrain_stage2_nwpu.sh

# 3. 更新模型配置文件
# 编辑 configs/models/bita/bita_caption_opt2.7b.yaml
# finetuned: "/home/wj/code/BITA-main/BITA/BITA/output/pretrain_stage2_nwpu/checkpoint_best.pth"

# 4. 微调
bash scripts/bita/train/finetune_nwpu.sh
```

### UCM 数据集完整流程

```bash
cd /home/wj/code/BITA-main/BITA

# 1. Stage 1 预训练
bash scripts/bita/train/pretrain_stage1_ucm.sh

# 2. Stage 2 预训练
bash scripts/bita/train/pretrain_stage2_ucm.sh

# 3. 更新模型配置文件
# 编辑 configs/models/bita/bita_caption_opt2.7b.yaml
# finetuned: "/home/wj/code/BITA-main/BITA/BITA/output/pretrain_stage2_ucm/checkpoint_best.pth"

# 4. 微调
bash scripts/bita/train/finetune_ucm.sh
```

---

## ⚙️ 关键配置说明

### 1. 分布式端口规划

为避免多任务并行时的端口冲突，已为每个数据集分配独立端口段：

| 数据集 | Stage 1 | Stage 2 | Fine-tune |
|--------|---------|---------|-----------|
| RSICD  | 29501   | 29502   | 29503     |
| NWPU   | 29511   | 29512   | 29513     |
| UCM    | 29521   | 29522   | 29523     |

✅ **优势**：可以同时运行多个数据集的训练任务，互不干扰。

### 2. Warmup Steps 调整策略

根据数据集规模调整 warmup_steps，避免收敛过慢：

| 数据集 | 样本数 | Stage 1 | Stage 2 | Fine-tune |
|--------|--------|---------|---------|-----------|
| RSICD  | ~8k    | 5000    | 3000    | 1000      |
| NWPU   | ~30k   | 5000    | 3000    | 1000      |
| UCM    | ~2k    | 1000    | 500     | 200       |

**原则**：warmup_steps ≈ 总训练步数的 1%-5%

### 3. 梯度检查点优化

所有配置文件中均显式设置：
```yaml
use_grad_checkpoint: False  # CRITICAL when freeze_vit=True
freeze_vit: True
```

**原因**：
- ✅ **性能提升**：冻结参数无需反向传播，禁用检查点可提升训练速度 ~5-10%
- ✅ **规范一致**：遵循"明确优于隐式"原则
- ✅ **数值稳定**：避免重计算带来的微小浮点差异

### 4. GPU 数量调整

如果只有 1-2 张 GPU，修改对应的 `.sh` 脚本：

#### 单卡训练示例
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port=29501 \
    train.py \
    --cfg-path ...
```

#### 双卡训练示例
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=29501 \
    train.py \
    --cfg-path ...
```

**注意**：降低 GPU 数量时，建议相应降低 `batch_size_train` 以防止 OOM。

---

## 📉 Batch Size 调整建议

根据显存大小调整配置文件中的 `batch_size_train`：

| 显存容量 | Stage 1 | Stage 2/Fine-tune |
|----------|---------|-------------------|
| 24GB+    | 96      | 48                |
| 16GB     | 64      | 32                |
| 12GB     | 32      | 16                |

---

## 🔍 训练监控与调试

### 查看训练日志
```bash
tail -f output/pretrain_stage1_rsicd/log.txt
```

### 检查 checkpoint
```bash
ls -lh output/pretrain_stage1_rsicd/
# 应看到: checkpoint_best.pth, checkpoint_*.pth, log.txt 等
```

### 验证配置是否正确加载
在训练启动时，检查终端输出中的配置信息，确认：
- ✅ 数据集名称正确（只有一个数据集）
- ✅ output_dir 路径正确
- ✅ pretrained 路径（Stage 2）指向正确的 Stage 1 checkpoint

---

## ⚠️ 常见问题与解决方案

### Q1: RuntimeError: Address already in use
**原因**: 多个训练任务使用了相同的 master_port  
**解决**: 使用本指南中分配的独立端口号

### Q2: 训练收敛过慢
**原因**: warmup_steps 相对于数据集过大  
**解决**: 
- UCM: 降低到 1000/500/200
- RSICD/NWPU: 可适当降低到 3000/1500/800

### Q3: CUDA out of memory
**原因**: batch_size 过大  
**解决**: 
1. 降低 `batch_size_train`
2. 减少 GPU 数量（相应调整 batch_size）
3. 启用梯度累积（微调阶段可设置 `accum_grad_iters: 2`）

### Q4: Stage 2 找不到 Stage 1 的权重
**原因**: 路径配置错误或 Stage 1 未完成  
**解决**: 
1. 检查 `pretrain_stage2_{dataset}.yaml` 中的 `model.pretrained` 路径
2. 确认 `output/pretrain_stage1_{dataset}/checkpoint_best.pth` 存在

### Q5: 微调前忘记更新模型配置
**症状**: 微调加载了错误的权重  
**解决**: 
1. 停止当前训练
2. 编辑 `configs/models/bita/bita_caption_opt2.7b.yaml`
3. 更新 `finetuned` 字段指向正确的 Stage 2 checkpoint
4. 重新启动微调

---

## 📝 快速参考卡片

### RSICD 完整流程
```bash
# 1. Stage 1
bash scripts/bita/train/pretrain_stage1_rsicd.sh

# 2. Stage 2 (自动加载 Stage 1 权重)
bash scripts/bita/train/pretrain_stage2_rsicd.sh

# 3. 更新模型配置
# 编辑 configs/models/bita/bita_caption_opt2.7b.yaml
# finetuned: ".../output/pretrain_stage2_rsicd/checkpoint_best.pth"

# 4. Fine-tune
bash scripts/bita/train/finetune_rsicd.sh
```

### NWPU 完整流程
```bash
bash scripts/bita/train/pretrain_stage1_nwpu.sh
bash scripts/bita/train/pretrain_stage2_nwpu.sh
# 更新 finetuned 路径为 pretrain_stage2_nwpu/checkpoint_best.pth
bash scripts/bita/train/finetune_nwpu.sh
```

### UCM 完整流程
```bash
bash scripts/bita/train/pretrain_stage1_ucm.sh
bash scripts/bita/train/pretrain_stage2_ucm.sh
# 更新 finetuned 路径为 pretrain_stage2_ucm/checkpoint_best.pth
bash scripts/bita/train/finetune_ucm.sh
```

---

## 🎯 最佳实践建议

### 1. 训练顺序
```bash
# 推荐：先完成一个数据集的完整流程，再开始下一个
RSICD: Stage1 → Stage2 → Finetune
NWPU: Stage1 → Stage2 → Finetune
UCM:  Stage1 → Stage2 → Finetune
```

### 2. 并行训练
如果需要同时训练多个数据集，确保：
- ✅ 使用不同的端口（已配置好）
- ✅ 有足够的 GPU 显存
- ✅ 监控资源使用情况

```bash
# 示例：同时启动 RSICD Stage 1 和 NWPU Stage 1
bash scripts/bita/train/pretrain_stage1_rsicd.sh &
bash scripts/bita/train/pretrain_stage1_nwpu.sh &
wait
```

### 3. 实验管理
- 📊 记录每次实验的配置和超参数
- 📈 定期备份重要的 checkpoint
- 📝 维护实验日志，记录训练进度和结果

---

## 📚 相关文档

- [BITA项目README](../README.md)
- [数据集路径配置](../../datasetfilepath.md)

---

## ✨ 总结

分数据集独立训练的优势：
- ✅ **清晰的实验隔离**：每个数据集的训练过程完全独立
- ✅ **针对性的超参数优化**：可为不同规模数据集定制配置
- ✅ **公平的对比**：便于进行 dataset-wise 性能对比
- ✅ **灵活的并行训练**：支持同时运行多个任务

祝您训练顺利！🚀
