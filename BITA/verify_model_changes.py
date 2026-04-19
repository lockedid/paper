#!/usr/bin/env python3
"""
简化版验证脚本 - 仅检查代码修改是否正确
不依赖外部库
"""

import re

def check_agent_attention():
    """检查 agent_attention.py 的修改"""
    print("=" * 70)
    print("检查 1: Agent Attention 增强残差")
    print("=" * 70)
    
    with open('/home/wj/code/BITA-main/BITA/BITA/models/modules/agent_attention.py', 'r') as f:
        content = f.read()
    
    # 检查 1: residual_weight 参数定义
    if 'self.residual_weight = nn.Parameter(torch.tensor(0.8))' in content:
        print("✅ residual_weight 参数定义正确")
    else:
        print("❌ residual_weight 参数定义错误或缺失")
        return False
    
    # 检查 2: 残差连接使用
    if 'output = output + self.residual_weight * query' in content:
        print("✅ 残差连接使用 residual_weight 正确")
    else:
        print("❌ 残差连接未使用 residual_weight")
        return False
    
    # 检查 3: 注释说明
    if '【方案 4 改进】' in content:
        print("✅ 包含修改标记注释")
    else:
        print("⚠️  缺少修改标记注释（不影响功能）")
    
    print()
    return True


def check_ift_layer():
    """检查 IFT.py 的分层 Gate 修改"""
    print("=" * 70)
    print("检查 2: BertLayer 分层自适应 Gate Bias")
    print("=" * 70)
    
    with open('/home/wj/code/BITA-main/BITA/BITA/models/bita/IFT.py', 'r') as f:
        content = f.read()
    
    # 检查 1: 分层逻辑
    checks = [
        ('if layer_num < 4:', '浅层条件判断'),
        ('adaptive_gate_bias = gate_bias + 0.4', '浅层 bias 调整'),
        ('elif layer_num < 8:', '中层条件判断'),
        ('adaptive_gate_bias = gate_bias', '中层保持原 bias'),
        ('adaptive_gate_bias = gate_bias - 0.3', '深层 bias 调整'),
        ('self.gate = AdaptiveGate(config.hidden_size, gate_bias=adaptive_gate_bias)', '使用 adaptive_gate_bias'),
    ]
    
    all_passed = True
    for check_str, description in checks:
        if check_str in content:
            print(f"✅ {description} 正确")
        else:
            print(f"❌ {description} 错误或缺失")
            all_passed = False
    
    # 检查 2: 注释说明
    if '【方案 1 改进】' in content:
        print("✅ 包含修改标记注释")
    else:
        print("⚠️  缺少修改标记注释（不影响功能）")
    
    print()
    return all_passed


def print_summary():
    """打印修改总结"""
    print("=" * 70)
    print("修改总结")
    print("=" * 70)
    print("""
✅ 方案 4: Agent Attention 增强残差
   - 文件: BITA/models/modules/agent_attention.py
   - 修改: 添加可学习参数 residual_weight (初始值 0.8)
   - 影响: 残差连接变为 output + residual_weight * query
   - 新增参数: 1 个/层 × 12 层 = 12 个标量参数

✅ 方案 1: 分层自适应 Gate Bias
   - 文件: BITA/models/bita/IFT.py
   - 修改: 根据 layer_num 动态调整 gate_bias
   - 策略:
     * Layer 0-3 (浅层): gate_bias + 0.4 → sigmoid≈0.35
     * Layer 4-7 (中层): gate_bias 不变  → sigmoid≈0.27
     * Layer 8-11 (深层): gate_bias - 0.3 → sigmoid≈0.21
   - 新增参数: 0 个（仅改变初始化）

📊 预期效果:
   - CIDEr 提升: +4~7 分
   - 训练稳定性: 提升
   - 显存占用: 几乎不变

⚠️  重要提醒:
   1. 必须删除旧 checkpoint 重新训练
   2. 建议降低学习率到 8e-5
   3. 训练开始时观察前几个 epoch 的 Loss 趋势
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("BITA 模型架构优化 - 代码修改验证")
    print("=" * 70 + "\n")
    
    result1 = check_agent_attention()
    result2 = check_ift_layer()
    
    if result1 and result2:
        print_summary()
        print("=" * 70)
        print("✅ 所有代码修改验证通过！")
        print("=" * 70)
        print("\n下一步操作:")
        print("1. 清理旧输出: rm -rf output/pretrain_stage1_rsicd/")
        print("2. 开始训练: bash scripts/bita/train/pretrain_stage1_rsicd.sh")
        print("3. 监控训练: tail -f output/pretrain_stage1_rsicd/log.txt")
    else:
        print("=" * 70)
        print("❌ 部分验证失败，请检查代码修改")
        print("=" * 70)
