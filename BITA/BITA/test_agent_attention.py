import torch
import torch.nn as nn
import math

class AgentAttention(nn.Module):
    def __init__(self, dim, num_heads=8, num_agents=16, dropout=0.1):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.num_agents = num_agents
        self.head_dim = dim // num_heads
        self.dropout = nn.Dropout(dropout)

        assert dim % num_heads == 0

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.agent_tokens = nn.Parameter(
            torch.randn(1, num_agents, dim) * 0.02
        )

        self.proj = nn.Linear(dim, dim)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
        # 处理标准 BertAttention 调用方式
        query = hidden_states
        context = encoder_hidden_states 
        B, Nq, C = query.shape 
        Nk = context.shape[1] 

        Q = self.q_proj(query) 
        
        # 处理 past_key_value
        if past_key_value is not None:
            # 如果有缓存的 key 和 value，使用它们
            K = past_key_value[0]
            V = past_key_value[1]
        else:
            # 否则，从上下文计算
            K = self.k_proj(context)
            V = self.v_proj(context)
            K = K.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        Q = Q.view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2) 

        agent = self.agent_tokens.expand(B, -1, -1) 
        A = agent.view(B, self.num_agents, self.num_heads, self.head_dim).transpose(1, 2) 

        attn1 = (A @ K.transpose(-2, -1)) / (self.head_dim ** 0.5) 
        
        # 应用编码器注意力掩码
        if encoder_attention_mask is not None:
            attn1 = attn1 + encoder_attention_mask
        
        attn1 = attn1.softmax(dim=-1) 
        attn1 = self.dropout(attn1)
        agent_context = attn1 @ V 

        attn2 = (Q @ agent_context.transpose(-2, -1)) / (self.head_dim ** 0.5) 
        attn2 = attn2.softmax(dim=-1) 
        attn2 = self.dropout(attn2)
        
        # 应用 head mask
        if head_mask is not None:
            attn2 = attn2 * head_mask
        
        out = attn2 @ agent_context 

        out = out.transpose(1, 2).contiguous().view(B, Nq, C) 

        output = self.proj(out) 
        
        # Return in the same format as BertAttention
        past_key_value = (K, V)
        if output_attentions:
            return (output, attn2, past_key_value)
        else:
            return (output, past_key_value)

# Test the AgentAttention module
B, Nq, Nk, C = 2, 64, 32, 256

query = torch.randn(B, Nq, C)
context = torch.randn(B, Nk, C)

attn = AgentAttention(C)

# 测试直接调用方式
out = attn(query, encoder_hidden_states=context)

print(f"Input query shape: {query.shape}")
print(f"Input context shape: {context.shape}")
print(f"Output shape: {out[0].shape}")
print(f"Expected output shape: [{B}, {Nq}, {C}]")
print(f"Test passed: {out[0].shape == (B, Nq, C)}")

# 测试带有 past_key_value 的调用
out_with_cache = attn(query, encoder_hidden_states=context)
past_key_value = out_with_cache[1]
out2 = attn(query, encoder_hidden_states=context, past_key_value=past_key_value)

print(f"\nTest with past_key_value:")
print(f"Output shape with cache: {out2[0].shape}")
print(f"Test passed: {out2[0].shape == (B, Nq, C)}")