import torch 
import torch.nn as nn 
import math


class AgentAttention(nn.Module): 
    def __init__(self, dim, num_heads=8, num_agents=16, dropout=0.1, encoder_width=None): 
        super().__init__() 

        self.dim = dim 
        self.num_heads = num_heads 
        self.num_agents = num_agents 
        self.head_dim = dim // num_heads 
        self.dropout = nn.Dropout(dropout)
        self.encoder_width = encoder_width if encoder_width is not None else dim

        assert dim % num_heads == 0 

        self.q_proj = nn.Linear(dim, dim) 
        self.k_proj = nn.Linear(self.encoder_width, dim) 
        self.v_proj = nn.Linear(self.encoder_width, dim) 

        self.agent_tokens = nn.Parameter( 
            torch.randn(1, num_agents, dim) * 0.02 
        ) 

        self.proj = nn.Linear(dim, dim)
        self.output_layer_norm = nn.LayerNorm(dim)
        
        # 【方案 4 改进】：可学习的残差权重，初始化为 0.8
        # 允许模型自适应调整残差连接强度
        self.residual_weight = nn.Parameter(torch.tensor(0.8))

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False): 
        # 处理标准 BertSelfAttention 调用方式
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
            # 确保掩码维度匹配
            if encoder_attention_mask.dim() == 4:
                # [B, 1, 1, Nk] -> [B, num_heads, num_agents, Nk]
                encoder_attention_mask = encoder_attention_mask.expand(-1, self.num_heads, self.num_agents, -1)
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
        
        # 【方案 4 改进】：使用可学习的残差权重
        # output = proj_out + residual_weight * query
        output = output + self.residual_weight * query
        
        # Add LayerNorm after residual connection for stability
        output = self.output_layer_norm(output)
        
        # Return in the same format as BertSelfAttention
        past_key_value = (K, V)
        if output_attentions:
            return (output, attn2, past_key_value)
        else:
            return (output, past_key_value)
