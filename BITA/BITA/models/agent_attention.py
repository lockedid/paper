import torch 
import torch.nn as nn 


class AgentAttention(nn.Module): 
    def __init__(self, dim, num_heads=8, num_agents=16): 
        super().__init__() 

        self.dim = dim 
        self.num_heads = num_heads 
        self.num_agents = num_agents 
        self.head_dim = dim // num_heads 

        assert dim % num_heads == 0 

        self.q_proj = nn.Linear(dim, dim) 
        self.k_proj = nn.Linear(dim, dim) 
        self.v_proj = nn.Linear(dim, dim) 

        self.agent_tokens = nn.Parameter( 
            torch.randn(1, num_agents, dim) * 0.02 
        ) 

        self.proj = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, hidden_states, context=None, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False): 
        # 处理直接传递 query, context 的情况
        if context is not None:
            query = hidden_states
            encoder_hidden_states = context
        else:
            # 处理标准 BertAttention 调用方式
            query = hidden_states
            context = encoder_hidden_states
        
        if context is None:
            raise ValueError("context or encoder_hidden_states must be provided")
            
        B, Nq, C = query.shape 
        Nk = context.shape[1] 

        Q = self.q_proj(query) 
        K = self.k_proj(context) 
        V = self.v_proj(context) 

        Q = Q.view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2) 
        K = K.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2) 
        V = V.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2) 

        agent = self.agent_tokens.expand(B, -1, -1) 
        A = agent.view(B, self.num_agents, self.num_heads, self.head_dim).transpose(1, 2) 

        attn1 = (A @ K.transpose(-2, -1)) / (self.head_dim ** 0.5) 
        attn1 = attn1.softmax(dim=-1) 
        agent_context = attn1 @ V
        
        # Add LayerNorm for stability
        agent_context = self.layer_norm(agent_context + agent.view(B, self.num_agents, self.num_heads, self.head_dim).transpose(1, 2))

        attn2 = (Q @ agent_context.transpose(-2, -1)) / (self.head_dim ** 0.5) 
        attn2 = attn2.softmax(dim=-1) 
        out = attn2 @ agent_context 

        out = out.transpose(1, 2).contiguous().view(B, Nq, C) 

        output = self.proj(out)
        
        # Add residual connection to preserve original information
        output = output + query
        
        # Return in the same format as BertAttention
        if output_attentions:
            return (output, attn1, attn2)
        else:
            return (output,)