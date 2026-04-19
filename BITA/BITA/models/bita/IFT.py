import math
import os
import warnings
from functools import partial
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
from torch import Tensor, device, dtype, nn
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.file_utils import (
    ModelOutput,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from transformers.models.bert.configuration_bert import BertConfig


logger = logging.get_logger(__name__)


class AdaptiveGate(nn.Module):
    """Token-level adaptive gate for fusion"""
    def __init__(self, hidden_dim, gate_bias=-1.0):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        # Initialize bias to gate_bias (default -1.0 for better agent utilization)
        # Sigmoid(-1.0) = 0.269, allowing ~27% agent feature weight
        # Increased from -1.5 (18%) to -1.0 (27%) based on experimental results
        self.gate_net[-2].bias.data.fill_(gate_bias)
    
    def forward(self, query_tokens):
        # [B, Q, D] -> [B, Q, 1]
        return self.gate_net(query_tokens)


class MultiScaleFusion(nn.Module):
    """Learnable multi-scale fusion weights"""
    def __init__(self, num_scales=3):
        super().__init__()
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, scale_outputs):
        # scale_outputs: list of [B, Q, D]
        weights = self.softmax(self.scale_weights)  # [3]
        fused_output = sum(w * out for w, out in zip(weights, scale_outputs))
        return fused_output


class FNetBasicFourierTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._init_fourier_transform(config)

    def _init_fourier_transform(self, config):
        self.fourier_transform = partial(torch.fft.fftn, dim=(1, 2))

    def forward(self, hidden_states):
        # NOTE: We do not use torch.vmap as it is not integrated into PyTorch stable versions.
        # Interested users can modify the code to use vmap from the nightly versions, getting the vmap from here:
        # https://pytorch.org/docs/master/generated/torch.vmap.html. Note that fourier transform methods will need
        # change accordingly.

        outputs = self.fourier_transform(hidden_states).real
        return (outputs,)


class FNetBasicOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.LayerNorm(input_tensor + hidden_states)
        return hidden_states


class FNetFourierTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = FNetBasicFourierTransform(config)
        self.output = FNetBasicOutput(config)

    def forward(self, hidden_states):
        self_outputs = self.self(hidden_states)
        fourier_output = self.output(self_outputs[0], hidden_states)
        outputs = (fourier_output,)
        return outputs
    

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word and position embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )

        self.config = config

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        query_embeds=None,
        past_key_values_length=0,
    ):
        if input_ids is not None:
            seq_length = input_ids.size()[1]
        else:
            seq_length = 0

        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ].clone()

        if input_ids is not None:
            embeddings = self.word_embeddings(input_ids)
            if self.position_embedding_type == "absolute":
                position_embeddings = self.position_embeddings(position_ids)
                embeddings = embeddings + position_embeddings

            if query_embeds is not None:
                embeddings = torch.cat((query_embeds, embeddings), dim=1)
        else:
            embeddings = query_embeds

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config, is_cross_attention):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads   # 12
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)     # 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        if is_cross_attention:
            self.key = nn.Linear(config.encoder_width, self.all_head_size)
            self.value = nn.Linear(config.encoder_width, self.all_head_size)
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )
        self.save_attention = False

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        mixed_query_layer = self.query(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(-1, 1)
            position_ids_r = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if is_cross_attention and self.save_attention:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_dropped = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        context_layer = torch.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.self = BertSelfAttention(config, is_cross_attention)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them       # (attention_output, past_keys, past_values)
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


from BITA.models.modules.agent_attention import AgentAttention

class BertLayer(nn.Module):
    def __init__(self, config, layer_num, agent_num=64, gate_bias=-1.0):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.fourier = FNetFourierTransform(config)
        self.layer_num = layer_num
        
        # 【方案 1 改进】：分层自适应 Gate Bias（保守版本）
        # 根据层索引动态调整 gate_bias，实现不同层次的信息融合策略
        # 调整原因：RSICD 实验显示原策略（+0.4/-0.3）过于激进
        # 新策略：减小分层差异，避免过度依赖 Agent 特征
        # 
        # 浅层 (0-3): 需要更多视觉细节 → 适度激进的 gate (bias + 0.2)
        # 中层 (4-7): 平衡视觉与语义 → 保持原始 bias
        # 深层 (8-11): 偏向语言先验 → 适度保守的 gate (bias - 0.2)
        if layer_num < 4:
            adaptive_gate_bias = gate_bias + 0.2  # 例如：-1.0 + 0.2 = -0.8 (sigmoid≈0.31)
        elif layer_num < 8:
            adaptive_gate_bias = gate_bias        # 例如：-1.0 (sigmoid≈0.27)
        else:
            adaptive_gate_bias = gate_bias - 0.2  # 例如：-1.0 - 0.2 = -1.2 (sigmoid≈0.23)
        
        if (
            self.config.add_cross_attention
            # and layer_num % self.config.cross_attention_freq == 0
        ):
            # 保留 CrossAttention
            self.crossattention = BertAttention(
                config,
                is_cross_attention=True
            )
            # 保留 AgentAttention，使用传入的 agent_num
            self.agentattention = AgentAttention(
                dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_agents=agent_num,
                dropout=config.attention_probs_dropout_prob,
                encoder_width=config.encoder_width
            )
            # 【方案 1 改进】：使用分层自适应的 gate_bias
            self.gate = AdaptiveGate(config.hidden_size, gate_bias=adaptive_gate_bias)
            
            # 新增 Multi-scale Fusion
            self.multi_scale_fusion = MultiScaleFusion(num_scales=3)
            
            # 新增 Fusion LayerNorm
            self.fusion_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            
            # 保留 cross_norm
            self.cross_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
         ################################################
        self.intermediate_query = BertIntermediate(config)
        self.output_query = BertOutput(config)
        ################################################

    def build_multi_scale(self, context, encoder_attention_mask=None):
        """
        构建多尺度特征表示
        
        Args:
            context: [B, N, D] 输入特征，可能包含 CLS token
            encoder_attention_mask: [B, 1, 1, N] 或 [B, N] 注意力掩码
        
        Returns:
            tuple: (scales, masks)
                - scales: list of [B, Ni, D] 特征列表
                - masks: list of [B, 1, 1, Ni] 对应的注意力掩码列表
        """
        B, N, D = context.shape
        
        # 分离 CLS token（如果有）
        # ViT-L/14: N=257 (1 CLS + 256 image patches)
        has_cls = False
        cls_token = None
        
        # 检测是否有 CLS token（第一个 token）
        if N > 256:
            # 假设有 CLS token
            cls_token = context[:, 0:1, :]  # [B, 1, D]
            spatial_features = context[:, 1:, :]  # [B, N-1, D]
            N_spatial = spatial_features.shape[1]
            has_cls = True
        else:
            spatial_features = context
            N_spatial = N
        
        # 处理 attention mask（如果有）
        original_mask = None
        spatial_mask = None
        if encoder_attention_mask is not None:
            # 标准化 mask 格式为 [B, 1, 1, N]
            if encoder_attention_mask.dim() == 2:
                # [B, N] -> [B, 1, 1, N]
                original_mask = encoder_attention_mask.unsqueeze(1).unsqueeze(2)
            elif encoder_attention_mask.dim() == 3:
                # [B, 1, N] -> [B, 1, 1, N]
                original_mask = encoder_attention_mask.unsqueeze(2)
            elif encoder_attention_mask.dim() == 4:
                original_mask = encoder_attention_mask
            else:
                logger.warning(f"Unexpected mask dimension: {encoder_attention_mask.dim()}")
                original_mask = encoder_attention_mask
            
            # 分离 CLS token 的 mask
            if has_cls:
                cls_mask = original_mask[:, :, :, :1]  # [B, 1, 1, 1]
                spatial_mask = original_mask[:, :, :, 1:]  # [B, 1, 1, N-1]
            else:
                spatial_mask = original_mask
        
        # 计算空间特征的尺寸
        size = int(N_spatial ** 0.5)
        
        # 验证是否为平方数，如果不是则进行调整
        if size * size != N_spatial:
            # 尝试找到最接近的平方数
            if size * size < N_spatial:
                # 截断到最近的平方数
                spatial_features = spatial_features[:, :size*size, :]
                if spatial_mask is not None:
                    spatial_mask = spatial_mask[:, :, :, :size*size]
            else:
                # 填充到最近的平方数
                pad_len = (size + 1) * (size + 1) - N_spatial
                # 使用最后一个特征重复填充
                pad_features = spatial_features[:, -1:, :].repeat(1, pad_len, 1)
                spatial_features = torch.cat([spatial_features, pad_features], dim=1)
                
                if spatial_mask is not None:
                    # 也填充 mask，使用相同的值
                    pad_mask = spatial_mask[:, :, :, -1:].repeat(1, 1, 1, pad_len)
                    spatial_mask = torch.cat([spatial_mask, pad_mask], dim=-1)
                
                size = size + 1
        
        # 重构为 2D 空间图 [B, D, H, W]
        spatial_map = spatial_features.transpose(1, 2).view(B, D, size, size)
        
        # 对 mask 也进行同样的重构 [B, 1, 1, H*W] -> [B, 1, H, W]
        spatial_mask_map = None
        if spatial_mask is not None:
            spatial_mask_map = spatial_mask[:, 0, 0, :].view(B, 1, size, size)  # [B, 1, H, W]
        
        scales = []
        masks = []
        
        # Scale 1: 原始分辨率
        scale1 = spatial_features
        if has_cls and cls_token is not None:
            # 重新添加 CLS token
            scale1 = torch.cat([cls_token, scale1], dim=1)
            # CLS token 的 mask 保持有效
            if spatial_mask is not None:
                scale1_mask = torch.cat([cls_mask, spatial_mask], dim=-1)
            else:
                scale1_mask = None
        else:
            scale1_mask = spatial_mask
        
        scales.append(scale1)
        masks.append(scale1_mask)
        
        # Scale 2: 2x 下采样
        try:
            scale2_map = F.avg_pool2d(spatial_map, kernel_size=2, stride=2)
            scale2 = scale2_map.view(B, D, -1).transpose(1, 2)
            
            # 对 mask 进行下采样
            scale2_mask = None
            if spatial_mask_map is not None:
                # 使用 avg_pool2d 下采样 mask
                scale2_mask_map = F.avg_pool2d(spatial_mask_map, kernel_size=2, stride=2)
                # 阈值化：只要有一个位置有效，该区域就有效
                scale2_mask_map = (scale2_mask_map > 0).float()
                scale2_mask = scale2_mask_map.view(B, 1, 1, -1)  # [B, 1, 1, H'*W']
            
            if has_cls and cls_token is not None:
                scale2 = torch.cat([cls_token, scale2], dim=1)
                if scale2_mask is not None:
                    scale2_mask = torch.cat([cls_mask, scale2_mask], dim=-1)
            
            scales.append(scale2)
            masks.append(scale2_mask)
            
        except Exception as e:
            # 如果池化失败，回退到 scale1
            logger.warning(f"Scale2 pooling failed: {e}, using scale1 instead")
            scales.append(scale1)
            masks.append(scale1_mask)
        
        # Scale 3: 4x 下采样
        try:
            scale3_map = F.avg_pool2d(spatial_map, kernel_size=4, stride=4)
            scale3 = scale3_map.view(B, D, -1).transpose(1, 2)
            
            # 对 mask 进行下采样
            scale3_mask = None
            if spatial_mask_map is not None:
                # 使用 avg_pool2d 下采样 mask
                scale3_mask_map = F.avg_pool2d(spatial_mask_map, kernel_size=4, stride=4)
                # 阈值化：只要有一个位置有效，该区域就有效
                scale3_mask_map = (scale3_mask_map > 0).float()
                scale3_mask = scale3_mask_map.view(B, 1, 1, -1)  # [B, 1, 1, H''*W'']
            
            if has_cls and cls_token is not None:
                scale3 = torch.cat([cls_token, scale3], dim=1)
                if scale3_mask is not None:
                    scale3_mask = torch.cat([cls_mask, scale3_mask], dim=-1)
            
            scales.append(scale3)
            masks.append(scale3_mask)
            
        except Exception as e:
            # 如果池化失败，回退到 scale1
            logger.warning(f"Scale3 pooling failed: {e}, using scale1 instead")
            scales.append(scale1)
            masks.append(scale1_mask)
        
        return scales, masks

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        query_length=0,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # self_attn_past_key_value = (
        #     past_key_value[:2] if past_key_value is not None else None
        # )
        self_fourier_outputs = self.fourier(
            hidden_states
        )
        fourier_output = self_fourier_outputs[0]

        if query_length > 0:
            query_attention_output = fourier_output[:, :query_length, :]

            if self.has_cross_attention:
                assert (
                    encoder_hidden_states is not None
                ), "encoder_hidden_states must be given for cross-attention layers"
                
                # ==================== 方案 5：串行融合架构 ====================
                # 原逻辑（并行）：
                #   cross_out = cross_attention(query, visual)
                #   agent_out = agent_attention(query, visual)
                #   output = cross_out + gate * agent_out
                #
                # 新逻辑（串行）：
                #   cross_out = cross_attention(query, visual) + query  [残差 1]
                #   agent_out = agent_attention(cross_out, visual) + cross_out  [残差 2]
                #   output = cross_out + gate * agent_out + cross_out  [残差 3]
                # =================================================================
                
                # 【串行第 1 步】：Cross-Attention 提取基础视觉特征
                cross_out = self.crossattention(
                    self.cross_norm(query_attention_output),
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions=output_attentions,
                )[0]
                
                # 【残差连接 1】：保留原始 query 信息
                cross_out = cross_out + query_attention_output
                
                # 【串行第 2 步】：用 cross_out 作为 Agent 的输入（而非原始 query）
                # 这样 Agent 能理解 Cross-Attention 已经提取的视觉语义
                scales, masks = self.build_multi_scale(encoder_hidden_states, encoder_attention_mask)
                
                outs = []
                for i, s in enumerate(scales):
                    # 使用对应尺度的 mask
                    scale_mask = masks[i] if i < len(masks) else None
                    outs.append(self.agentattention(
                        self.cross_norm(cross_out),  # ← 关键改动：使用 cross_out 而非 original_query
                        attention_mask,
                        head_mask,
                        s,
                        scale_mask,
                        output_attentions=output_attentions,
                    )[0])
                
                # 【残差连接 2】：多尺度融合后保留 cross_out 信息
                multi_scale_out = self.multi_scale_fusion(outs)
                multi_scale_out = multi_scale_out + cross_out  # ← 新增残差
                
                # 【串行第 3 步】：Token-level adaptive gate fusion
                gate_values = self.gate(cross_out)  # ← 使用 cross_out 计算 gate
                
                # 【残差连接 3】：Gate 融合后再次保留 cross_out
                query_attention_output = cross_out + gate_values * multi_scale_out
                query_attention_output = query_attention_output + cross_out  # ← 新增残差
                
                # 【归一化】：Fusion LayerNorm
                query_attention_output = self.fusion_norm(query_attention_output)

            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk_query,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                query_attention_output,
            )
            if fourier_output.shape[1] > query_length:
                layer_output_text = apply_chunking_to_forward(
                    self.feed_forward_chunk,
                    self.chunk_size_feed_forward,
                    self.seq_len_dim,
                    fourier_output[:, query_length:, :],
                )
                layer_output = torch.cat([layer_output, layer_output_text], dim=1)
        else:
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                fourier_output,
            )
        # outputs = (layer_output,) + outputs
        outputs = (layer_output,)

        # outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def feed_forward_chunk_query(self, attention_output):
        intermediate_output = self.intermediate_query(attention_output)
        layer_output = self.output_query(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config, agent_num=64, gate_bias=-1.0):  # Updated default values
        super().__init__()
        self.config = config
        # Pass agent_num and gate_bias to each BertLayer
        self.layer = nn.ModuleList(
            [BertLayer(config, i, agent_num=agent_num, gate_bias=gate_bias) for i in range(config.num_hidden_layers)]
        )
        self.agent_num = agent_num
        self.gate_bias = gate_bias

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        query_length=0,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        next_decoder_cache = () if use_cache else None

        for i in range(self.config.num_hidden_layers):
            layer_module = self.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(
                            *inputs, past_key_value, output_attentions, query_length
                        )

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    query_length,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    base_model_prefix = "bert"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertModel(BertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=False, agent_num=64, gate_bias=-1.0):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)

        self.encoder = BertEncoder(config, agent_num=agent_num, gate_bias=gate_bias)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_extended_attention_mask(
        self,
        attention_mask: Tensor,
        input_shape: Tuple[int],
        device: device,
        is_decoder: bool,
        has_query: bool = False,
    ) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if is_decoder:
                batch_size, seq_length = input_shape

                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = (
                    seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
                    <= seq_ids[None, :, None]
                )

                # add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    if has_query:  # UniLM style attention mask
                        causal_mask = torch.cat(
                            [
                                torch.zeros(
                                    (batch_size, prefix_seq_len, seq_length),
                                    device=device,
                                    dtype=causal_mask.dtype,
                                ),
                                causal_mask,
                            ],
                            axis=1,
                        )
                    causal_mask = torch.cat(
                        [
                            torch.ones(
                                (batch_size, causal_mask.shape[1], prefix_seq_len),
                                device=device,
                                dtype=causal_mask.dtype,
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )
                extended_attention_mask = (
                    causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=self.dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        query_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_decoder=False,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = (           # False
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (        # False
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (                 # True
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is None:
            assert (
                query_embeds is not None
            ), "You have to specify query_embeds when input_ids is None"

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] - self.config.query_length
            if past_key_values is not None
            else 0
        )

        query_length = query_embeds.shape[1] if query_embeds is not None else 0

        embedding_output = self.embeddings(             # [B, 32, 768]
            input_ids=input_ids,
            position_ids=position_ids,
            query_embeds=query_embeds,
            past_key_values_length=past_key_values_length,
        )

        input_shape = embedding_output.size()[:-1]      # [B, 32]
        batch_size, seq_length = input_shape            
        device = embedding_output.device

        if attention_mask is None:
            attention_mask = torch.ones(                # [B, 32 + past_key_values_length]
                ((batch_size, seq_length + past_key_values_length)), device=device
            )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if is_decoder:
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask,
                input_ids.shape,
                device,
                is_decoder,
                has_query=(query_embeds is not None),
            )
        else:
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask, input_shape, device, is_decoder
            )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            if type(encoder_hidden_states) == list:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[
                    0
                ].size()
            else:
                (
                    encoder_batch_size,
                    encoder_sequence_length,
                    _,
                ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)

            if type(encoder_attention_mask) == list:
                encoder_extended_attention_mask = [
                    self.invert_attention_mask(mask) for mask in encoder_attention_mask
                ]
            elif encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                encoder_extended_attention_mask = self.invert_attention_mask(
                    encoder_attention_mask
                )
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(
                    encoder_attention_mask          # For query_embed, encoder_attention_mask is all zero
                )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            query_length=query_length,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class BertLMHeadModel(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config, agent_num=64, gate_bias=-1.0):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False, agent_num=agent_num, gate_bias=gate_bias)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()
        self.agent_num = agent_num
        self.gate_bias = gate_bias

    @classmethod  
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Custom from_pretrained to handle agent_num and gate_bias parameters"""
        agent_num = kwargs.pop('agent_num', 64)  # Extract agent_num from kwargs
        gate_bias = kwargs.pop('gate_bias', -1.0)  # Extract gate_bias from kwargs
        
        # Create config with agent_num if needed
        config = kwargs.get('config', None)
        if config is not None:
            # Update config to include agent_num for proper initialization
            pass
            
        # Initialize model with correct agent_num and gate_bias
        model = cls.__new__(cls)
        model.__init__(config if config is not None else BertConfig.from_pretrained(pretrained_model_name_or_path), 
                      agent_num=agent_num, gate_bias=gate_bias)
        
        # Load state dict from pretrained model
        state_dict = torch.load(pretrained_model_name_or_path, map_location='cpu') if isinstance(pretrained_model_name_or_path, str) and os.path.isfile(pretrained_model_name_or_path) else None
        
        if state_dict is not None:
            # Handle state dict loading manually to preserve agent_num specific layers
            model.load_state_dict(state_dict, strict=False)
        else:
            # Use parent from_pretrained for standard cases
            try:
                model_parent = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
                # Copy parameters that are compatible
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in model_parent.state_dict().items() if k in model_dict and model_dict[k].shape == v.shape}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict, strict=False)
            except Exception as e:
                logging.warning(f"Failed to load pretrained model normally: {e}")
                # Initialize from scratch with correct agent_num and gate_bias
                pass
                
        return model

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        query_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_logits=False,
        is_decoder=True,
        reduction="mean",
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are
            ignored (masked), the loss is only computed for the tokens with labels n ``[0, ..., config.vocab_size]``
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        Returns:
        Example::
            >>> from transformers import BertTokenizer, BertLMHeadModel, BertConfig
            >>> import torch
            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            >>> config = BertConfig.from_pretrained("bert-base-cased")
            >>> model = BertLMHeadModel.from_pretrained('bert-base-cased', config=config)
            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> prediction_logits = outputs.logits
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if labels is not None:
            use_cache = False
        if past_key_values is not None:
            query_embeds = None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            query_embeds=query_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_decoder=is_decoder,
        )

        sequence_output = outputs[0]
        if query_embeds is not None:
            sequence_output = outputs[0][:, query_embeds.shape[1] :, :]

        prediction_scores = self.cls(sequence_output)

        if return_logits:
            return prediction_scores[:, :-1, :].contiguous()

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss(reduction=reduction, label_smoothing=0.1)
            lm_loss = loss_fct(
                shifted_prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )
            if reduction == "none":
                lm_loss = lm_loss.view(prediction_scores.size(0), -1).sum(1)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, query_embeds, past=None, attention_mask=None, **model_kwargs
    ):
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)
        query_mask = input_ids.new_ones(query_embeds.shape[:-1])
        attention_mask = torch.cat([query_mask, attention_mask], dim=-1)

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "query_embeds": query_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past,
            "encoder_hidden_states": model_kwargs.get("encoder_hidden_states", None),
            "encoder_attention_mask": model_kwargs.get("encoder_attention_mask", None),
            "is_decoder": True,
        }

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx) for past_state in layer_past
                ),
            )
        return reordered_past


class BertForMaskedLM(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        query_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_logits=False,
        is_decoder=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            query_embeds=query_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_decoder=is_decoder,
        )

        if query_embeds is not None:
            sequence_output = outputs[0][:, query_embeds.shape[1] :, :]
        prediction_scores = self.cls(sequence_output)

        if return_logits:
            return prediction_scores

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )