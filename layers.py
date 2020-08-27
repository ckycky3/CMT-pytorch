from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def _gen_bias_mask(max_length):
    """
    Generates bias values (-Inf) to mask future timesteps during attention
    """
    np_mask = np.triu(np.full([max_length, max_length], -np.inf), 1)
    torch_mask = torch.from_numpy(np_mask).to(torch.float32)
    return torch_mask.unsqueeze(0).unsqueeze(1)


class DynamicPositionEmbedding(nn.Module):
    def __init__(self, hidden_dim, max_len):
        super(DynamicPositionEmbedding, self).__init__()

        embed_sinusoid_list = np.array([[
            [
                math.sin(
                    pos * math.exp(-math.log(10000) * i/hidden_dim) *
                    math.exp(math.log(10000)/hidden_dim * (i % 2))
                    + 0.5 * math.pi * (i % 2)
                )
                for i in range(hidden_dim)
            ]
            for pos in range(max_len)
        ]])
        self.positional_embedding = nn.Parameter(torch.tensor(embed_sinusoid_list, dtype=torch.float), requires_grad=False)

    def forward(self, input):
        return input + self.positional_embedding[:, :input.shape[1], :].to(input.device)


class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, input_dim, key_dim, value_dim, output_dim,
                 max_len, num_heads, preceding_only=True, dropout=0.0):
        """
        Parameters:
            input_dim: Size of last dimension of input
            key_dim: Size of last dimension of keys. Must be divisible by num_head
            value_dim: Size of last dimension of values. Must be divisible by num_head
            output_dim: Size last dimension of the final output
            num_heads: Number of attention heads
            max_len: the length of input
            preceding_only: False if attention to succeeding elements is needed
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(RelativeMultiHeadAttention, self).__init__()
        # Checks borrowed from
        # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
        if key_dim % num_heads != 0:
            raise ValueError("Key depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (key_dim, num_heads))
        if value_dim % num_heads != 0:
            raise ValueError("Value depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (value_dim, num_heads))

        self.num_heads = num_heads
        self.query_scale = (key_dim // num_heads) ** -0.5
        self.max_len = max_len

        # Key and query depth will be same
        self.query_linear = nn.Linear(input_dim, key_dim, bias=False)
        self.key_linear = nn.Linear(input_dim, key_dim, bias=False)
        self.value_linear = nn.Linear(input_dim, value_dim, bias=False)
        self.output_linear = nn.Linear(value_dim, output_dim, bias=False)

        # relative positional encoding
        rel_emb_len = max_len if preceding_only else 2 * max_len - 1
        self.relative_embedding = nn.Parameter(torch.randn(num_heads, key_dim // num_heads, rel_emb_len))

        # dropout for attention probs
        self.attention_dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(shape[0], shape[2], shape[3] * self.num_heads)

    def _calc_positional_embedding(self, queries, mask):
        # batch_size * num_heads * max_len * max_len
        if mask is not None:
            embedding = torch.matmul(queries, self.relative_embedding[:, :, :self.max_len])
            embedding = self._qe_masking(embedding)
            embedding = F.pad(embedding, (1, 0, 0, 0))
            embedding = embedding.view(-1, embedding.size(1), embedding.size(3), embedding.size(2))
            embedding = embedding[:, :, 1:, :]
        else:
            embedding = torch.matmul(queries, self.relative_embedding)
            embedding = F.pad(embedding, (1, 0, 0, 0))
            embedding = embedding.view(embedding.size(0), embedding.size(1), -1)[:, :, self.max_len:]
            embedding = embedding.view(embedding.size(0), embedding.size(1), self.max_len, -1)[:, :, :, :self.max_len]

        return embedding

    @staticmethod
    def _qe_masking(qe):
        lengths = torch.arange(qe.size(-1) - 1, qe.size(-1) - qe.size(-2) - 1, -1)
        maxlen = qe.size(-1)
        mask = torch.arange(maxlen).unsqueeze(0) >= lengths.unsqueeze(1)
        return mask.float().to(qe.device) * qe

    def forward(self, queries, keys, values, attention_map=False, mask=None):

        # Do a linear for each component
        queries = self.query_linear(queries)
        keys = self.key_linear(keys)
        values = self.value_linear(values)

        # Split into multiple heads
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        # Combine queries and keys
        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))

        # add position embedding
        logits += self._calc_positional_embedding(queries, mask)

        # Add bias to mask future values
        if mask is not None:
            logits += mask[:, :, :logits.shape[-2], :logits.shape[-1]].type_as(logits.data).to(logits.device)

        # Scale logits
        logits *= self.query_scale

        # Convert to probabilites
        weights = nn.functional.softmax(logits, dim=-1)

        # Dropout
        weights = self.attention_dropout(weights)

        # Combine with values to get context
        contexts = torch.matmul(weights, values)

        # Merge heads
        contexts = self._merge_heads(contexts)

        # Linear to get output
        outputs = self.output_linear(contexts)

        result = {'output': outputs}
        if attention_map:
            result['weight'] = weights

        return result


class SelfAttentionBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, key_dim, value_dim, num_heads, max_len,
                 preceding_only=True, layer_dropout=0.0, attention_dropout=0.0):
        super(SelfAttentionBlock, self).__init__()
        self.mask = _gen_bias_mask(max_len)
        self.mha = RelativeMultiHeadAttention(input_dim, key_dim, value_dim, hidden_dim,
                                              max_len, num_heads, preceding_only, attention_dropout)
        self.FFN_pre = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.FFN_suf = nn.Linear(hidden_dim // 2, hidden_dim)

        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_bsa = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.layer_norm_bff = nn.LayerNorm(hidden_dim, eps=1e-6)
    
    def forward(self, inputs, attention_map=False, masking=True):
        x = inputs
        x_norm = self.layer_norm_bsa(x)

        if masking:
            mask = self.mask
        else:
            mask = None
        result = self.mha(x_norm, x_norm, x_norm, attention_map, mask)
        y = result['output']
        
        y = self.dropout(y)

        x_add = x + y
        x_norm = self.layer_norm_bff(x_add)

        y = self.FFN_pre(x_norm)
        y = self.relu(y)
        y = self.FFN_suf(y)
        y = self.dropout(y)

        y = x_add + y

        result['output'] = y
        return result