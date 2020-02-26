from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from utils import logger
use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda")


def _gen_bias_mask(max_length):
    """
    Generates bias values (-Inf) to mask future timesteps during attention
    """
    np_mask = np.triu(np.full([max_length, max_length], -np.inf), 1)
    torch_mask = torch.from_numpy(np_mask).to(torch.float32)
    return torch_mask.unsqueeze(0).unsqueeze(1)


def gen_multiscale_bias_mask(max_length, scale):
    if scale == 4:
        return None
    np_mask = np.full([max_length, max_length], -np.inf)
    for i in range(0, max_length, 2 ** (4 - scale)):
        for j in range(0, max_length, 2 ** (4 - scale)):
            np_mask[i, j] = 0
    torch_mask = torch.from_numpy(np_mask).to(torch.float32)
    return torch_mask.unsqueeze(0).unsqueeze(1)

def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    Adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(
        np.arange(num_timescales).astype(np.float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]],
                    'constant', constant_values=[0.0, 0.0])
    signal = signal.reshape([1, length, channels])

    return torch.from_numpy(signal).to(torch.float32)


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
        return input + self.positional_embedding[:, :input.shape[1], :].to(device)


class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional embeddig as per https://arxiv.org/abs/1809.04281
    """

    def __init__(self, input_dim, key_dim, value_dim, output_dim,
                 max_len, num_heads, dropout=0.0):
        """
        Parameters:
            input_dim: Size of last dimension of input
            key_dim: Size of last dimension of keys. Must be divisible by num_head
            value_dim: Size of last dimension of values. Must be divisible by num_head
            output_dim: Size last dimension of the final output
            num_heads: Number of attention heads
            max_len: the length of input
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
        # self.relative_embedding = nn.Parameter(torch.randn(num_heads, key_dim // num_heads, 2 * max_len - 1))
        self.relative_embedding = nn.Parameter(torch.randn(num_heads, key_dim // num_heads, max_len))
        # num_heads * (key_dim // num_heads) * max_len

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
        return mask.float().to(device) * qe

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
            logits += mask[:, :, :logits.shape[-2], :logits.shape[-1]].type_as(logits.data)

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


class DecoderLayer0(nn.Module):
    def __init__(self, input_dim, hidden_dim, key_dim, value_dim, num_heads, max_len,
                 layer_dropout=0.0, attention_dropout=0.0):
        super(DecoderLayer0, self).__init__()
        self.mask = _gen_bias_mask(max_len)
        self.self_mha = RelativeMultiHeadAttention(input_dim, key_dim, value_dim, hidden_dim,
                                                   max_len, num_heads, attention_dropout)

        self.FFN_pre = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.FFN_suf = nn.Linear(hidden_dim // 2, hidden_dim)

        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, inputs, attention_map=False):
        x = inputs

        # Self-attention
        x_norm = self.layer_norm1(x)
        mask = self.mask
        result = self.self_mha(x_norm, x_norm, x_norm, attention_map, mask)
        y = result['output']

        y = self.dropout(y)
        x_add = x + y

        # Feed-forward layer
        x_norm = self.layer_norm2(x_add)

        y = self.FFN_pre(x_norm)
        y = self.relu(y)
        y = self.FFN_suf(y)

        y = self.dropout(y)
        y = x_add + y

        result['output'] = y
        return result


class EncoderLayer1(nn.Module):
    def __init__(self, input_dim, hidden_dim, key_dim, value_dim, num_heads, max_len,
                 layer_dropout=0.0, attention_dropout=0.0):
        super(EncoderLayer1, self).__init__()
        self.mha = RelativeMultiHeadAttention(input_dim, key_dim, value_dim, hidden_dim,
                                              max_len, num_heads, attention_dropout)
        self.FFN_pre = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.FFN_suf = nn.Linear(hidden_dim // 2, hidden_dim)

        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_bsa = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.layer_norm_bff = nn.LayerNorm(hidden_dim, eps=1e-6)
    
    def forward(self, inputs, attention_map=False):
        x = inputs
        x_norm = self.layer_norm_bsa(x)

        result = self.mha(x_norm, x_norm, x_norm, attention_map)
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


class DecoderLayer1(nn.Module):
    def __init__(self, input_dim, hidden_dim, key_dim, value_dim, num_heads, max_len,
                 layer_dropout=0.0, attention_dropout=0.0):
        super(DecoderLayer1, self).__init__()

        self.mask = _gen_bias_mask(max_len)

        self.self_mha = RelativeMultiHeadAttention(input_dim, key_dim, value_dim, hidden_dim,
                                                   max_len, num_heads, attention_dropout)
        self.mha2 = RelativeMultiHeadAttention(input_dim, key_dim, value_dim, hidden_dim,
                                               max_len, num_heads, attention_dropout)

        self.FFN_pre = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.FFN_suf = nn.Linear(hidden_dim // 2, hidden_dim)

        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.layer_norm3 = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, inputs, encoder_output, attention_map=False):
        x = inputs

        # 1st : Self-attention
        x_norm = self.layer_norm1(x)
        mask = self.mask
        result1 = self.self_mha(x_norm, x_norm, x_norm, attention_map, mask)
        y = result1['output']

        y = self.dropout(y)
        x_add = x + y

        # 2nd : Encoder output attention
        x_norm = self.layer_norm2(x_add)
        result2 = self.mha2(x_norm, encoder_output, encoder_output, attention_map)
        y = result2['output']

        y = self.dropout(y)
        x_add = x_add + y

        # 3rd : Feed-forward layer
        x_norm = self.layer_norm3(x_add)

        y = self.FFN_pre(x_norm)
        y = self.relu(y)
        y = self.FFN_suf(y)

        y = self.dropout(y)
        y = x_add + y

        result = {'output': y}
        if attention_map:
            result['weight1'] = result1['weight']
            result['weight2'] = result2['weight']
        return result


class EncoderLayer2(nn.Module):
    def __init__(self, input_dim, hidden_dim, key_dim, value_dim, num_heads, max_len,
                 layer_dropout=0.0, attention_dropout=0.0):
        super(EncoderLayer2, self).__init__()
        self.mask = _gen_bias_mask(max_len)
        self.mha = RelativeMultiHeadAttention(input_dim, key_dim, value_dim, hidden_dim,
                                              max_len, num_heads, attention_dropout)
        # self.multi_head_attention = RelativeMultiHeadAttention(input_dim, key_dim, value_dim, hidden_dim,
        #                                                        max_len, num_heads, attention_dropout)
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


class DecoderLayer2(nn.Module):
    def __init__(self, input_dim, hidden_dim, key_dim, value_dim, num_heads, max_len,
                 layer_dropout=0.0, attention_dropout=0.0, beat=True):
        super(DecoderLayer2, self).__init__()

        self.beat = beat
        self.mask = _gen_bias_mask(max_len)

        self.self_mha = RelativeMultiHeadAttention(input_dim, key_dim, value_dim, hidden_dim,
                                                   max_len, num_heads, attention_dropout)
        self.mha_enc = RelativeMultiHeadAttention(input_dim, key_dim, value_dim, hidden_dim,
                                                  max_len, num_heads, attention_dropout)

        self.FFN_pre = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.FFN_suf = nn.Linear(hidden_dim // 2, hidden_dim)
        if self.beat:
            self.FFN_pre_E = nn.Linear(hidden_dim, hidden_dim // 2)
            self.FFN_suf_E = nn.Linear(hidden_dim // 2, hidden_dim)

        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.layer_norm3 = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, inputs, encoder_output=None, attention_map=False):
        x = inputs

        # 1st : Self-attention
        x_norm = self.layer_norm1(x)
        if encoder_output is None:
            # Encoder, no masking needed
            mask = None
        else:
            # Decoder, needs masking
            mask = self.mask
        result1 = self.self_mha(x_norm, x_norm, x_norm, attention_map, mask)
        y = result1['output']

        y = self.dropout(y)
        x_add = x + y

        # 2nd : Encoder output attention
        if encoder_output is not None:
            x_norm = self.layer_norm2(x_add)
            result2 = self.mha_enc(x_norm, encoder_output, encoder_output, attention_map)
            y = result2['output']

            y = self.dropout(y)
            x_add = x_add + y

        # 3rd : Feed-forward layer
        x_norm = self.layer_norm3(x_add)

        if self.beat and encoder_output is None:
            # Beat Encoder
            y = self.FFN_pre_E(x_norm)
            y = self.relu(y)
            y = self.FFN_suf_E(y)
        else:
            # Decoder (Beat or Pitch)
            y = self.FFN_pre(x_norm)
            y = self.relu(y)
            y = self.FFN_suf(y)

        y = self.dropout(y)
        y = x_add + y

        result = {'output': y}
        if attention_map:
            result['weight1'] = result1['weight']
            if encoder_output is not None:
                result['weight2'] = result2['weight']
        return result


class DecoderLayer3(nn.Module):
    def __init__(self, input_dim, hidden_dim, key_dim, value_dim, num_heads, max_len, scale,
                 layer_dropout=0.0, attention_dropout=0.0):
        super(DecoderLayer3, self).__init__()

        self.mask = gen_multiscale_bias_mask(max_len, scale)

        self.self_mha = RelativeMultiHeadAttention(input_dim, key_dim, value_dim, hidden_dim,
                                                   max_len, num_heads, attention_dropout)

        self.FFN_pre = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.FFN_suf = nn.Linear(hidden_dim // 2, hidden_dim)

        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, inputs, encoder_output=None, attention_map=False):
        x = inputs

        # Self-attention
        if encoder_output is None:
            mask = self.mask
        else:
            mask = None
        x_norm = self.layer_norm1(x)
        result = self.self_mha(x_norm, x_norm, x_norm, attention_map, mask)
        y = result['output']

        y = self.dropout(y)
        x_add = x + y

        # Feed-forward layer
        x_norm = self.layer_norm2(x_add)

        y = self.FFN_pre(x_norm)
        y = self.relu(y)
        y = self.FFN_suf(y)

        y = self.dropout(y)
        y = x_add + y

        result['output'] = y
        return result


class OutputLayer(nn.Module):
    """
    Abstract base class for output layer.
    Handles projection to output labels
    """
    def __init__(self, hidden_dim, output_size, probs_out=False):
        super(OutputLayer, self).__init__()
        self.output_size = output_size
        self.output_projection = nn.Linear(hidden_dim, output_size)
        self.probs_out = probs_out
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_dim=int(hidden_dim/2), batch_first=True, bidirectional=True)
        self.hidden_dim = hidden_dim

    def loss(self, hidden, labels):
        raise NotImplementedError('Must implement {}.loss'.format(self.__class__.__name__))


class SoftmaxOutputLayer(OutputLayer):
    """
    Implements a softmax based output layer
    """
    def forward(self, hidden):
        logits = self.output_projection(hidden)
        probs = F.softmax(logits, -1)
        # _, predictions = torch.max(probs, dim=-1)
        topk, indices = torch.topk(probs, 2)
        predictions = indices[:,:,0]
        second = indices[:,:,1]
        if self.probs_out is True:
            return probs
        return predictions, second

    def loss(self, hidden, labels):
        logits = self.output_projection(hidden)
        log_probs = F.log_softmax(logits, -1)
        return F.nll_loss(log_probs.view(-1, self.output_size), labels.view(-1))


# reference: http://pytorch.org/docs/master/_modules/torch/optim/lr_scheduler.html#ReduceLROnPlateau
def transformer_learning_rate(optimizer, model_dim, step_num, warmup_steps=4000):
    for i, param_group in enumerate(optimizer.param_groups):
        new_lr = model_dim**(-0.5) * min(step_num**(-0.5), step_num*warmup_steps**(-1.5))
        # old_lr = float(param_group['lr'])
        # new_lr = max(old_lr*factor, min_lr)
        param_group['lr'] = new_lr
        logger.info('adjusting learning rate : %.6f' % new_lr)
