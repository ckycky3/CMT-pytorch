import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from queue import PriorityQueue
from layers import gen_multiscale_bias_mask, DynamicPositionEmbedding, DecoderLayer0, EncoderLayer1, DecoderLayer1, EncoderLayer2, DecoderLayer2, DecoderLayer3
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class BaseModel(nn.Module):
    def __init__(self, num_events):
        super(BaseModel, self).__init__()

        self.num_events = num_events
        self.num_outputs = num_events

    def forward(self, input, chord_input, chord_target):
        raise NotImplementedError()

    def idx_to_one_hot(self, idx):
        one_hot = torch.zeros(idx.size(0), idx.size(1), self.num_events).to(device)
        one_hot.scatter_(2, idx.long().unsqueeze(-1), 1)
        return one_hot


def inflate(tensor, times, dim):
    repeat_dims = [1] * tensor.dim()
    repeat_dims[dim] = times
    return tensor.repeat(*repeat_dims)


# TODO: Needs modification
class MusicTransformer(BaseModel):
    def __init__(self, num_events=130, frame_per_bar=16, num_bars=8,
                 chord_emb=64, embedding_size=128, hidden_dim=128, num_layers=6, num_heads=4,
                 total_key_depth=128, total_value_depth=128, filter_size=128,
                 input_dropout=0.0, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        super(MusicTransformer, self).__init__(num_events)

        self.max_len = frame_per_bar * num_bars
        self.frame_per_bar = frame_per_bar
        self.num_chords = 12

        self.hidden_dim = hidden_dim

        # embedding layer
        # self.emb_mat = nn.Parameter(torch.randn(self.num_outputs, embedding_size, dtype=torch.float, requires_grad=True))
        self.note_emb = nn.Embedding(self.num_outputs, embedding_size)
        self.chord_emb = nn.Parameter(torch.randn(self.num_chords, chord_emb, dtype=torch.float, requires_grad=True))

        self.pos_encoding = DynamicPositionEmbedding(self.hidden_dim, self.max_len)

        # embedding dropout
        self.emb_dropout = nn.Dropout(input_dropout)

        # Decoding layers
        params = (
            hidden_dim,
            total_key_depth,
            total_value_depth,
            filter_size,
            num_heads,
            self.max_len,
            layer_dropout,
            attention_dropout,
            relu_dropout,
            False
        )
        self.layers = nn.ModuleList([
            # TODO: Needs modification
            DecoderLayer0(*params) for _ in range(num_layers)
        ])

        # output layer
        self.output_layer = nn.Linear(hidden_dim, self.num_outputs)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, chord_input, chord_target):
        # if input.size(-1) != self.num_events:
        #     input = self.idx_to_one_hot(input)
        # emb = torch.matmul(input, self.emb_mat)
        note_emb = self.note_emb(input)
        chord_input_emb = torch.matmul(chord_input.float(), self.chord_emb)
        chord_target_emb = torch.matmul(chord_target.float(), self.chord_emb)
        emb = torch.cat([note_emb, chord_input_emb, chord_target_emb], -1)
        emb *= torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float))
        emb = self.pos_encoding(emb)
        emb = self.emb_dropout(emb)
        # emb = self.embedding(batch_data)

        # model
        for _, layer in enumerate(self.layers):
            emb = layer(emb)

        # output layer
        output = self.output_layer(emb)
        output = self.log_softmax(output)

        return output

    def sampling(self, prime_idx, chord_input, chord_target, topk=None):
        # batch_size * prime_len * num_outputs
        batch_size = prime_idx.size(0)
        pad_length = self.max_len - prime_idx.size(1)
        pad = torch.ones([batch_size, pad_length], dtype=torch.long).to(device)
        pad *= (self.num_events - 1)
        result = torch.cat([prime_idx, pad], dim=1)

        # sampling phase
        for i in range(prime_idx.size(1), self.max_len):
            output = self.forward(result, chord_input, chord_target)
            if topk is None:
                idx = torch.argmax(output[:, i - 1, :], dim=1)
            else:
                topk_probs, topk_idxs = torch.topk(output[:, i-1, :], topk, dim=-1)
                idx = torch.gather(topk_idxs, 1, torch.multinomial(F.softmax(topk_probs, dim=-1), 1)).squeeze()
            result[:, i] = idx

        return result


class ChordLSTM_MusicTransformer(nn.Module):
    def __init__(self, num_pitch=130, frame_per_bar=16, num_bars=8,
                 chord_emb_size=128, note_emb_size=128, hidden_dim=128,
                 key_dim=128, value_dim=128, num_layers=6, num_heads=4,
                 input_dropout=0.0, layer_dropout=0.0, attention_dropout=0.0,
                 chord_add=True):
        super(ChordLSTM_MusicTransformer, self).__init__()

        self.max_len = frame_per_bar * num_bars
        self.frame_per_bar = frame_per_bar
        self.num_chords = 12
        self.chord_add = chord_add
        self.num_pitch = num_pitch

        self.hidden_dim = hidden_dim
        self.note_emb_size = note_emb_size
        self.chord_emb_size = chord_emb_size

        # embedding layer
        self.note_emb = nn.Embedding(self.num_pitch, self.note_emb_size)
        self.chord_emb = nn.Embedding(self.num_chords + 1, self.chord_emb_size, padding_idx=12)

        lstm_input = self.chord_emb_size if chord_add else self.chord_emb_size * 5
        self.chord_lstm = nn.LSTM(lstm_input, self.note_emb_size // 2, num_layers=1,
                                  batch_first=True, bidirectional=True)

        self.pos_encoding = DynamicPositionEmbedding(self.hidden_dim, self.max_len)

        # embedding dropout
        self.emb_dropout = nn.Dropout(input_dropout)

        # Decoding layers
        params = (
            2 * self.note_emb_size,
            hidden_dim,
            key_dim,
            value_dim,
            num_heads,
            self.max_len,
            layer_dropout,
            attention_dropout
        )
        self.layers = nn.ModuleList([
            EncoderLayer2(*params) for _ in range(num_layers)
        ])

        # output layer
        self.output_layer = nn.Linear(hidden_dim, self.num_pitch)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def init_lstm_hidden(self, batch_size):
        h0 = Variable(torch.zeros(2, batch_size, self.note_emb_size // 2)).to(device)
        c0 = Variable(torch.zeros(2, batch_size, self.note_emb_size // 2)).to(device)
        return (h0, c0)

    def forward(self, input, chord, attention_map=False):
        size = chord.size()
        note_emb = self.note_emb(input)
        if self.chord_add:
            # sum B * T * 5 * D to B * T * D
            chord_emb = torch.sum(self.chord_emb(chord), dim=2)
        else:
            # concat B * T * 5 * D to B * T * 5D
            chord_emb = self.chord_emb(chord).view(size[0], size[1], -1)

        h0, c0 = self.init_lstm_hidden(size[0])
        self.chord_lstm.flatten_parameters()
        out, _ = self.chord_lstm(chord_emb, (h0, c0))
        forward = out[:, 1:, :(self.note_emb_size // 2)]
        backward = out[:, 1:, (self.note_emb_size // 2):]

        emb = torch.cat([note_emb, forward, backward], -1)
        emb *= torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float))
        emb = self.pos_encoding(emb)
        emb = self.emb_dropout(emb)

        # model
        weights = []
        for _, layer in enumerate(self.layers):
            result = layer(emb, attention_map)
            emb = result['output']
            if attention_map:
                weights.append(result['weight'])

        # output layer
        output = self.output_layer(emb)
        output = self.log_softmax(output)

        result = {'output': output}
        if attention_map:
            result['weights'] = weights
        return result

    def sampling(self, prime_idx, chord, topk=None, attention_map=False):
        # batch_size * prime_len * num_outputs
        batch_size = prime_idx.size(0)
        pad_length = self.max_len - prime_idx.size(1)
        pad = torch.ones([batch_size, pad_length], dtype=torch.long).to(device)
        pad *= (self.num_pitch - 1)
        result = torch.cat([prime_idx, pad], dim=1)

        # sampling phase
        for i in range(prime_idx.size(1), self.max_len):
            result_dict = self.forward(result, chord, attention_map)
            output = result_dict['output']
            if topk is None:
                idx = torch.argmax(output[:, i - 1, :], dim=1)
            else:
                topk_probs, topk_idxs = torch.topk(output[:, i-1, :], topk, dim=-1)
                idx = torch.gather(topk_idxs, 1, torch.multinomial(F.softmax(topk_probs, dim=-1), 1)).squeeze()
            result[:, i] = idx

        result_dict['output'] = result
        return result_dict


class ChordLSTM_BeatMusicTransformer(nn.Module):
    def __init__(self, num_pitch=89, frame_per_bar=16, num_bars=8,
                 chord_emb_size=128, note_emb_size=128, hidden_dim=128,
                 key_dim=128, value_dim=128, num_layers=6, num_heads=4,
                 input_dropout=0.0, layer_dropout=0.0, attention_dropout=0.0,
                 chord_add=True):
        super(ChordLSTM_BeatMusicTransformer, self).__init__()

        self.max_len = frame_per_bar * num_bars
        self.frame_per_bar = frame_per_bar
        self.num_chords = 12
        self.chord_add = chord_add
        self.num_pitch = num_pitch
        self.num_beat = 3

        # self.beat_emb_size = chord_emb
        self.chord_emb_size = chord_emb_size
        self.beat_emb_size = note_emb_size // 8
        self.note_emb_size = note_emb_size
        self.chord_hidden = 7 * (note_emb_size // 32)  # 2 * chord_hidden + beat_emb = beat_hidden
        self.beat_hidden = 9 * (note_emb_size // 16)   # 2 * chord_hidden + beat_hidden = note_emb
        self.hidden_dim = hidden_dim

        # embedding layer
        self.chord_emb = nn.Embedding(self.num_chords + 1, self.chord_emb_size, padding_idx=12)
        self.beat_emb = nn.Embedding(self.num_beat, self.beat_emb_size)
        self.note_emb = nn.Embedding(self.num_pitch, self.note_emb_size)        

        # self.beat_lstm = nn.LSTM(self.beat_emb_size, self.note_emb_size // 4, num_layers=1,
        #                           batch_first=True, bidirectional=True)
        lstm_input = self.chord_emb_size if chord_add else self.chord_emb_size * 5
        self.chord_lstm = nn.LSTM(lstm_input, self.chord_hidden, num_layers=1,
                                  batch_first=True, bidirectional=True)
        self.beat_pos_enc = DynamicPositionEmbedding(self.beat_hidden, self.max_len)
        self.pos_encoding = DynamicPositionEmbedding(self.hidden_dim, self.max_len)

        # embedding dropout
        self.emb_dropout = nn.Dropout(input_dropout)

        # Decoding layers
        beat_params = (
            2 * self.chord_hidden + self.beat_emb_size,
            self.beat_hidden,
            key_dim // 4,
            value_dim // 4,
            num_heads,
            self.max_len,
            layer_dropout,
            attention_dropout
        )
        self.beat_layers = nn.ModuleList([
            EncoderLayer2(*beat_params) for _ in range(num_layers)
        ])
        
        pitch_params = (
            2 * self.note_emb_size,
            self.hidden_dim,
            key_dim,
            value_dim,
            num_heads,
            self.max_len,
            layer_dropout,
            attention_dropout
        )
        self.pitch_layers = nn.ModuleList([
            EncoderLayer2(*pitch_params) for _ in range(num_layers)
        ])

        # output layer
        self.beatout_layer = nn.Linear(self.beat_hidden, self.num_beat)
        self.output_layer = nn.Linear(self.hidden_dim, self.num_pitch)
        # self.output_layer = nn.Linear(self.hidden_dim + self.beat_hidden, self.num_pitch)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # for param in self.beat_emb.parameters():
        #     param.requires_grad = False
        # for param in self.beat_pos_enc.parameters():
        #     param.requires_grad = False
        # for param in self.beat_layers.parameters():
        #     param.requires_grad = False
        # for param in self.beatout_layer.parameters():
        #     param.requires_grad = False

        # for i in range(1, num_layers):
        #     self.beat_layers[i].mha.relative_embedding = self.enc_layers[i-1].mha.relative_embedding
        #     self.pitch_layers[i].mha.relative_embedding = self.pitch_layers[i-1].mha.relative_embedding

    def init_lstm_hidden(self, batch_size):
        h0 = Variable(torch.zeros(2, batch_size, self.chord_hidden)).to(device)
        c0 = Variable(torch.zeros(2, batch_size, self.chord_hidden)).to(device)
        return (h0, c0)

    # beat : time_len + 1   (input & target)
    # pitch : time_len      (input only)
    # chord : time_len + 1  (input & target)
    def forward(self, beat, pitch, chord, attention_map=False):
        # chord_hidden : time_len   (target timestep)
        chord_hidden = self.chord_forward(chord)

        beat_dec_result = self.beat_forward(beat[:, :-1], chord_hidden, attention_map, masking=True)
        beat_out = self.beatout_layer(beat_dec_result['output'])
        beat_out = self.log_softmax(beat_out)
        
        beat_enc_result = self.beat_forward(beat[:, 1:], chord_hidden, attention_map, masking=False)
        beat_emb = beat_enc_result['output']
        note_emb = self.note_emb(pitch)
        emb = torch.cat([note_emb, chord_hidden[0], chord_hidden[1], beat_emb], -1)
        emb *= torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float))
        pitch_output = self.pitch_forward(emb, attention_map)

        result = {'beat': beat_out,
                  # }
                  'pitch': pitch_output['output']}
        if attention_map:
            result['weights_bdec'] = beat_dec_result['weights']
            result['weights_benc'] = beat_enc_result['weights']
            result['weights_pitch'] = pitch_output['weights']
        return result

    def chord_forward(self, chord):
        size = chord.size()
        if self.chord_add:
            # sum B * T * 5 * D to B * T * D
            chord_emb = torch.sum(self.chord_emb(chord), dim=2)
        else:
            # concat B * T * 5 * D to B * T * 5D
            chord_emb = self.chord_emb(chord).view(size[0], size[1], -1)

        h0, c0 = self.init_lstm_hidden(size[0])
        self.chord_lstm.flatten_parameters()
        chord_out, _ = self.chord_lstm(chord_emb, (h0, c0))
        chord_for = chord_out[:, 1:, :self.chord_hidden]
        chord_back = chord_out[:, 1:, self.chord_hidden:]
        return chord_for, chord_back
    
    def beat_forward(self, beat, chord_hidden, attention_map=False, masking=True):
        beat_emb = self.beat_emb(beat)
        beat_emb = torch.cat([beat_emb, chord_hidden[0], chord_hidden[1]], -1)
        beat_emb *= torch.sqrt(torch.tensor(self.beat_hidden, dtype=torch.float))
        beat_emb = self.beat_pos_enc(beat_emb)
        beat_emb = self.emb_dropout(beat_emb)
        
        weights = []
        for _, layer in enumerate(self.beat_layers):
            result = layer(beat_emb, attention_map, masking)
            beat_emb = result['output']
            if attention_map:
                weights.append(result['weight']) 
        
        result = {'output': beat_emb}
        if attention_map:
            result['weights'] = weights
        
        return result

    def pitch_forward(self, note_emb, attention_map=False, masking=True):
        emb = self.pos_encoding(note_emb)
        emb = self.emb_dropout(emb)

        # pitch model
        pitch_weights = []
        for _, layer in enumerate(self.pitch_layers):
            pitch_result = layer(emb, attention_map, masking)
            emb = pitch_result['output']
            if attention_map:
                pitch_weights.append(pitch_result['weight'])

        # output layer
        output = self.output_layer(emb)
        output = self.log_softmax(output)
        
        result = {'output': output}
        if attention_map:
            result['weights'] = pitch_weights
        
        return result

    def sampling(self, prime_beat, prime_pitch, chord, topk=None, attention_map=False):
        chord_hidden = self.chord_forward(chord)

        # batch_size * prime_len * num_outputs
        batch_size = prime_pitch.size(0)
        pad_length = self.max_len - prime_pitch.size(1)
        beat_pad = torch.zeros([batch_size, pad_length], dtype=torch.long).to(device)
        beat_result = torch.cat([prime_beat, beat_pad], dim=1)

        # sampling phase
        for i in range(prime_pitch.size(1), self.max_len):
            beat_dec_result = self.beat_forward(beat_result, chord_hidden, attention_map, masking=True)
            beat_out = self.beatout_layer(beat_dec_result['output'])
            beat_out = self.log_softmax(beat_out)
            # idx = torch.argmax(beat_out[:, i - 1, :], dim=1)
            if topk is None:
            # if i % (4 * self.frame_per_bar) < (2 * self.frame_per_bar):
                idx = torch.argmax(beat_out[:, i - 1, :], dim=1)
            else:
                top3_probs, top3_idxs = torch.topk(beat_out[:, i - 1, :], 3, dim=-1)
                idx = torch.gather(top3_idxs, 1, torch.multinomial(F.softmax(top3_probs, dim=-1), 1)).squeeze()
            beat_result[:, i] = idx

        # # beam search
        # beat_result = self.beam_search(prime_beat, chord_hidden, mode='beat', k=3, attention_map=attention_map)

        beat_dict = self.beat_forward(beat_result, chord_hidden, attention_map, masking=True)
        beat_out = self.beatout_layer(beat_dict['output'])
        beat_out = self.log_softmax(beat_out)
        idx = torch.argmax(beat_out[:, -1, :], dim=1)
        beat_temp = torch.cat([beat_result[:, 1:], idx.unsqueeze(-1)], dim=1)
        beat_enc_dict = self.beat_forward(beat_temp, chord_hidden, attention_map, masking=False)
        beat_emb = beat_enc_dict['output']

        pitch_pad = torch.ones([batch_size, pad_length], dtype=torch.long).to(device)
        pitch_pad *= (self.num_pitch - 1)
        pitch_result = torch.cat([prime_pitch, pitch_pad], dim=1)
        for i in range(prime_pitch.size(1), self.max_len):
            note_emb = self.note_emb(pitch_result)
            emb = torch.cat([note_emb, chord_hidden[0], chord_hidden[1], beat_emb], -1)
            emb *= torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float))
            pitch_dict = self.pitch_forward(emb, attention_map)
            # pitch_dict = self.pitch_forward(emb, beat_emb, attention_map)
            if topk is None:
                idx = torch.argmax(pitch_dict['output'][:, i - 1, :], dim=1)
            else:
                topk_probs, topk_idxs = torch.topk(pitch_dict['output'][:, i-1, :], topk, dim=-1)
                idx = torch.gather(topk_idxs, 1, torch.multinomial(F.softmax(topk_probs, dim=-1), 1)).squeeze()
            pitch_result[:, i] = idx

        # pitch_result = self.beam_search(prime_pitch, chord_hidden, enc_output=beat_emb, mode='pitch',
        #                                 k=5, attention_map=attention_map)

        result = {'beat': beat_result,
                  'pitch': pitch_result}
        if attention_map:
            result['weights_bdec'] = beat_dict['weights']
            result['weights_benc'] = beat_enc_dict['weights']
            result['weights_pitch'] = pitch_dict['weights']
        return result


    def beam_search(self, prime, chord_hidden, enc_output=None,
                    mode='beat', k=3, attention_map=False):
        batch_size = prime.size(0)
        pad_length = self.max_len - prime.size(1)
        pad = torch.zeros([batch_size, pad_length], dtype=torch.long).to(device)
        beam_result = torch.cat([prime, pad], dim=1)
        pos_index = (torch.LongTensor(range(batch_size)) * k).view(-1, 1).to(device)

        sequence_scores = torch.Tensor(batch_size * k, 1)
        sequence_scores.fill_(-float('Inf'))
        sequence_scores.index_fill_(0, torch.LongTensor([i * k for i in range(0, batch_size)]), 0.0)

        beam_result = beam_result.unsqueeze(1).repeat(1, k, 1).view(batch_size * k, -1)
        chord_size = chord_hidden[0].size()
        chord_hidden = tuple([chord_hidden[d].unsqueeze(1).repeat(1, k, 1, 1).view(-1, *chord_size[1:]) for d in range(len(chord_hidden))])

        if enc_output is not None:
            enc_output = enc_output.unsqueeze(1).repeat(1, k, 1, 1).view(-1, *enc_output.size()[1:])
        for i in range(prime.size(1), self.max_len):
            if mode == 'beat':
                dec_result = self.beat_forward(beam_result, chord_hidden, attention_map, masking=True)
                out = self.beatout_layer(dec_result['output'])
                log_softmax_output = self.log_softmax(out)[:, i - 1]
                vocab_size = self.num_beat
            elif mode == 'pitch':
                note_emb = self.note_emb(beam_result)
                emb = torch.cat([note_emb, chord_hidden[0], chord_hidden[1], enc_output], -1)
                emb *= torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float))
                log_softmax_output = self.pitch_forward(emb, attention_map)['output'][:, i - 1]
                vocab_size = self.num_pitch

            sequence_scores = torch.cat(vocab_size * [sequence_scores], dim=1).to(device)
            sequence_scores += log_softmax_output
            scores, candidates = sequence_scores.view(batch_size, -1).topk(k, dim=1)

            select_idx = ((candidates // vocab_size) + pos_index).view(-1)
            beam_result = torch.index_select(beam_result, 0, select_idx)
            beam_result[:, i] = (candidates % vocab_size).view(batch_size * k)
            sequence_scores = scores.view(batch_size * k, 1)

        return torch.index_select(beam_result, 0, pos_index.squeeze())

class MusicTransformerCE(nn.Module):
    def __init__(self, num_events=130, frame_per_bar=16, num_bars=8,
                 chord_emb_size=128, note_emb_size=128, hidden_dim=128,
                 key_dim=128, value_dim=128, num_layers=6, num_heads=4,
                 input_dropout=0.0, layer_dropout=0.0, attention_dropout=0.0):
        super(MusicTransformerCE, self).__init__()

        self.num_events = num_events
        self.max_len = frame_per_bar * num_bars
        self.frame_per_bar = frame_per_bar
        self.num_chords = 12

        self.hidden_dim = hidden_dim
        self.note_emb = nn.Embedding(self.num_events, note_emb_size)
        self.chord_emb = nn.Embedding(self.num_chords + 1, chord_emb_size, padding_idx=12)

        self.pos_encoding = DynamicPositionEmbedding(self.hidden_dim, self.max_len)
        # self.chord_pos_enc = DynamicPositionEmbedding(chord_emb_size, self.max_len)

        self.emb_dropout = nn.Dropout(input_dropout)

        # Encoding layers
        enc_params = (
            chord_emb_size,
            hidden_dim,
            key_dim,
            value_dim,
            # chord_emb_size,
            # key_dim // 2,
            # value_dim // 2,
            num_heads,
            self.max_len,
            layer_dropout,
            attention_dropout
        )
        self.enc_layers = nn.ModuleList([
            EncoderLayer1(*enc_params) for _ in range(num_layers)
        ])

        # Decoding layers
        dec_params = (
            note_emb_size,
            # note_emb_size + chord_emb_size,
            hidden_dim,
            key_dim,
            value_dim,
            num_heads,
            self.max_len,
            layer_dropout,
            attention_dropout
        )
        self.dec_layers = nn.ModuleList([
            DecoderLayer1(*dec_params) for _ in range(num_layers)
            # DecoderLayer0(*dec_params) for _ in range(num_layers)
        ])
        
        # output layer
        self.output_layer = nn.Linear(hidden_dim, self.num_events)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
        self.output_layer.weight = self.note_emb.weight
        for i in range(1, num_layers):
            self.enc_layers[i].mha.relative_embedding = self.enc_layers[i-1].mha.relative_embedding
            self.dec_layers[i].self_mha.relative_embedding = self.dec_layers[i-1].self_mha.relative_embedding
            self.dec_layers[i].mha2.relative_embedding = self.dec_layers[i-1].mha2.relative_embedding

    # def forward(self, note, chord_root, chord_interval, attention_map=False):
    def forward(self, note, chord, attention_map=False):
        size = chord.size()
        chord_emb = self.chord_emb(chord).view(size[0], size[1], -1)
        chord_emb *= torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float))
        chord_emb = self.pos_encoding(chord_emb[:, 1:, :])
        # chord_emb = self.chord_pos_enc(chord_emb[:, 1:, :])
        chord_emb = self.emb_dropout(chord_emb)

        weights = []
        for _, enc_layer in enumerate(self.enc_layers):
            chord_result = enc_layer(chord_emb, attention_map)
            chord_emb = chord_result['output']
            if attention_map:
                weights.append(chord_result['weight'])

        note_emb = self.note_emb(note)
        # note_emb = torch.cat([note_emb, chord_emb], -1)
        note_emb *= torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float))
        note_emb = self.pos_encoding(note_emb)
        note_emb = self.emb_dropout(note_emb)

        weights1 = []
        weights2 = []
        for _, dec_layer in enumerate(self.dec_layers):
            note_result = dec_layer(note_emb, chord_emb, attention_map)
            note_emb = note_result['output']
            if attention_map:
                weights1.append(note_result['weight1'])
                weights2.append(note_result['weight2'])

        # output layer
        output = self.output_layer(note_emb)
        output = self.log_softmax(output)

        result = {'output': output}
        if attention_map:
            result['weights'] = weights
            result['weights1'] = weights1
            result['weights2'] = weights2
        return result

    def sampling(self, prime_idx, chord, topk=None, attention_map=False):
        # batch_size * prime_len * num_outputs
        batch_size = prime_idx.size(0)
        pad_length = self.max_len - prime_idx.size(1)
        pad = torch.ones([batch_size, pad_length], dtype=torch.long).to(device)
        pad *= (self.num_events - 1)
        result = torch.cat([prime_idx, pad], dim=1)
        
        # sampling phase
        for i in range(prime_idx.size(1), self.max_len):
            result_dict = self.forward(result, chord, attention_map)
            output = result_dict['output']
            if topk is None:
                idx = torch.argmax(output[:, i - 1, :], dim=1)
            else:
                topk_probs, topk_idxs = torch.topk(output[:, i-1, :], topk, dim=-1)
                idx = torch.gather(topk_idxs, 1, torch.multinomial(F.softmax(topk_probs, dim=-1), 1)).squeeze()
            result[:, i] = idx

        result_dict['output'] = result
        return result_dict


class BeatMusicTransformer(nn.Module):
    def __init__(self, num_pitch=89, frame_per_bar=16, num_bars=8,
                 chord_emb_size=128, note_emb_size=128, hidden_dim=128,
                 key_dim=128, value_dim=128, num_layers=6, num_heads=4,
                 input_dropout=0.0, layer_dropout=0.0, attention_dropout=0.0):
        super(BeatMusicTransformer, self).__init__()

        self.max_len = frame_per_bar * num_bars
        self.frame_per_bar = frame_per_bar
        self.num_chords = 12
        self.num_pitch = num_pitch
        self.num_beat = 3

        self.hidden_dim = hidden_dim
        # self.note_emb_size = embedding_size
        # self.chord_emb_size = chord_emb

        # embedding layer
        # self.emb_mat = nn.Parameter(torch.randn(self.num_outputs, embedding_size, dtype=torch.float, requires_grad=True))
        self.beat_emb = nn.Embedding(self.num_beat, self.hidden_dim // 2)
        self.note_emb = nn.Embedding(self.num_pitch, self.hidden_dim)
        self.chord_emb = nn.Embedding(self.num_chords + 1, chord_emb_size, padding_idx=12)
        # chord_emb_size == self.hidden_dim // 2 if chord add
        # chord_emb_size == self.hidden_dim // 10 if chord concat

        self.pos_half_dim = DynamicPositionEmbedding(self.hidden_dim // 2, self.max_len)
        self.pos_encoding = DynamicPositionEmbedding(self.hidden_dim, self.max_len)

        # embedding dropout
        self.emb_dropout = nn.Dropout(input_dropout)

        chord_params = (
            hidden_dim // 2,
            hidden_dim // 2,
            key_dim // 2,
            value_dim // 2,
            num_heads,
            self.max_len,
            layer_dropout,
            attention_dropout
        )
        self.chord_layers = nn.ModuleList([
            EncoderLayer2(*chord_params) for _ in range(num_layers)
        ])
        
        beat_params = (
            hidden_dim // 2,
            hidden_dim // 2,
            key_dim // 2,
            value_dim // 2,
            num_heads,
            self.max_len,
            layer_dropout,
            attention_dropout
        )
        self.beat_layers = nn.ModuleList([
            EncoderLayer2(*beat_params) for _ in range(num_layers // 2)
        ])

        pitch_params = (
            hidden_dim,
            hidden_dim,
            key_dim,
            value_dim,
            num_heads,
            self.max_len,
            layer_dropout,
            attention_dropout
        )
        self.pitch_layers = nn.ModuleList([
            DecoderLayer2(*pitch_params) for _ in range(num_layers)
        ])

        # output layer
        self.beat_output = nn.Linear(hidden_dim // 2, self.num_beat)
        self.pitch_output = nn.Linear(hidden_dim, self.num_pitch)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    # beat_length = pitch_length + 1
    def forward(self, beat, pitch, chord, attention_map=False):
        beat_emb = self.beat_emb(beat)
        beat_emb *= torch.sqrt(torch.tensor(self.hidden_dim // 2, dtype=torch.float))
        beat_dec = self.pos_half_dim(beat_emb[:, :-1])
        beat_dec = self.emb_dropout(beat_dec)
        beat_emb = self.pos_half_dim(beat_emb[:, 1:])
        beat_emb = self.emb_dropout(beat_emb)

        for _, beat_layer in enumerate(self.beat_layers):
            # Beat Decoder
            beat_dec = beat_layer(beat_dec, attention_map, masking=True)
            # Beat Encoder
            beat_emb = beat_layer(beat_emb, attention_map, masking=False)
        beat_out = self.beat_output(beat_dec)
        beat_out = self.log_softmax(beat_out)

        chord_emb = self.chord_forward(chord[:, 1:], attention_map)
        enc_output = torch.cat([chord_emb, beat_emb], -1)

        output = self.pitch_forward(pitch, enc_output)
        
        return beat_out, output

    def chord_forward(self, chord, attention_map=False):
        size = chord.size()
        chord_emb = self.chord_emb(chord).view(size[0], size[1], -1)
        chord_emb *= torch.sqrt(torch.tensor(self.hidden_dim // 2, dtype=torch.float))
        chord_emb = self.pos_half_dim(chord_emb)
        chord_emb = self.emb_dropout(chord_emb)
        
        for _, chord_layer in enumerate(self.chord_layers):
            chord_emb = chord_layer(chord_emb, attention_map, masking=False)
        
        return chord_emb
    
    def beat_forward(self, beat, decoder=False):
        beat_emb = self.beat_emb(beat)
        beat_emb *= torch.sqrt(torch.tensor(self.hidden_dim // 2, dtype=torch.float))
        beat_emb = self.pos_half_dim(beat_emb)
        beat_emb = self.emb_dropout(beat_emb)
        for _, beat_layer in enumerate(self.beat_layers):
            beat_emb = beat_layer(beat_emb, masking=decoder)
        
        if decoder:
            beat_out = self.beat_output(beat_emb)
            beat_out = self.log_softmax(beat_out)            
            return beat_out
        else:
            return beat_emb
    
    def pitch_forward(self, pitch, enc_output):
        note_emb = self.note_emb(pitch)
        note_emb *= torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float))
        note_emb = self.pos_encoding(note_emb)
        note_emb = self.emb_dropout(note_emb)
        
        for _, pitch_layer in enumerate(self.pitch_layers):
            note_emb = pitch_layer(note_emb, enc_output)

        # output layer
        output = self.pitch_output(note_emb)
        output = self.log_softmax(output)
        
        return output

    def sampling(self, prime_beat, prime_pitch, chord, topk=None):
        # batch_size * prime_len * num_outputs
        batch_size = prime_pitch.size(0)
        pad_length = self.max_len - prime_pitch.size(1)
        beat_pad = torch.zeros([batch_size, pad_length], dtype=torch.long).to(device)
        pitch_pad = torch.ones([batch_size, pad_length], dtype=torch.long).to(device)
        pitch_pad *= (self.num_pitch - 1)
        beat_result = torch.cat([prime_beat, beat_pad], dim=1)
        pitch_result = torch.cat([prime_pitch, pitch_pad], dim=1)

        # sampling phase
        for i in range(prime_beat.size(1), self.max_len):
            beat_out = self.beat_forward(beat_result, decoder=True)
            idx = torch.argmax(beat_out[:, i - 1, :], dim=1)
            # idx = torch.multinomial(F.softmax(beat_out[:, i - 1, :], dim=-1), 1).squeeze()
            beat_result[:, i] = idx
        
        beat_out = self.beat_forward(beat_result, decoder=True)
        idx = torch.argmax(beat_out[:, -1, :], dim=1)
        
        chord_emb = self.chord_forward(chord[:, 1:])
        beat_emb = self.beat_forward(torch.cat([beat_result[:, 1:], idx.unsqueeze(-1)], dim=1), decoder=False)
        enc_output = torch.cat([chord_emb, beat_emb], -1)
        
        for i in range(prime_pitch.size(1), self.max_len):
            pitch_out = self.pitch_forward(pitch_result, enc_output)
            if topk is None:
                idx = torch.argmax(pitch_out[:, i - 1, :], dim=-1)
            else:
                topk_probs, topk_idxs = torch.topk(pitch_out[:, i-1, :], topk, dim=-1)
                idx = torch.gather(topk_idxs, 1, torch.multinomial(F.softmax(topk_probs, dim=-1), 1)).squeeze()
            pitch_result[:, i] = idx

        return beat_result, pitch_result