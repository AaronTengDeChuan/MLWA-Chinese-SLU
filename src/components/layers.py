# -*- coding: utf-8 -*-

"""
@CreateTime :       2020/4/3 15:13
@Author     :       dcteng
@File       :       layers.py
@Software   :       PyCharm
@Framework  :       Pytorch
@LastModify :       2020/4/3 15:13
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.components import operation as op

MASK_VALUE = -2 ** 32 + 1


class NullOp(nn.Module):
    def forward(self, input):
        return input


class BilinearLayer(nn.Module):
    def __init__(self, x_size, y_size, class_num):
        super(BilinearLayer, self).__init__()
        self.linear = nn.Linear(y_size, x_size * class_num)
        self.class_num = class_num

    def forward(self, x, y):
        """
        x = batch * h1
        y = batch * h2
        """
        Wy = self.linear(y)
        Wy = Wy.view(Wy.size(0), self.class_num, x.size(1))
        xWy = torch.sum(x.unsqueeze(1).expand_as(Wy) * Wy, dim=2)
        return xWy # size = batch * class_num


# TODO: Related to Fusion
class FusionLayer(nn.Module):
    def __init__(self, x_size, y_size, dropout_rate, fusion_type, class_num):
        super(FusionLayer, self).__init__()

        self.__x_size = x_size
        self.__y_size = y_size
        self.__fusion_type = fusion_type

        assert fusion_type in ["add", "rate", "rate_linear", "rate_bilinear", "bilinear"]

        self.__dropout_layer = nn.Dropout(dropout_rate)

        # if fusion_type != "bilinear": self.__linear_layer = nn.Linear(x_size, class_num)
        self.__linear_layer = nn.Linear(x_size, class_num)

        if fusion_type == "rate": self.__fusion_rate = nn.Parameter(torch.randn(1), requires_grad=True)
        elif fusion_type == "rate_linear": self.__fusion_linear = nn.Linear(x_size, 1)
        elif fusion_type == "rate_bilinear": self.__fusion_bilinear = BilinearLayer(x_size, y_size, 1)
        # elif fusion_type == "bilinear": self.__bilinear = BilinearLayer(x_size, y_size, class_num)
        elif fusion_type == "bilinear": self.__bilinear = nn.Bilinear(x_size, y_size, x_size)

    def forward(self, x, y=None, dropout=True):
        """
            x = batch * h1
            y = batch * h2
        """
        assert len(x.shape) == 2 and x.size(1) == self.__x_size

        if y is None:
            return self.__linear_layer(self.__dropout_layer(x) if dropout else x)

        assert len(y.shape) == 2 and y.size(1) == self.__y_size

        # if self.__fusion_type == "bilinear":
        #     dropout_x = self.__dropout_layer(x) if dropout else x
        #     dropout_y = self.__dropout_layer(y) if dropout else y
        #     return self.__bilinear(dropout_x, dropout_y)
        if self.__fusion_type == "bilinear":
            fusion_rate = torch.sigmoid(self.__bilinear(x, y))

        if self.__fusion_type == "rate": fusion_rate = torch.sigmoid(self.__fusion_rate)
        elif self.__fusion_type == "rate_linear": fusion_rate = torch.sigmoid(self.__fusion_linear(x))
        elif self.__fusion_type == "rate_bilinear": fusion_rate = torch.sigmoid(self.__fusion_bilinear(x, y))

        if self.__fusion_type == "add":
            fusion = x + y
        else:
            fusion = fusion_rate * x + (1 - fusion_rate) * y
        if dropout: fusion = self.__dropout_layer(fusion)

        return self.__linear_layer(fusion)


# TODO: Related to Encoder and Decoder

class EmbeddingCollection(nn.Module):
    """
    TODO: Provide position vector encoding
    Provide word vector encoding.
    """

    def __init__(self, input_dim, embedding_dim, pretrain_embedding=None, max_len=5000):
        super(EmbeddingCollection, self).__init__()

        self.__input_dim = input_dim
        # Here embedding_dim must be an even embedding.
        self.__embedding_dim = embedding_dim
        self.__max_len = max_len

        # Word vector encoder.
        self.__embedding_layer = nn.Embedding(
            self.__input_dim, self.__embedding_dim
        )
        if pretrain_embedding is not None:
            self.__embedding_layer.weight.data.copy_(torch.from_numpy(pretrain_embedding))

    def forward(self, input_x):
        # Get word vector encoding.
        embedding_x = self.__embedding_layer(input_x)

        # Board-casting principle.
        return embedding_x


class LSTMEncoder(nn.Module):
    """
    Encoder structure based on bidirectional LSTM.
    """

    def __init__(self, embedding_dim, hidden_dim, dropout_rate, bidirectional=True, extra_dim=None):
        super(LSTMEncoder, self).__init__()

        # Parameter recording.
        self.__embedding_dim = embedding_dim
        self.__hidden_dim = hidden_dim // 2 if bidirectional else hidden_dim
        self.__dropout_rate = dropout_rate
        self.__bidirectional = bidirectional
        self.__extra_dim = extra_dim

        lstm_input_dim = self.__embedding_dim + (0 if self.__extra_dim is None else self.__extra_dim)

        # Network attributes.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(input_size=lstm_input_dim, hidden_size=self.__hidden_dim, batch_first=True,
                                    bidirectional=self.__bidirectional, dropout=self.__dropout_rate, num_layers=1)

    def forward(self, embedded_text, seq_lens, extra_input=None):
        """ Forward process for LSTM Encoder.

        (batch_size, max_sent_len)
        -> (batch_size, max_sent_len, word_dim)
        -> (batch_size, max_sent_len, hidden_dim)
        -> (total_word_num, hidden_dim)

        :param embedded_text: padded and embedded input text.
        :param seq_lens: is the length of original input text.
        :return: is encoded word hidden vectors.
        """

        # Concatenate information tensor if possible.
        if extra_input is not None:
            input_tensor = torch.cat([embedded_text, extra_input], dim=-1)
        else:
            input_tensor = embedded_text

        # Padded_text should be instance of LongTensor.
        dropout_text = self.__dropout_layer(input_tensor)

        # # Pack and Pad process for input of variable length.
        # packed_text = pack_padded_sequence(dropout_text, seq_lens, batch_first=True)
        # lstm_hiddens, (h_last, c_last) = self.__lstm_layer(packed_text)
        # padded_hiddens, _ = pad_packed_sequence(lstm_hiddens, batch_first=True)

        padded_hiddens, _ = op.pack_and_pad_sequences_for_rnn(dropout_text,
                                                              torch.tensor(seq_lens, device=dropout_text.device),
                                                              self.__lstm_layer)

        # return torch.cat([padded_hiddens[i][:seq_lens[i], :] for i in range(0, len(seq_lens))], dim=0)
        return padded_hiddens


class LSTMDecoder(nn.Module):
    """
    Decoder structure based on unidirectional LSTM.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate, slot_fusion_type,
                 embedding_dim=None, extra_input_dim=None, extra_hidden_dim=None):
        """ Construction function for Decoder.

        :param input_dim: input dimension of Decoder. In fact, it's encoder hidden size.
        :param hidden_dim: hidden dimension of iterative LSTM.
        :param output_dim: output dimension of Decoder. In fact, it's total number of intent or slot.
        :param dropout_rate: dropout rate of network which is only useful for embedding.
        :param embedding_dim: if it's not None, the input and output are relevant.
        :param extra_dim: if it's not None, the decoder receives information tensors.
        """

        super(LSTMDecoder, self).__init__()

        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate
        self.__embedding_dim = embedding_dim
        self.__extra_input_dim = extra_input_dim
        self.__extra_hidden_dim = extra_hidden_dim

        # assert self.__hidden_dim == self.__extra_hidden_dim

        # If embedding_dim is not None, the output and input
        # of this structure is relevant.
        if self.__embedding_dim is not None:
            self.__embedding_layer = nn.Embedding(output_dim, embedding_dim)
            self.__init_tensor = nn.Parameter(torch.randn(1, self.__embedding_dim), requires_grad=True)

        # Make sure the input dimension of iterative LSTM.
        lstm_input_dim = self.__input_dim + \
                         (0 if self.__extra_input_dim is None else self.__extra_input_dim) + \
                         (0 if self.__embedding_dim is None else self.__embedding_dim)

        # Network parameter definition.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(input_size=lstm_input_dim, hidden_size=self.__hidden_dim, batch_first=True,
                                    bidirectional=False, dropout=self.__dropout_rate, num_layers=1)

        self.__slot_fusion_layer = FusionLayer(self.__hidden_dim, self.__hidden_dim,
                                               self.__dropout_rate, slot_fusion_type,
                                               class_num=self.__output_dim)

    def forward(self, encoded_hiddens, seq_lens, forced_input=None, extra_input=None,
                extra_hidden=None, attention_module=None):
        """ Forward process for decoder.

        :param encoded_hiddens: is encoded hidden tensors produced by encoder.
        :param seq_lens: is a list containing lengths of sentence.
        :param forced_input: is truth values of label, provided by teacher forcing.
        :param extra_input: comes from another decoder as information tensor.
        :return: is distribution of prediction labels.
        """
        if extra_hidden is not None and attention_module is not None:
            extra_hidden_mask = extra_hidden[1]
            extra_hidden = extra_hidden[0]
            assert extra_hidden_mask.shape == extra_hidden.shape[:-1]

        assert extra_hidden is None \
               or (extra_hidden.shape[0] == encoded_hiddens.shape[0] and extra_hidden.shape[-1] == self.__extra_hidden_dim)

        # Concatenate information tensor if possible.
        if extra_input is not None:
            input_tensor = torch.cat([encoded_hiddens, extra_input], dim=1)
        else:
            input_tensor = encoded_hiddens

        output_tensor_list, sent_start_pos = [], 0
        lstm_output_tensor_list = []
        if self.__embedding_dim is None or forced_input is not None:

            for sent_i in range(0, len(seq_lens)):
                sent_end_pos = sent_start_pos + seq_lens[sent_i]

                # Segment input hidden tensors.
                seg_hiddens = input_tensor[sent_start_pos: sent_end_pos, :]

                if self.__embedding_dim is not None and forced_input is not None:
                    if seq_lens[sent_i] > 1:
                        seg_forced_input = forced_input[sent_start_pos: sent_end_pos]
                        seg_forced_tensor = self.__embedding_layer(seg_forced_input).view(seq_lens[sent_i], -1)
                        seg_prev_tensor = torch.cat([self.__init_tensor, seg_forced_tensor[:-1, :]], dim=0)
                    else:
                        seg_prev_tensor = self.__init_tensor

                    # Concatenate forced target tensor.
                    combined_input = torch.cat([seg_hiddens, seg_prev_tensor], dim=1)
                else:
                    combined_input = seg_hiddens
                dropout_input = self.__dropout_layer(combined_input)

                lstm_out, _ = self.__lstm_layer(dropout_input.view(1, seq_lens[sent_i], -1))

                lstm_out = lstm_out.view(seq_lens[sent_i], -1)
                lstm_output_tensor_list.append(lstm_out)

                if self.__extra_hidden_dim is not None:
                    seg_extra_hidden = extra_hidden[sent_start_pos: sent_end_pos, :]

                    if attention_module is not None:
                        seg_extra_hidden_mask = extra_hidden_mask[sent_start_pos: sent_end_pos]

                        dropout_lstm_out = self.__dropout_layer(lstm_out)
                        dropout_seg_extra_hidden = self.__dropout_layer(seg_extra_hidden)
                        seg_extra_hidden = attention_module(dropout_lstm_out.unsqueeze(1),
                                                            dropout_seg_extra_hidden,
                                                            dropout_seg_extra_hidden,
                                                            mmask=seg_extra_hidden_mask.unsqueeze(1)).squeeze(1)
                linear_out = self.__slot_fusion_layer(lstm_out,
                                                      y=seg_extra_hidden if self.__extra_hidden_dim else None,
                                                      dropout=False)
                output_tensor_list.append(linear_out)
                sent_start_pos = sent_end_pos
        else:
            for sent_i in range(0, len(seq_lens)):
                prev_tensor = self.__init_tensor

                # It's necessary to remember h and c state
                # when output prediction every single step.
                last_h, last_c = None, None

                sent_end_pos = sent_start_pos + seq_lens[sent_i]
                for word_i in range(sent_start_pos, sent_end_pos):
                    seg_input = input_tensor[[word_i], :]
                    combined_input = torch.cat([seg_input, prev_tensor], dim=1)
                    dropout_input = self.__dropout_layer(combined_input).view(1, 1, -1)

                    if last_h is None and last_c is None:
                        lstm_out, (last_h, last_c) = self.__lstm_layer(dropout_input)
                    else:
                        lstm_out, (last_h, last_c) = self.__lstm_layer(dropout_input, (last_h, last_c))

                    lstm_out = lstm_out.view(1, -1)
                    lstm_output_tensor_list.append(lstm_out)

                    if self.__extra_hidden_dim is not None:
                        seg_extra_hidden = extra_hidden[[word_i], :]

                        if attention_module is not None:
                            seg_extra_hidden_mask = extra_hidden_mask[[word_i]]

                            dropout_lstm_out = self.__dropout_layer(lstm_out)
                            dropout_seg_extra_hidden = self.__dropout_layer(seg_extra_hidden)
                            seg_extra_hidden = attention_module(dropout_lstm_out.unsqueeze(1),
                                                                dropout_seg_extra_hidden,
                                                                dropout_seg_extra_hidden,
                                                                mmask=seg_extra_hidden_mask.unsqueeze(1)).squeeze(1)
                    linear_out = self.__slot_fusion_layer(lstm_out,
                                                          y=seg_extra_hidden if self.__extra_hidden_dim else None,
                                                          dropout=False)
                    output_tensor_list.append(linear_out)

                    _, index = linear_out.topk(1, dim=1)
                    prev_tensor = self.__embedding_layer(index).view(1, -1)
                sent_start_pos = sent_end_pos

        return torch.cat(output_tensor_list, dim=0), torch.cat(lstm_output_tensor_list, dim=0)


class QKVAttention(nn.Module):
    """
    Attention mechanism based on Query-Key-Value architecture. And
    especially, when query == key == value, it's self-attention.
    """

    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, output_dim, dropout_rate, input_linear=True, bilinear=False):
        super(QKVAttention, self).__init__()

        # Record hyper-parameters.
        self.__query_dim = query_dim
        self.__key_dim = key_dim
        self.__value_dim = value_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate
        self.__input_linear = input_linear
        self.__bilinear = bilinear

        # Declare network structures.
        if input_linear and not bilinear:
            self.__query_layer = nn.Linear(self.__query_dim, self.__hidden_dim)
            self.__key_layer = nn.Linear(self.__key_dim, self.__hidden_dim)
            self.__value_layer = nn.Linear(self.__value_dim, self.__output_dim)
        elif bilinear:
            self.__linear = nn.Linear(self.__query_dim, self.__key_dim)

        self.__dropout_layer = nn.Dropout(p=self.__dropout_rate)

    def forward(self, input_query, input_key, input_value, mmask=None):
        """ The forward propagation of attention.

        Here we require the first dimension of input key
        and value are equal.

        :param input_query: is query tensor, (n, d_q)
        :param input_key:  is key tensor, (m, d_k)
        :param input_value:  is value tensor, (m, d_v)
        :return: attention based tensor, (n, d_h)
        """

        # Linear transform to fine-tune dimension.
        linear_query = self.__query_layer(input_query) if self.__input_linear and not self.__bilinear else input_query
        linear_key = self.__key_layer(input_key) if self.__input_linear and not self.__bilinear else input_key
        linear_value = self.__value_layer(input_value) if self.__input_linear and not self.__bilinear else input_value

        if self.__input_linear and not self.__bilinear:
            score_tensor = torch.matmul(linear_query, linear_key.transpose(-2, -1)) / math.sqrt(
                self.__hidden_dim if self.__input_linear else self.__query_dim)
        elif self.__bilinear:
            score_tensor = torch.matmul(self.__linear(linear_query), linear_key.transpose(-2, -1))

        if mmask is not None:
            assert mmask.shape == score_tensor.shape
            score_tensor = mmask * score_tensor + (1 - mmask) * MASK_VALUE

        score_tensor = F.softmax(score_tensor, dim=-1)
        forced_tensor = torch.matmul(score_tensor, linear_value)
        # forced_tensor = self.__dropout_layer(forced_tensor)

        return forced_tensor


class SelfAttention(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(SelfAttention, self).__init__()

        # Record parameters.
        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Record network parameters.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__attention_layer = QKVAttention(
            self.__input_dim, self.__input_dim, self.__input_dim,
            self.__hidden_dim, self.__output_dim, self.__dropout_rate
        )

    def forward(self, input_x, seq_lens, mmask=None):
        dropout_x = self.__dropout_layer(input_x)
        attention_x = self.__attention_layer(dropout_x, dropout_x, dropout_x, mmask=mmask)

        # flat_x = torch.cat(
        #     [attention_x[i][:seq_lens[i], :] for
        #      i in range(0, len(seq_lens))], dim=0
        # )

        # return flat_x
        return attention_x


class FFN(nn.Module):

    def __init__(self, input_dim, out_dim_0, out_dim_1):
        super(FFN, self).__init__()

        self.input_dim = input_dim
        self.out_dim_0 = out_dim_0
        self.out_dim_1 = out_dim_1

        self.linear_1 = nn.Linear(in_features=self.input_dim, out_features=self.out_dim_0)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(in_features=self.out_dim_0, out_features=self.out_dim_1)

    def forward(self, x):
        y = self.linear_1(x)
        self.relu(y)
        z = self.linear_2(y)

        return z

class AttentiveModule(nn.Module):

    def __init__(self, query_dim, key_dim, value_dim, output_dim, dropout_rate, bilinear=False):
        super(AttentiveModule, self).__init__()

        # Record hyper-parameters.
        self.__query_dim = query_dim
        self.__key_dim = key_dim
        self.__value_dim = value_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        self.__attention_layer = QKVAttention(
            self.__query_dim, self.__key_dim, self.__value_dim, 0, 0, self.__dropout_rate,
            input_linear=False, bilinear=bilinear
        )

        self.__ffn = FFN(self.__value_dim, self.__output_dim, self.__output_dim)

    def forward(self, input_query, input_key, input_value, mmask=None):
        """

        :param input_query:
        :param input_key:
        :param input_value:
        :param mmask:
        :return:
        """
        att = self.__attention_layer(input_query, input_key, input_value, mmask=mmask)

        z = self.__ffn(att)

        return z


class MLPAttention(nn.Module):

    def __init__(self, input_dim, dropout_rate):
        super(MLPAttention, self).__init__()

        # Record parameters
        self.__input_dim = input_dim
        self.__dropout_rate = dropout_rate

        # Define network structures
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__sent_attention = nn.Linear(self.__input_dim, 1, bias=False)

    def forward(self, encoded_hiddens, rmask=None):
        """
        Merge a sequence of word representations as a sentence representation.
        :param encoded_hiddens: a tensor with shape of [bs, max_len, dim]
        :param rmask: a mask tensor with shape of [bs, max_len]
        :return:
        """
        # TODO: Do dropout ?
        dropout_input = self.__dropout_layer(encoded_hiddens)
        score_tensor = self.__sent_attention(dropout_input).squeeze(-1)

        if rmask is not None:
            assert score_tensor.shape == rmask.shape, "{} vs {}".format(score_tensor.shape, rmask.shape)
            score_tensor = rmask * score_tensor + (1 - rmask) * MASK_VALUE

        score_tensor = F.softmax(score_tensor, dim=-1)
        # matrix multiplication: [bs, 1, max_len] * [bs, max_len, dim] => [bs, 1, dim]
        sent_output = torch.matmul(score_tensor.unsqueeze(1), dropout_input).squeeze(1)

        return sent_output


# TODO: Related GNN layers

class MultiHeadAtt(nn.Module):
    def __init__(self, nhid, keyhid, nhead=10, head_dim=10, dropout=0.1, if_g=False):
        super(MultiHeadAtt, self).__init__()

        if if_g:
            self.WQ = nn.Conv2d(nhid * 3, nhead * head_dim, 1)
        else:
            self.WQ = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WK = nn.Conv2d(keyhid, nhead * head_dim, 1)
        self.WV = nn.Conv2d(keyhid, nhead * head_dim, 1)
        self.WO = nn.Conv2d(nhead * head_dim, nhid, 1)

        self.drop = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(nhid)

        self.nhid, self.nhead, self.head_dim = nhid, nhead, head_dim

    def forward(self, query_h, value, mask, query_g=None):

        if not (query_g is None):
            query = torch.cat([query_h, query_g], -1)
        else:
            query = query_h
        query = query.permute(0, 2, 1)[:, :, :, None]
        value = value.permute(0, 3, 1, 2)

        residual = query_h
        nhid, nhead, head_dim = self.nhid, self.nhead, self.head_dim

        B, QL, H = query_h.shape

        _, _, VL, VD = value.shape  # VD = 1 or VD = QL

        assert VD == 1 or VD == QL
        # q: (B, H, QL, 1)
        # v: (B, H, VL, VD)
        q, k, v = self.WQ(query), self.WK(value), self.WV(value)

        q = q.view(B, nhead, head_dim, 1, QL)
        k = k.view(B, nhead, head_dim, VL, VD)
        v = v.view(B, nhead, head_dim, VL, VD)

        alpha = (q * k).sum(2, keepdim=True) / np.sqrt(head_dim)
        alpha = alpha.masked_fill(mask[:, None, None, :, :], -np.inf)
        alpha = self.drop(F.softmax(alpha, 3))
        att = (alpha * v).sum(3).view(B, nhead * head_dim, QL, 1)

        output = F.leaky_relu(self.WO(att)).permute(0, 2, 3, 1).view(B, QL, H)
        output = self.norm(output + residual)

        return output


class GloAtt(nn.Module):
    def __init__(self, nhid, nhead=10, head_dim=10, dropout=0.1):
        # Multi-head Self Attention Case 2, a broadcastable query for a sequence key and value
        super(GloAtt, self).__init__()
        self.WQ = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WK = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WV = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WO = nn.Conv2d(nhead * head_dim, nhid, 1)

        self.drop = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(nhid)

        # print('NUM_HEAD', nhead, 'DIM_HEAD', head_dim)
        self.nhid, self.nhead, self.head_dim = nhid, nhead, head_dim

    def forward(self, x, y, mask=None):
        # x: B, H, 1, 1, 1 y: B H L 1
        nhid, nhead, head_dim = self.nhid, self.nhead, self.head_dim
        B, L, H = y.shape

        x = x.permute(0, 2, 1)[:, :, :, None]
        y = y.permute(0, 2, 1)[:, :, :, None]

        residual = x
        q, k, v = self.WQ(x), self.WK(y), self.WV(y)

        q = q.view(B, nhead, 1, head_dim)  # B, H, 1, 1 -> B, N, 1, h
        k = k.view(B, nhead, head_dim, L)  # B, H, L, 1 -> B, N, h, L
        v = v.view(B, nhead, head_dim, L).permute(0, 1, 3, 2)  # B, H, L, 1 -> B, N, L, h

        pre_a = torch.matmul(q, k) / np.sqrt(head_dim)
        if mask is not None:
            pre_a = pre_a.masked_fill(mask[:, None, None, :], -float('inf'))
        alphas = self.drop(F.softmax(pre_a, 3))  # B, N, 1, L
        att = torch.matmul(alphas, v).view(B, -1, 1, 1)  # B, N, 1, h -> B, N*h, 1, 1
        output = F.leaky_relu(self.WO(att)) + residual
        output = self.norm(output.permute(0, 2, 3, 1)).view(B, 1, H)

        return output


class Nodes_Cell(nn.Module):
    def __init__(self, input_h, hid_h, use_global=True, dropout=0.2):
        super(Nodes_Cell, self).__init__()

        self.use_global = use_global
        self.hidden_size = hid_h
        self.Wix = nn.Linear(input_h, hid_h)
        self.Wi2 = nn.Linear(input_h, hid_h)
        self.Wf = nn.Linear(input_h, hid_h)
        self.Wcx = nn.Linear(input_h, hid_h)

        self.drop = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, h, h2, x, glo=None):

        x = self.drop(x)

        if self.use_global:
            glo = self.drop(glo)
            cat_all = torch.cat([h, h2, x, glo], -1)
        else:
            cat_all = torch.cat([h, h2, x], -1)

        ix = torch.sigmoid(self.Wix(cat_all))
        i2 = torch.sigmoid(self.Wi2(cat_all))
        f = torch.sigmoid(self.Wf(cat_all))
        cx = torch.tanh(self.Wcx(cat_all))

        alpha = F.softmax(torch.cat([ix.unsqueeze(1), i2.unsqueeze(1), f.unsqueeze(1)], 1), 1)
        output = (alpha[:, 0] * cx) + (alpha[:, 1] * h2) + (alpha[:, 2] * h)

        return output


class Edges_Cell(nn.Module):
    def __init__(self, input_h, hid_h, use_global=True, dropout=0.2):
        super(Edges_Cell, self).__init__()

        self.use_global = use_global
        self.hidden_size = hid_h
        self.Wi = nn.Linear(input_h, hid_h)
        self.Wf = nn.Linear(input_h, hid_h)
        self.Wc = nn.Linear(input_h, hid_h)

        self.drop = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, h, x, glo=None):

        x = self.drop(x)

        if self.use_global:
            glo = self.drop(glo)
            cat_all = torch.cat([h, x, glo], -1)
        else:
            cat_all = torch.cat([h, x], -1)

        i = torch.sigmoid(self.Wi(cat_all))
        f = torch.sigmoid(self.Wf(cat_all))
        c = torch.tanh(self.Wc(cat_all))

        alpha = F.softmax(torch.cat([i.unsqueeze(1), f.unsqueeze(1)], 1), 1)
        output = (alpha[:, 0] * c) + (alpha[:, 1] * h)

        return output


class Global_Cell(nn.Module):
    def __init__(self, input_h, hid_h, dropout=0.2):
        super(Global_Cell, self).__init__()

        self.hidden_size = hid_h
        self.Wi = nn.Linear(input_h, hid_h)
        self.Wf = nn.Linear(input_h, hid_h)
        self.Wc = nn.Linear(input_h, hid_h)

        self.drop = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, h, x):

        x = self.drop(x)

        cat_all = torch.cat([h, x], -1)
        i = torch.sigmoid(self.Wi(cat_all))
        f = torch.sigmoid(self.Wf(cat_all))
        c = torch.tanh(self.Wc(cat_all))

        alpha = F.softmax(torch.cat([i.unsqueeze(1), f.unsqueeze(1)], 1), 1)
        output = (alpha[:, 0] * c) + (alpha[:, 1] * h)

        return output