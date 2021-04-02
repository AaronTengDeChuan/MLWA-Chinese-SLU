# -*- coding: utf-8 -*-

"""
@CreateTime :       2020/3/11 10:13
@Author     :       dcteng
@File       :       operation.py
@Software   :       PyCharm
@Framework  :       Pytorch
@LastModify :       2020/3/11 10:13
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence

from functools import reduce


# TODO: Related to RNN

def pack_and_pad_sequences_for_rnn(seq_embeds, seq_lens, rnn_module, hidden=None):
    '''
    similar to dynamic rnn
    supported rnn including GRU, LSTM
    batch_first of rnn_module must be True
    :param seq_embeds:  [batch, ..., seq_len, input_size]
    :param seq_lens:    [batch, ...]
    :param rnn_module: rnn module in Pytorch
    :return:
        rnn_output: [batch, ..., seq_len, num_directions * hidden_size],
        rnn_ht:     [num_layers * num_directions, batch, ..., hidden_size]
    '''
    sorted_seq_lens, seq_indices = torch.sort(seq_lens.view(-1), dim=0, descending=True)
    sorted_seq_embeds = torch.index_select(seq_embeds.view(-1, seq_embeds.shape[-2], seq_embeds.shape[-1]), dim=0,
                                           index=seq_indices)
    # varname(sorted_seq_embeds) # torch.Size([None, 50, 200])
    sorted_seq_lens = sorted_seq_lens + torch.tensor(sorted_seq_lens == 0, device=sorted_seq_lens.device,
                                                     dtype=torch.int64)
    # Packs a Tensor containing padded sequences of variable length in order to obtain a PackedSequence object
    packed_seq_input = pack_padded_sequence(sorted_seq_embeds, sorted_seq_lens, batch_first=True)
    # varname(packed_seq_input) # torch.Size([478, 200]), torch.Size([50])
    packed_seq_output, seq_ht = rnn_module(packed_seq_input, hidden)
    # varname(packed_seq_output) # torch.Size([478, 200]), torch.Size([50])
    # varname(seq_ht) # torch.Size([1, None, 200])
    # Pads a packed batch of variable length sequences
    # Ensure that the shape of output is not changed: check this
    seq_output, _ = pad_packed_sequence(packed_seq_output, batch_first=True, total_length=seq_embeds.shape[-2])
    # varname(seq_output) # torch.Size([None, 50, 200])
    # restore original order
    _, original_indices = torch.sort(seq_indices, dim=0, descending=False)
    seq_output = torch.index_select(seq_output, dim=0, index=original_indices)
    # varname(seq_output) # torch.Size([None, 50, 200])
    # restore original shape
    seq_output = seq_output.view(seq_embeds.shape[:-2] + seq_output.shape[-2:])
    # varname(seq_output)
    assert seq_output.shape[-2] == seq_embeds.shape[-2]
    # seq_ht: [num_layers * num_directions, batch, hidden_size]
    if isinstance(seq_ht, torch.Tensor):
        seq_ht = torch.index_select(seq_ht.transpose(0, 1), dim=0, index=original_indices).transpose(0, 1)
        seq_ht = seq_ht.view(seq_ht.shape[:-2] + seq_embeds.shape[:-2] + seq_ht.shape[-1:])
    else:
        tmp = []
        for ht in seq_ht:
            ht = torch.index_select(ht.transpose(0, 1), dim=0, index=original_indices).transpose(0, 1)
            tmp.append(ht.view(ht.shape[:-2] + seq_embeds.shape[:-2] + ht.shape[-1:]))
        seq_ht = tuple(tmp)
    # varname(seq_ht)
    return seq_output, seq_ht


# TODO: Related to Mask
def sequence_mask(lengths, max_length, mask_value=0):
    '''
    Returns a mask tensor representing the first N positions of each cell
    :param lengths:     a tensor with shape [d1,...]
    :param max_length:  an integer
    :mask_value:        0 or 1
    :return:            a tensor with shape [d1,..., max_length] and dtype torch.uint8
    '''
    device = lengths.device
    # left = torch.ones(lengths.shape[0], max_length, dtype=torch.int64, device=device) * torch.arange(1, max_length + 1,
    #                                                                                                  device=device)
    # right = lengths.unsqueeze(dim=-1).expand(-1, max_length)
    # n-dimension version
    left = torch.ones(*lengths.shape, max_length, dtype=torch.int64, device=device) * torch.arange(1, max_length + 1,
                                                                                                   device=device)
    right = lengths.unsqueeze(dim=-1).expand(*lengths.shape, max_length)
    return left <= right if mask_value == 0 else left > right

def mask(row_lengths, col_lengths, max_row_length, max_col_length):
    '''
        Return a mask tensor representing the first N positions of each row and each column.
            Args:
                row_lengths: a tensor with shape [batch]
                col_lengths: a tensor with shape [batch]
            Returns:
                a mask tensor with shape [batch, max_row_length, max_col_length]
            Raises:
    '''
    row_mask = sequence_mask(row_lengths, max_row_length) #bool, [batch, max_row_len]
    col_mask = sequence_mask(col_lengths, max_col_length) #bool, [batch, max_col_len]
    dtype = torch.get_default_dtype()
    row_mask = row_mask.unsqueeze(dim=-1).to(dtype=dtype)
    col_mask = col_mask.unsqueeze(dim=-1).to(dtype=dtype)
    # TODO: check this
    return torch.einsum('bik,bjk->bij', (row_mask, col_mask))

def generate_mask(seq_lens, device):
    dtype = torch.get_default_dtype()
    row_mask = sequence_mask(torch.tensor(seq_lens, device=device), max(seq_lens), mask_value=0).to(dtype=dtype)
    matrix_mask = mask(torch.tensor(seq_lens, device=device), torch.tensor(seq_lens, device=device),
                       max(seq_lens), max(seq_lens))
    return row_mask, matrix_mask


# TODO: Related to Padding
def pad_tensor_along_batch(input_tensor, batch_lens):
    """
    pad input tensor along the first dim (batch dim)
    :param input_tensor: tensor with shape [sum_seq_len, *]
    :param batch_lens: list of sequence lens
    :return:
    """
    device = input_tensor.device
    dtype = torch.get_default_dtype()
    assert input_tensor.shape[0] == sum(batch_lens)
    padded_input = pad_sequence(torch.split(input_tensor, batch_lens, dim=0), batch_first=True)
    mask = sequence_mask(torch.tensor(batch_lens, device=device), max(batch_lens), mask_value=0).to(dtype=dtype)

    return padded_input, mask

# TODO: Related to Alignment
def char_word_alignment(word_hiddens, char_seq_lens, word_seq_lens, align_info):
    # max_word_len = max([max(align) for align in align_info])
    # expanded_word_hiddens = word_hiddens.unsqueeze(1).expand(-1, max_word_len, -1)
    # expanded_align = reduce(lambda x, y: x + y, align_info, [])
    # aligned_word_hiddens = torch.cat([expanded_word_hiddens[i][:num] for i, num in enumerate(expanded_align)])
    # assert aligned_word_hiddens.size(0) == sum(char_seq_lens)
    # return aligned_word_hiddens

    aligned_word_hiddens = []
    word_idx = 0
    # examine the correctness of alignment information
    for cl, wl, align in zip(char_seq_lens, word_seq_lens, align_info):
        assert cl == sum(align) and wl == len(align)
        for num in align:
            aligned_word_hiddens.extend([word_hiddens[word_idx: word_idx + 1]] * num)
            word_idx += 1

    assert len(aligned_word_hiddens) == sum(char_seq_lens)

    return torch.cat(aligned_word_hiddens, dim=0)


def batch_char_word_alignment(word_hiddens, char_seq_lens, word_seq_lens, align_info, sent_seg_lens=None):
    if sent_seg_lens is None:
        sent_seg_lens = [1] * len(char_seq_lens)
    assert len(sent_seg_lens) == len(char_seq_lens)
    assert word_hiddens.size(0) == len(word_seq_lens) == len(align_info)

    max_sent_num = max(sent_seg_lens)
    batch_start_pos = 0
    aligned_char_hiddens = []
    char_sent_num = []

    for sl, cl in zip(sent_seg_lens, char_seq_lens):
        char_hiddens_list = []
        batch_end_pos = batch_start_pos + sl
        for i, (wl, align) in enumerate(
            zip(word_seq_lens[batch_start_pos: batch_end_pos], align_info[batch_start_pos: batch_end_pos])):
            assert cl == sum(align) and wl == len(align)
            char_hidden_list = []
            for word_idx, num in enumerate(align):
                # list of [1, emb_dim]
                char_hidden_list.extend([word_hiddens[batch_start_pos + i][word_idx: word_idx + 1]] * num)
            # list of [char_seq_len, 1, emb_dim]
            char_hiddens_list.append(torch.cat(char_hidden_list, dim=0).unsqueeze(1))
        # [char_seq_len, sent_seg_len, emb_dim]
        char_hiddens = torch.cat(char_hiddens_list, dim=1)
        # list of [char_seq_len, max_sent_num, emb_dim]
        aligned_char_hiddens.append(F.pad(char_hiddens, (0, 0, 0, max_sent_num - sl), mode="constant", value=0))
        char_sent_num.extend([sl] * cl)
        batch_start_pos = batch_end_pos

    device = word_hiddens.device
    dtype = torch.get_default_dtype()
    # [num_char, max_sent_num]
    char_sent_mask = sequence_mask(torch.tensor(char_sent_num, device=device), max_sent_num, mask_value=0).to(dtype=dtype)
    # [num_char, max_sent_num, emb_dim]
    return torch.cat(aligned_char_hiddens, dim=0), char_sent_mask


def construct_sent_segmentation(nodes, edges, sent_segs, max_sent_seg_num=None):
    device = nodes.device
    batch_size = len(sent_segs)

    sent_seg_lens = []
    word_align_info = []
    word_seq_lens = []
    word_hidden_list = []
    for i in range(batch_size):
        sent_seg_len = len(sent_segs[i][:max_sent_seg_num]) if max_sent_seg_num else len(sent_segs[i])
        sent_seg_lens.append(sent_seg_len)
        for j in range(sent_seg_len):
            word_align_info.append(sent_segs[i][j][3])
            # [num_sent]
            word_seq_lens.append(len(sent_segs[i][j][3]))
            word_hidden = None
            if sent_segs[i][j][1]:
                word_hidden = torch.index_select(nodes[i], dim=0, index=torch.tensor(sent_segs[i][j][1], device=device))
            if sent_segs[i][j][2]:
                temp_hidden = torch.index_select(edges[i], dim=0, index=torch.tensor(sent_segs[i][j][2], device=device))
                if word_hidden is not None:
                    word_hidden = torch.cat([word_hidden, temp_hidden], dim=0)
                else:
                    word_hidden = temp_hidden
            # [word_seq_len, emb_dim]
            word_hidden = torch.index_select(word_hidden, dim=0, index=torch.tensor(sent_segs[i][j][4], device=device))
            word_hidden_list.append(word_hidden)

    # [num_word, emb_dim]
    word_hiddens = torch.cat(word_hidden_list, dim=0)
    # [num_sent, max_word_seq_len, emb_dim], [num_sent, max_word_seq_len]
    padded_word_hiddens, _ = pad_tensor_along_batch(word_hiddens, word_seq_lens)
    word_rmask, word_mmask = generate_mask(word_seq_lens, device)

    return padded_word_hiddens, word_rmask, word_mmask, word_seq_lens, word_align_info, sent_seg_lens


# TODO: Related to Lexicon

def construct_word4char(word_list, only_one_none=True):
    word_ids = []
    char_idxs4word = []
    char_bmes = []
    word_lens = []

    batch_word_idxs4char = []
    batch_bmes4char = []
    batch_word_num4char = []
    num_pre_char = 0
    for batch_idx in range(len(word_list)):
        char_seq_len = len(word_list[batch_idx])
        # B M E S for each char
        word_idxs4char = [[[], [], [], []] for _ in range(char_seq_len)]
        for char_idx in range(char_seq_len):
            wl = word_list[batch_idx][char_idx]
            if not wl:
                continue
            for word, word_len in zip(wl[0], wl[1]):
                # add new word
                if word not in word_ids:
                    word_ids.append(word)
                    char_idxs4word.extend(range(num_pre_char + char_idx, num_pre_char + char_idx + word_len))
                    if word_len == 1:
                        char_bmes.append(3)
                    else:
                        char_bmes.extend([0] + [1] * (word_len - 2) + [2])
                    word_lens.append(word_len)

                word_idx = word_ids.index(word) + 1
                if word_len == 1:
                    # S
                    word_idxs4char[char_idx][3].append(word_idx)
                else:
                    # B
                    word_idxs4char[char_idx][0].append(word_idx)
                    # M
                    for offset in range(1, word_len - 1):
                        word_idxs4char[char_idx + offset][1].append(word_idx)
                    # E
                    word_idxs4char[char_idx + word_len - 1][2].append(word_idx)

        for temp in word_idxs4char:
            word_idx = []
            bmes4char = []
            for flag, item in enumerate(temp):
                if len(item) == 0 and not only_one_none:
                    word_idx.append(0)
                    bmes4char.append(flag)
                    continue
                word_idx.extend(item)
                bmes4char.extend([flag] * len(item))

            if only_one_none and len(word_idx) == 0:
                word_idx.append(0)
                bmes4char.append(4)

            batch_word_idxs4char.extend(word_idx)
            batch_bmes4char.extend(bmes4char)
            batch_word_num4char.append(len(bmes4char))

        num_pre_char += char_seq_len

    return word_ids, char_idxs4word, char_bmes, word_lens, batch_word_idxs4char, batch_bmes4char, batch_word_num4char


if __name__ == '__main__':
    word_list = [[ [[1,2,3], [2,1,3]], [] , [[4], [1]] ], [ [[6], [3]], [], [] ]]
    print (construct_word4char(word_list))