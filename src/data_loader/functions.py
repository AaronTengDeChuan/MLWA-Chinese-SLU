# -*- coding: utf-8 -*-

"""
@CreateTime :       2020/3/27 8:32
@Author     :       dcteng
@File       :       functions.py
@Software   :       PyCharm
@Framework  :       Pytorch
@LastModify :       2020/3/27 8:32
"""

import numpy as np
from tqdm import tqdm


MODEL_PATH_MAP = {
    'chinese_bert': "bert-base-chinese",
    'bert': 'bert-base-uncased',
}

def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def build_pretrain_embedding(embedding_path, word_alphabet, norm=True, pre_embedding=None, embedd_dim=50):

    def norm2one(vec):
        root_sum_square = np.sqrt(np.sum(np.square(vec)))
        return vec / root_sum_square

    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)

    scale = np.sqrt(3.0 / embedd_dim)
    if pre_embedding is not None:
        assert pre_embedding.shape[0] == len(word_alphabet) and pre_embedding.shape[1] == embedd_dim
        pretrain_emb = pre_embedding
    else:
        pretrain_emb = np.empty([len(word_alphabet), embedd_dim])

    not_match = 0
    for word, index in word_alphabet.instance2index.items():
        if word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embedd_dict[word.lower()]
        elif word in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
        else:
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, match:%s, oov:%s, oov%%:%.4f" %
          (pretrained_size, len(word_alphabet) - not_match, not_match, (not_match+0.)/len(word_alphabet)))

    return pretrain_emb, embedd_dim


def load_pretrain_emb(embedding_path):
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            embedd_dim = len(tokens) - 1
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
    return embedd_dict, embedd_dim


def find_sent_seg_combination(word_list, chars, index, order, full_sent_seg):
    if index == len(word_list):
        return [[[], [], [], [], [], [], 0]]

    segs = []
    num_pre_word = sum([len(words) for words in word_list[:index]])
    if full_sent_seg:
        words = [chars[index]] + word_list[index]
    else:
        words = word_list[index] if word_list[index] else [chars[index]]
    for i, word in enumerate(words):
        seg_word = [word]
        seg_char_index = [index] if len(word) == 1 else []
        seg_word_index = [] if len(word) == 1 else [num_pre_word + i + (0 if full_sent_seg else 1)]
        seg_align_info = [len(word)]
        seg_char_order = [order] if len(word) == 1 else []
        seg_word_order = [] if len(word) == 1 else [order]
        seg_score = 1 if len(word) == 1 else 0

        suffix_segs = find_sent_seg_combination(word_list, chars, index + len(word), order + 1, full_sent_seg)
        for suffix_seg in suffix_segs:
            segs.append([seg_word + suffix_seg[0], seg_char_index + suffix_seg[1],
                         seg_word_index + suffix_seg[2], seg_align_info + suffix_seg[3],
                         seg_char_order + suffix_seg[4], seg_word_order + suffix_seg[5],
                         seg_score + suffix_seg[6]])

    return segs

# def demo_recursive(word_list, chars, index, order):
#     if index == len(word_list):
#         return
#     lens = [1] + [len(word) for word in word_list[index]]
#     for i, word in lens:
#
# def demo(word_list, chars):
#     if index

def get_seg_info(word_text, word_list):
    char_idx = 0
    char_index, word_index, align_info, char_order, word_order = [], [], [], [], []
    for i, word in enumerate(word_text):
        if len(word) == 1:
            char_index.append(char_idx)
            char_order.append(i)
        else:
            num_pre_word = sum([len(words) for words in word_list[:char_idx]])
            word_index.append(num_pre_word + word_list[char_idx].index(word) + 1)
            word_order.append(i)
        align_info.append((len(word)))
        char_idx += len(word)
    return [word_text, char_index, word_index, align_info, np.argsort(char_order + word_order).tolist(), len(char_index)]


def build_sentence_segmentation(word_data, word_list_data, no_progressbar, full_sent_seg):
    max_sent_num = 10
    seg_sents = []
    bar = tqdm(total=len(word_data), desc="Build sentence segmentation")
    for word_text, word_list in zip(word_data, word_list_data):
        chars = "".join(word_text)
        seg_sent = find_sent_seg_combination(word_list, chars, 0, 0, full_sent_seg)

        new_seg_sent = [get_seg_info(word_text, word_list)]

        for seg in sorted(seg_sent, key=lambda x: x[6]):
            assert "".join(seg[0]) == chars
            if max_sent_num <= len(new_seg_sent): break
            if seg[0] == word_text or seg[6] == len(chars):
                continue
            new_seg_sent.append([seg[0], seg[1], seg[2], seg[3], np.argsort(seg[4] + seg[5]).tolist(), seg[6]])

        seg_sents.append(new_seg_sent)

        if not no_progressbar:
            bar.update(1)

    bar.close()
    return seg_sents


def chinese_tokenization_check(ori_tokens, res_tokens):
    check_flag = len(ori_tokens) == len(res_tokens)
    for ori_token, res_token in zip(ori_tokens, res_tokens):
        if not check_flag:
            break
        if ori_token != res_token and res_token != '[UNK]':
            check_flag = False

    if not check_flag:
        raise Exception("Not Match: {} vs {}".format(ori_tokens, res_tokens))
    return


def convert_examples_to_features(texts, max_seq_len, tokenizer,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    total_input_ids = []
    total_attention_mask = []
    total_token_type_ids = []

    for (ex_index, text) in enumerate(texts):
        if ex_index % 5000 == 0:
            print("Writing example %d of %d" % (ex_index, len(texts)))

        tokens = []
        for word in text:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)

        chinese_tokenization_check(text, tokens)

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[: (max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)

        total_input_ids.append(input_ids)
        total_attention_mask.append(attention_mask)
        total_token_type_ids.append(token_type_ids)

    return total_input_ids, total_attention_mask, total_token_type_ids


if __name__ == '__main__':
    char_emb_path = "D:\Researches\Codes\LGN-master\data\gigaword_chn.all.a2b.uni.ite50.vec"
    word_emb_path = "D:\Researches\Codes\LGN-master\data\ctb.50d.vec"

    char_emb, _ = load_pretrain_emb(char_emb_path)
    word_emb, _ = load_pretrain_emb(word_emb_path)

    print("{} chars in char emb file.".format(len(char_emb)))
    print("{} words in word emb file.".format(len(word_emb)))
    print("{} tokens in the intersection of char and word dict.".format(len(char_emb.keys() & word_emb.keys())))
    print("{} tokens in the difference set between char and word dict.".format(len(char_emb.keys() - word_emb.keys())))
    print("{} tokens in the difference set between word and char dict.".format(len(word_emb.keys() - char_emb.keys())))