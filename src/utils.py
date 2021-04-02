# -*- coding: utf-8 -*-

"""
@CreateTime :       2020/6/12 10:37
@Author     :       dcteng
@File       :       utils.py
@Software   :       PyCharm
@Framework  :       Pytorch
@LastModify :       2020/6/12 10:37
"""

import os
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score

def computeF1Score(correct_slots, pred_slots):
    assert len(pred_slots) == len(correct_slots)

    f1 = precision_score(correct_slots, pred_slots)
    recall = recall_score(correct_slots, pred_slots)
    precision =  f1_score(correct_slots, pred_slots)

    return f1, precision, recall


def restore_order(sorted_list, sorted_index):
    second_sorted_index = np.argsort(sorted_index)
    restored_list = []
    for index in second_sorted_index:
        restored_list.append(sorted_list[index])

    return restored_list

def save_slu_error_result(pred_slot, real_slot, pred_intent, real_intent, sent_info, save_dir, mode):
    # To make sure the directory for save error prediction.
    mistake_dir = os.path.join(save_dir, "error", mode)
    if not os.path.exists(mistake_dir):
        os.makedirs(mistake_dir, exist_ok=True)

    slot_file_path = os.path.join(mistake_dir, "slot.txt")
    intent_file_path = os.path.join(mistake_dir, "intent.txt")
    both_file_path = os.path.join(mistake_dir, "both.txt")

    char_lists = sent_info[0]
    word_lists = sent_info[1]

    # Write those sample with mistaken slot prediction.
    with open(slot_file_path, 'w') as fw:
        for c_list, w_list, r_slot_list, p_slot_list in zip(char_lists, word_lists, real_slot, pred_slot):
            if r_slot_list != p_slot_list:
                for c, r, p in zip(c_list, r_slot_list, p_slot_list):
                    fw.write(c + '\t' + r + '\t' + p + '\n')
                fw.write(" ".join(w_list[:len(w_list) - w_list.count("<PAD>")]) + '\n')
                fw.write('\n')

    # Write those sample with mistaken intent prediction.
    with open(intent_file_path, 'w') as fw:
        for c_list, w_list, r_intent, p_intent in zip(char_lists, word_lists, real_intent, pred_intent):
            if p_intent != r_intent:
                for c in c_list:
                    fw.write(c + '\n')
                fw.write(" ".join(w_list[:len(w_list) - w_list.count("<PAD>")]) + '\n')
                fw.write(r_intent + '\t' + p_intent + '\n\n')

    # Write those sample both have intent and slot errors.
    with open(both_file_path, 'w') as fw:
        for c_list, w_list, r_slot_list, p_slot_list, r_intent, p_intent in \
                zip(char_lists, word_lists, real_slot, pred_slot, real_intent, pred_intent):

            if r_slot_list != p_slot_list or r_intent != p_intent:
                for c, r_slot, p_slot in zip(c_list, r_slot_list, p_slot_list):
                    fw.write(c + '\t' + r_slot + '\t' + p_slot + '\n')
                fw.write(" ".join(w_list[:len(w_list) - w_list.count("<PAD>")]) + '\n')
                fw.write(r_intent + '\t' + p_intent + '\n\n')


def save_slu_result(pred_slot, real_slot, pred_intent, real_intent, sent_info, save_dir, mode):
    # To make sure the directory for save entire prediction.
    result_dir = os.path.join(save_dir, "result", mode)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    entire_result_file_path = os.path.join(result_dir, "entire.txt")

    char_lists = sent_info[0]
    word_lists = sent_info[1]

    # Write entire results.
    with open(entire_result_file_path, 'w') as fw:
        for c_list, w_list, r_slot_list, p_slot_list, r_intent, p_intent in \
                zip(char_lists, word_lists, real_slot, pred_slot, real_intent, pred_intent):
            for c, r_slot, p_slot in zip(c_list, r_slot_list, p_slot_list):
                fw.write(c + '\t' + r_slot + '\t' + p_slot + '\n')
            fw.write(" ".join(w_list[:len(w_list) - w_list.count("<PAD>")]) + '\n')
            fw.write(r_intent + '\t' + p_intent + '\n\n')