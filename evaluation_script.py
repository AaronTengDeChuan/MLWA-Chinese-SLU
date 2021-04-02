# -*- coding: utf-8 -*-

"""
@CreateTime :       2020/6/14 13:20
@Author     :       dcteng
@File       :       evaluation_script.py
@Software   :       PyCharm
@Framework  :       Pytorch
@LastModify :       2020/6/14 13:20
"""

import json
import codecs
import sys
import numpy as np
from collections import OrderedDict

from src.miulab import computeF1Score
from seqeval.metrics import precision_score, recall_score, f1_score

'''
    Calculate the slu metric
        txt file format: 
            请   O   O
            帮   O   O
            我   O   O
            打   O   O
            开   O   O
            u B-name    B-category
            c E-name    E-category
            请 帮 我 打开 uc
            LAUNCH  LAUNCH
            
            打   O   B
            开   O   E
            汽   B-name  B-name
            车   I-name  I-name
            之   I-name  I-name
            家   E-name  E-name
            打开 汽车 之 家
            LAUNCH  LAUNCH
        
        output format:
            {
                "Slot Precision" : 0.5000,
                "Slot Recall" : 0.5000,
                "Slot F1" : 0.5000,
                "Intent Accurary" : 0.5000,
                "Semantic Accurary" : 0.5000
            }
'''

def get_intent_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        "Intent Accurary": acc
    }

def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    return OrderedDict({
        "Slot Precision": precision_score(labels, preds),
        "Slot Recall": recall_score(labels, preds),
        "Slot F1": f1_score(labels, preds)
    })

def get_miulab_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    f1, precision, recall = computeF1Score(labels, preds)
    return OrderedDict({
        "MiuLab Slot Precision": precision,
        "MiuLab Slot Recall": recall,
        "MiuLab Slot F1": f1
    })

def get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels):
    """For the cases that intent and all the slots are correct (in one sentence)"""
    # Get the intent comparison result
    intent_result = (intent_preds == intent_labels)

    # Get the slot comparision result
    slot_result = []
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)
    slot_result = np.array(slot_result)

    sementic_acc = np.multiply(intent_result, slot_result).mean()
    return {
        "Semantic Accurary": sementic_acc
    }

def calculate_slu_metrics(intent_preds, intent_labels, slot_preds, slot_labels):
    intent_preds = np.array(intent_preds)
    intent_labels = np.array(intent_labels)
    assert len(intent_preds) == len(intent_labels) == len(slot_preds) == len(slot_labels)
    results = OrderedDict()
    intent_result = get_intent_acc(intent_preds, intent_labels)
    slot_result = get_slot_metrics(slot_preds, slot_labels)
    # compute p r f1 using miulab
    miulab_slot_result = get_miulab_slot_metrics(slot_preds, slot_labels)
    sementic_result = get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels)

    results.update(miulab_slot_result)
    results.update(slot_result)
    results.update(intent_result)
    results.update(sementic_result)

    return results

def read_result_file(in_file):
    with codecs.open(in_file, 'r', encoding="utf-8") as fi:
        chunks = [
            [line.split('\t') for line in chunk.strip().split('\n')]
            for chunk in fi.read().strip().split("\n\n")
        ]

    normalized_chunks = []
    for chunk in chunks:
        if len(chunk) == 1:
            normalized_chunks[-1].extend([["incorrect sentence"], chunk[0]])
        else:
            normalized_chunks.append(chunk)

    intent_preds, intent_labels, slot_preds, slot_labels = [], [], [], []
    char_texts, sentences = [], []
    for chunk in normalized_chunks:
        char_text, slot_label, slot_pred = list(zip(*chunk[:-2]))[:3]
        char_texts.append(list(char_text))
        slot_labels.append(list(slot_label))
        slot_preds.append(list(slot_pred))
        sentences.append(chunk[-2][0])
        intent_labels.append(chunk[-1][0])
        intent_preds.append(chunk[-1][1])

    return char_texts, sentences, slot_labels, slot_preds, intent_labels, intent_preds

if __name__ == '__main__':
    assert len(sys.argv) == 2, "Usage: python evaluation_script.py in_file"
    char_texts, sentences, slot_labels, slot_preds, intent_labels, intent_preds = read_result_file(sys.argv[1])
    result = calculate_slu_metrics(intent_preds, intent_labels, slot_preds, slot_labels)
    result["Chunk Num"] = len(char_texts)
    print (json.dumps(result, indent=4))