# -*- coding: utf-8 -*-

"""
@CreateTime :       2020/3/10 10:39
@Author     :       dcteng
@File       :       train.py
@Software   :       PyCharm
@Framework  :       Pytorch
@LastModify :       2020/3/10 10:39
"""

from src.modules.module import ModelManager
from src.modules.gnn_module import GnnModelManager
from src.modules.deeper_module import DeeperModelManager
from src.data_loader.loader import DatasetManager
from src.process import Processor

import torch

import os
import json
import random
import argparse
import numpy as np
from copy import deepcopy


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()

# Training parameters.
# TODO:
parser.add_argument('--do_evaluation', '-eval', action="store_true", default=False)
parser.add_argument('--data_dir', '-dd', type=str, default='data/cais')
parser.add_argument('--train_file_name', '-train_file', type=str, default='train.txt')
parser.add_argument('--valid_file_name', '-valid_file', type=str, default='dev.txt')
parser.add_argument('--test_file_name', '-test_file', type=str, default='test.txt')
parser.add_argument('--save_dir', '-sd', type=str, default='save')
parser.add_argument("--random_state", '-rs', type=int, default=0)
parser.add_argument('--num_epoch', '-ne', type=int, default=100)
parser.add_argument('--batch_size', '-bs', type=int, default=16)
parser.add_argument('--l2_penalty', '-lp', type=float, default=1e-6)
parser.add_argument("--learning_rate", '-lr', type=float, default=0.001)
parser.add_argument("--max_grad_norm", "-mgn", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument('--dropout_rate', '-dr', type=float, default=0.3)
# parser.add_argument('--intent_forcing_rate', '-ifr', type=float, default=0.9)
parser.add_argument("--differentiable", "-d", action="store_true", default=False)
# parser.add_argument("--tokenization", "-t", action="store_true", default=False)
parser.add_argument('--slot_forcing_rate', '-sfr', type=float, default=0.9)
# parser.add_argument('--token_forcing_rate', '-tfr', type=float, default=0.9)
parser.add_argument("--no_progressbar", "-npb", action="store_true", default=False)

# Parameters related to Simple GNN
parser.add_argument('--use_simple_lexicon_gnn', '-uslg', action="store_true", default=False)
parser.add_argument('--allow_single_char', '-asc', action="store_true", default=False)
parser.add_argument('--use_char_bmes_emb', '-ucbe', type=str2bool, default=True)
parser.add_argument('--use_word_bmes_emb', '-uwbe', type=str2bool, default=True)
parser.add_argument('--use_c2w_encoder_qkv_input_linear', '-uc2weil', type=str2bool, default=True)
parser.add_argument('--use_c2w_qkv_input_linear', '-uc2wil', type=str2bool, default=True)
parser.add_argument('--bilinear_attention', '-blatt', action="store_true", default=False)

# Parameters related to GNN
parser.add_argument('--use_lexicon_gnn', '-ulg', action="store_true", default=False)
parser.add_argument('--use_pretrained_emb', '-upe', action="store_true", default=False)
parser.add_argument('--char_emb_path', '-cep', type=str, default='data/gigaword_chn.all.a2b.uni.ite50.vec')
parser.add_argument('--word_emb_path', '-wep', type=str, default='data/ctb.50d.vec')
parser.add_argument('--gnn_use_edge', "-gue", type=str2bool, default=True, help='If use lexicon embeddings (edge embeddings).')
parser.add_argument('--gnn_use_global', "-gug", type=str2bool, default=True, help='If use the global node.')
parser.add_argument('--gnn_bidirectional', "-gbi", type=str2bool, default=True, help='If use bidirectional digraph.')
parser.add_argument('--gnn_iters', "-giters", type=int, default=4, help='The number of Graph iterations.')
parser.add_argument('--gnn_hidden_dim', "-ghd", type=int, default=50, help='Hidden state size.')
parser.add_argument('--gnn_num_head', type=int, default=10, help='Number of transformer head.')
parser.add_argument('--gnn_head_dim', type=int, default=20, help='Head dimension of transformer.')
parser.add_argument('--tf_drop_rate', "-tfdr", type=float, default=0.1, help='Transformer dropout rate.')
parser.add_argument('--cell_drop_rate', "-cdr", type=float, default=0.2, help='Aggregation module dropout rate.')
parser.add_argument('--full_sent_seg', "-fss", action="store_true", default=False, help='Find all sentence segmentation combination.')
parser.add_argument('--sent_seg_num', "-ssn", type=int, default=1, help='The number of sentence segmentations .')
# parser.add_argument('--intent_encoder_hidden_dim', '-iehd', type=int, default=64)
parser.add_argument('--intent_c2w_attention', '-ic2watt', action="store_true", default=False)
parser.add_argument('--intent_c2w_attention_hidden_dim', '-ic2wahd', type=int, default=256)
parser.add_argument('--slot_c2w_attention', '-sc2watt', action="store_true", default=False)
parser.add_argument('--slot_c2w_attention_hidden_dim', '-sc2wahd', type=int, default=256)

# model parameters.
# TODO:
parser.add_argument('--use_bert_input', '-use_bert', action="store_true", default=False)
parser.add_argument('--percent_of_encoder_hidden_dim', '-pehd', type=float, default=0.5)
parser.add_argument('--unique_vocabulary', '-u', action="store_true", default=False)
parser.add_argument('--golden_intent', '-gi', action="store_true", default=False)
parser.add_argument('--single_channel_intent', '-sci', action="store_true", default=False)
parser.add_argument('--single_channel_slot', '-scs', action="store_true", default=False)
parser.add_argument('--no_multi_level', '-nml', action="store_true", default=False)
parser.add_argument('--char_embedding_dim', '-ced', type=int, default=64)
parser.add_argument('--word_embedding_dim', '-wed', type=int, default=64)
parser.add_argument('--encoder_hidden_dim', '-ehd', type=int, default=256)
parser.add_argument('--char_attention_hidden_dim', '-cahd', type=int, default=1024)
parser.add_argument('--word_attention_hidden_dim', '-wahd', type=int, default=1024)
parser.add_argument('--attention_output_dim', '-aod', type=int, default=128)
# parser.add_argument('--token_embedding_dim', '-ted', type=int, default=8)
# parser.add_argument('--intent_embedding_dim', '-ied', type=int, default=8)
parser.add_argument('--intent_fusion_type', '-ift', type=str, default='rate_bilinear')
parser.add_argument('--slot_fusion_type', '-sft', type=str, default='rate_bilinear')
parser.add_argument('--slot_embedding_dim', '-sed', type=int, default=32)
# parser.add_argument('--token_decoder_hidden_dim', '-tdhd', type=int, default=64)
parser.add_argument('--slot_decoder_hidden_dim', '-sdhd', type=int, default=64)
# parser.add_argument('--intent_decoder_hidden_dim', '-idhd', type=int, default=64)
parser.add_argument('--undirectional_word_level_slot_encoder', '-udwse', action="store_true", default=False)

if __name__ == "__main__":
    args = parser.parse_args()

    if not args.do_evaluation:
        # Save training and model parameters.
        if not os.path.exists(args.save_dir):
            os.system("mkdir -p " + args.save_dir)

        log_path = os.path.join(args.save_dir, "param.json")
        with open(log_path, "w") as fw:
            fw.write(json.dumps(args.__dict__, indent=True))

        # Fix the random seed of package random.
        random.seed(args.random_state)
        np.random.seed(args.random_state)

        # Fix the random seed of Pytorch when using GPU.
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.random_state)
            torch.cuda.manual_seed(args.random_state)

        # Fix the random seed of Pytorch when using CPU.
        torch.manual_seed(args.random_state)
        torch.random.manual_seed(args.random_state)

        # Instantiate a dataset object.
        dataset = DatasetManager(args)
        dataset.quick_build()
        dataset.show_summary()

        model_fn = ModelManager
        if args.use_lexicon_gnn: model_fn = GnnModelManager
        if args.use_simple_lexicon_gnn: model_fn = DeeperModelManager

        # Instantiate a network model object.
        model = model_fn(
            args, len(dataset.char_alphabet),
            len(dataset.word_alphabet),
            len(dataset.slot_alphabet),
            len(dataset.intent_alphabet),
            char_emb=dataset.char_embedding,
            word_emb=dataset.word_embedding
        )
        model.show_summary()

        # To train and evaluate the models.
        process = Processor(dataset, model, args.batch_size)
        try:
            process.train()
        except KeyboardInterrupt:
            print ("Exiting from training early.")

    model = torch.load(os.path.join(args.save_dir, "model/model.pkl"))
    dataset = torch.load(os.path.join(args.save_dir, "model/dataset.pkl"))

    print('\nAccepted performance: ' + str(Processor.validate(
        model, dataset, args.batch_size * 2)) + " at test dataset;\n")
