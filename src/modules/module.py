# -*- coding: utf-8 -*-

"""
@CreateTime :       2020/3/10 13:52
@Author     :       dcteng
@File       :       module.py
@Software   :       PyCharm
@Framework  :       Pytorch
@LastModify :       2020/3/10 13:52
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel, BertModel
from transformers import BertConfig

from src.components import operation as op
from src.components.layers import FusionLayer, EmbeddingCollection, LSTMEncoder, LSTMDecoder, SelfAttention, MLPAttention

from src.data_loader.functions import MODEL_PATH_MAP

MASK_VALUE = -2 ** 32 + 1

class ModelManager(nn.Module):

    def __init__(self, args, num_char, num_word, num_slot, num_intent, char_emb=None, word_emb=None):
        super(ModelManager, self).__init__()

        # hyper-parameters
        self.__num_char = num_char
        self.__num_word = num_word
        self.__num_slot = num_slot
        self.__num_intent = num_intent
        self.__args = args

        if not self.__args.unique_vocabulary and not self.__args.use_bert_input:
            # Initialize an char embedding object.
            self.__char_embedding = EmbeddingCollection(self.__num_char, self.__args.char_embedding_dim,
                                                        pretrain_embedding=char_emb)
        elif not self.__args.use_bert_input:
            assert self.__args.char_embedding_dim == self.__args.word_embedding_dim
        else:
            self.__bert_config = BertConfig.from_pretrained(MODEL_PATH_MAP["chinese_bert"], finetuning_task="chinese_slu")
            bert_hidden_dim = self.__bert_config.hidden_size
            encoder_hidden_dim = int(bert_hidden_dim * self.__args.percent_of_encoder_hidden_dim)
            self.__args.encoder_hidden_dim = encoder_hidden_dim + encoder_hidden_dim % 2
            self.__args.attention_output_dim = bert_hidden_dim - self.__args.encoder_hidden_dim

        # Initialize an word embedding object.
        self.__word_embedding = EmbeddingCollection(self.__num_word, self.__args.word_embedding_dim,
                                                    pretrain_embedding=word_emb)

        if self.__args.use_bert_input:
            self.__bert = BertModel.from_pretrained("bert-base-chinese", config=self.__bert_config)
        else:
            # TODO: Now, output dim of char encoder must be the same with that of word encoder
            # Initialize an LSTM Encoder object for char level
            self.__char_encoder = LSTMEncoder(self.__args.char_embedding_dim, self.__args.encoder_hidden_dim,
                                         self.__args.dropout_rate)
            # Initialize an self-attention layer for char level
            self.__char_attention = SelfAttention(self.__args.char_embedding_dim, self.__args.char_attention_hidden_dim,
                                                  self.__args.attention_output_dim, self.__args.dropout_rate)

        # Initialize an LSTM Encoder object for word level
        self.__word_encoder = LSTMEncoder(self.__args.word_embedding_dim, self.__args.encoder_hidden_dim,
                                          self.__args.dropout_rate)
        # Initialize an self-attention layer for word level
        self.__word_attention = SelfAttention(self.__args.word_embedding_dim, self.__args.word_attention_hidden_dim,
                                              self.__args.attention_output_dim, self.__args.dropout_rate)

        self.__encoder_output_dim = self.__args.encoder_hidden_dim + self.__args.attention_output_dim

        # dropout layer
        self.__dropout_layer = nn.Dropout(self.__args.dropout_rate)

        # MLP Attention
        # TODO: Insert a linear layer between encoder and MLP Attention Layer ?
        self.__char_sent_attention = MLPAttention(self.__encoder_output_dim, self.__args.dropout_rate)
        self.__word_sent_attention = MLPAttention(self.__encoder_output_dim, self.__args.dropout_rate)

        self.__intent_fusion_layer = FusionLayer(self.__encoder_output_dim, self.__encoder_output_dim,
                                                 self.__args.dropout_rate, self.__args.intent_fusion_type,
                                                 class_num=self.__num_intent)
        # One-hot encoding for augment data feed.
        self.__intent_embedding = nn.Embedding(self.__num_intent, self.__num_intent)
        self.__intent_embedding.weight.data = torch.eye(self.__num_intent)
        self.__intent_embedding.weight.requires_grad = False

        # TODO: Now, lstm output dim of char-level slot decoder must be the same with that of word-level slot decoder
        # Initialize an Decoder object for char-level slot.
        self.__char_slot_decoder = LSTMDecoder(
            self.__encoder_output_dim,
            self.__args.slot_decoder_hidden_dim,
            self.__num_slot, self.__args.dropout_rate, self.__args.slot_fusion_type,
            embedding_dim=self.__args.slot_embedding_dim,
            extra_input_dim=self.__num_intent,
            extra_hidden_dim=None if self.__args.single_channel_slot else self.__args.slot_decoder_hidden_dim
        )
        # # Initialize an Decoder object for char-level slot.
        # # TODO: word-level slot decoder has no forced input
        # self.__word_slot_decoder = LSTMDecoder(
        #     self.__encoder_output_dim,
        #     self.__args.slot_decoder_hidden_dim,
        #     self.__num_slot, self.__args.dropout_rate,
        #     embedding_dim=self.__args.slot_embedding_dim,
        #     extra_dim=self.__num_intent
        # )
        # Initialize an Encoder object for word-level slot.
        self.__word_slot_encoder = LSTMEncoder(self.__encoder_output_dim, self.__args.slot_decoder_hidden_dim,
                                               self.__args.dropout_rate,
                                               bidirectional=not self.__args.undirectional_word_level_slot_encoder,
                                               extra_dim=self.__num_intent)
        if self.__args.no_multi_level:
            self.__word_sent_attention4slot = MLPAttention(self.__args.slot_decoder_hidden_dim, self.__args.dropout_rate)

        # self.__slot_fusion_rate = nn.Parameter(torch.randn(1), requires_grad=True)
        # self.__slot_linear_layer = nn.Linear(self.__args.slot_decoder_hidden_dim, self.__num_slot)


    def show_summary(self):
        """
        print the abstract of the defined model.
        """
        print('Model parameters are listed as follows:\n')

        print('\tdropout rate:						                    {};'.format(self.__args.dropout_rate))
        print('\tdifferentiable:						                {};'.format(self.__args.differentiable))
        print('\tunique vocabulary:						                {};'.format(self.__args.unique_vocabulary))
        print('\tuse bert input:						                {};'.format(self.__args.use_bert_input))
        print('\tpercent of encoder hidden dim:                         {};'.format(self.__args.percent_of_encoder_hidden_dim))
        print('\tgolden intent:                                         {};'.format(self.__args.golden_intent))
        print('\tsingle channel intent:                                 {};'.format(self.__args.single_channel_intent))
        print('\tsingle channel slot:                                   {};'.format(self.__args.single_channel_slot))
        print('\tno multiple levels:                                    {};'.format(self.__args.no_multi_level))
        print('\tundirectional word-level slot encoder:                 {};'.format(self.__args.undirectional_word_level_slot_encoder))
        print('\tnumber of char:						                {};'.format(self.__num_char))
        print('\tnumber of word:                                        {};'.format(self.__num_word))
        print('\tnumber of slot:                                        {};'.format(self.__num_slot))
        print('\tnumber of intent:						                {};'.format(self.__num_intent))
        print('\tchar embedding dimension:				                {};'.format(self.__args.char_embedding_dim))
        print('\tword embedding dimension:				                {};'.format(self.__args.word_embedding_dim))
        print('\tencoder hidden dimension:				                {};'.format(self.__args.encoder_hidden_dim))
        print('\thidden dimension of char-level self-attention:         {};'.format(self.__args.char_attention_hidden_dim))
        print('\thidden dimension of word-level self-attention:         {};'.format(self.__args.word_attention_hidden_dim))
        print('\toutput dimension of self-attention:                    {};'.format(self.__args.attention_output_dim))
        print('\tintent fusion type:                                    {};'.format(self.__args.intent_fusion_type))
        print('\tslot fusion type:                                      {};'.format(self.__args.slot_fusion_type))
        print('\tdimension of slot embedding:			                {};'.format(self.__args.slot_embedding_dim))
        print('\tdimension of slot decoder hidden:  	                {};'.format(self.__args.slot_decoder_hidden_dim))

        print('\nEnd of parameters show. Now training begins.\n\n')

    def forward(self, char_text, char_seq_lens, word_text, word_seq_lens, align_info, word_list,
                n_predicts=None, forced_slot=None, golden_intent=None):
        """

        :param char_text: list of list of char ids
        :param char_seq_lens: list of the number of chars, e.g. [6, 7, 7]
        :param word_text: list of list of word ids
        :param word_seq_lens: list of the number of words, e.g. [4, 3, 4]
        :param align_info: list of list of the number of chars in each word, e.g. [ [1, 2, 1, 2], [2, 2, 3], [2, 1, 3, 1] ]
        :param n_predicts:
        :param forced_slot:
        :return:
        """

        # Get embeddings
        if self.__args.use_bert_input:
            input_ids, attention_mask, token_type_ids = char_text
        else:
            char_tensor = self.__char_embedding(char_text) if not self.__args.unique_vocabulary else \
                self.__word_embedding(char_text)
        word_tensor = self.__word_embedding(word_text)

        # Get mask
        device = word_tensor.device
        char_rmask, char_mmask = op.generate_mask(char_seq_lens, device)
        word_rmask, word_mmask = op.generate_mask(word_seq_lens, device)

        if self.__args.use_bert_input:
            max_char_sequence_len = max(char_seq_lens)
            # sequence_output, pooled_output, (hidden_states), (attentions)
            outputs = self.__bert(input_ids[:, :max_char_sequence_len + 2],
                                  attention_mask=attention_mask[:, :max_char_sequence_len + 2],
                                  token_type_ids=token_type_ids[:, :max_char_sequence_len + 2])
            sequence_output = outputs[0]
            char_sent_output = outputs[1]
            char_hiddens = sequence_output[:, 1: 1 + max_char_sequence_len, :]
        else:
            # TODO: take masking self-attention into account
            # Pass char encoder
            char_lstm_hiddens = self.__char_encoder(char_tensor, char_seq_lens)
            char_attention_hiddens = self.__char_attention(char_tensor, char_seq_lens, mmask=char_mmask)
            char_hiddens = torch.cat([char_attention_hiddens, char_lstm_hiddens], dim=-1)
            char_sent_output = self.__char_sent_attention(char_hiddens, rmask=char_rmask)

        # Pass word encoder
        word_lstm_hiddens = self.__word_encoder(word_tensor, word_seq_lens)
        word_attention_hiddens = self.__word_attention(word_tensor, word_seq_lens, mmask=word_mmask)
        word_hiddens = torch.cat([word_attention_hiddens, word_lstm_hiddens], dim=-1)

        # MLP Attention for Intent Detection
        word_sent_output = self.__word_sent_attention(word_hiddens, rmask=word_rmask)

        # Intent Prediction
        pred_intent = self.__intent_fusion_layer(char_sent_output,
                                                 y=None if self.__args.single_channel_intent else word_sent_output,
                                                 dropout=True)

        if not self.__args.differentiable:
            _, idx_intent = pred_intent.topk(1, dim=-1)
            if self.__args.golden_intent:
                assert golden_intent is not None
                feed_intent = self.__intent_embedding(golden_intent)
            else:
                feed_intent = self.__intent_embedding(idx_intent.squeeze(1))
        else:
            assert not self.__args.golden_intent
            feed_intent = pred_intent

        # TODO: double supervision signal ?
        # # pass word-level slot decoder
        # word_feed_intent = feed_intent.unsqueeze(1).expand(-1, word_hiddens.size(1), -1)
        # flat_word_feed_intent = torch.cat([word_feed_intent[i][:word_seq_lens[i], :]
        #                                    for i in range(0, len(word_seq_lens))], dim=0)
        # flat_word_hiddens = torch.cat([word_hiddens[i][:word_seq_lens[i], :]
        #                                for i in range(0, len(word_seq_lens))], dim=0)
        # _, word_slot_out = self.__word_slot_decoder(flat_word_hiddens, word_seq_lens, forced_input=None,
        #                                             extra_input=flat_word_feed_intent)

        # pass word-level slot encoder
        word_feed_intent = feed_intent.unsqueeze(1).expand(-1, word_hiddens.size(1), -1)
        word_slot_out = self.__word_slot_encoder(word_hiddens, word_seq_lens, extra_input=word_feed_intent)

        if self.__args.no_multi_level:
            word_slot_out = self.__word_sent_attention4slot(word_slot_out, rmask=word_rmask)
            word_slot_out = word_slot_out.unsqueeze(1).expand(-1, max(char_seq_lens), -1)
            aligned_word_slot_out = torch.cat([word_slot_out[i][:char_seq_lens[i], :]
                                               for i in range(0, len(char_seq_lens))], dim=0)
        else:
            flat_word_slot_out = torch.cat([word_slot_out[i][:word_seq_lens[i], :]
                                               for i in range(0, len(word_seq_lens))], dim=0)
            aligned_word_slot_out = op.char_word_alignment(flat_word_slot_out, char_seq_lens, word_seq_lens, align_info)

        # Pass char-level slot decoder
        char_feed_intent = feed_intent.unsqueeze(1).expand(-1, char_hiddens.size(1), -1)
        flat_char_feed_intent = torch.cat([char_feed_intent[i][:char_seq_lens[i], :]
                                           for i in range(0, len(char_seq_lens))], dim=0)
        flat_char_hiddens = torch.cat([char_hiddens[i][:char_seq_lens[i], :]
                                       for i in range(0, len(char_seq_lens))], dim=0)
        # _, char_slot_out = self.__char_slot_decoder(flat_char_hiddens, char_seq_lens, forced_input=forced_slot,
        #                                             extra_input=flat_char_feed_intent)
        pred_slot, char_slot_out = self.__char_slot_decoder(flat_char_hiddens, char_seq_lens, forced_input=forced_slot,
                                                    extra_input=flat_char_feed_intent,
                                                    extra_hidden=None if self.__args.single_channel_slot else aligned_word_slot_out)

        # # Slot Prediction, including the alignment between char slot out and word slot out
        # aligned_word_slot_out = op.char_word_alignment(word_slot_out, char_seq_lens, word_seq_lens, align_info)
        # slot_fusion_rate = torch.sigmoid(self.__slot_fusion_rate)
        # dropout_slot_fusion = self.__dropout_layer(
        #     slot_fusion_rate * char_slot_out + (1 - slot_fusion_rate) * aligned_word_slot_out)
        # pred_slot = self.__slot_linear_layer(dropout_slot_fusion)

        if n_predicts is None:
            return F.log_softmax(pred_slot, dim=1), F.log_softmax(pred_intent, dim=1)
        else:
            _, slot_index = pred_slot.topk(n_predicts, dim=1)
            _, intent_index = pred_intent.topk(n_predicts, dim=1)

            return slot_index.cpu().data.numpy().tolist(), \
                   intent_index.cpu().data.numpy().tolist()