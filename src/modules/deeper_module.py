# -*- coding: utf-8 -*-

"""
@CreateTime :       2020/4/20 9:59
@Author     :       dcteng
@File       :       deeper_module.py
@Software   :       PyCharm
@Framework  :       Pytorch
@LastModify :       2020/4/20 9:59
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.components import operation as op
from src.components.layers import FusionLayer, EmbeddingCollection, LSTMEncoder, LSTMDecoder
from src.components.layers import QKVAttention, SelfAttention, AttentiveModule, MLPAttention

MASK_VALUE = -2 ** 32 + 1

class DeeperModelManager(nn.Module):

    def __init__(self, args, num_char, num_word, num_slot, num_intent, char_emb=None, word_emb=None):
        super(DeeperModelManager, self).__init__()

        # hyper-parameters
        self.__num_char = num_char
        self.__num_word = num_word
        self.__num_slot = num_slot
        self.__num_intent = num_intent
        self.__bmes_dim = 10
        self.__args = args

        if not self.__args.unique_vocabulary:
            # Initialize an char embedding object.
            self.__char_embedding = EmbeddingCollection(self.__num_char, self.__args.char_embedding_dim,
                                                        pretrain_embedding=char_emb)
        else:
            assert self.__args.char_embedding_dim == self.__args.word_embedding_dim
        # Initialize an word embedding object.
        self.__word_embedding = EmbeddingCollection(self.__num_word, self.__args.word_embedding_dim,
                                                    pretrain_embedding=word_emb)

        # Initialize an LSTM Encoder object for char level
        self.__char_encoder = LSTMEncoder(self.__args.char_embedding_dim, self.__args.encoder_hidden_dim,
                                          self.__args.dropout_rate)
        # Initialize an self-attention layer for char level
        self.__char_attention = SelfAttention(self.__args.char_embedding_dim, self.__args.char_attention_hidden_dim,
                                              self.__args.attention_output_dim, self.__args.dropout_rate)
        self.__char_encoder_output_dim = self.__args.encoder_hidden_dim + self.__args.attention_output_dim

        # dropout layer
        self.__dropout_layer = nn.Dropout(self.__args.dropout_rate)

        # encoder that merges chars into word
        # TODO: set char2word encoder_hidden_dim and attention_output_dim ?
        # TODO: set char bmes embedding for char2word ?
        self.__bmes_embedding4char = nn.Embedding(4, self.__bmes_dim)
        self.__char2word_input_dim = self.__args.char_embedding_dim + (self.__bmes_dim if self.__args.use_char_bmes_emb else 0)
        self.__char2word_encoder = LSTMEncoder(self.__char2word_input_dim, self.__args.encoder_hidden_dim,
                                               self.__args.dropout_rate)
        if self.__args.use_c2w_encoder_qkv_input_linear:
            self.__char2word_attention = QKVAttention(
                self.__args.word_embedding_dim, self.__char2word_input_dim, self.__char2word_input_dim,
                self.__args.word_attention_hidden_dim, self.__args.attention_output_dim, self.__args.dropout_rate,
                input_linear=True, bilinear=False
            )
        else:
            self.__char2word_attention = AttentiveModule(
                self.__args.word_embedding_dim, self.__char2word_input_dim, self.__char2word_input_dim,
                self.__args.attention_output_dim, self.__args.dropout_rate,
                bilinear=self.__args.bilinear_attention
            )

        self.__word_encoder_dim = self.__args.word_embedding_dim + self.__args.encoder_hidden_dim + self.__args.attention_output_dim
        # 4: none
        self.__bmes_embedding4word = nn.Embedding(5, self.__bmes_dim)
        self.__word_encoder_output_dim = self.__word_encoder_dim + (self.__bmes_dim if self.__args.use_word_bmes_emb else 0)
        self.__word_none_embedding = nn.Parameter(torch.randn(self.__word_encoder_dim), requires_grad=True)

        # Intent detection
        self.__char_sent_attention = MLPAttention(self.__char_encoder_output_dim, self.__args.dropout_rate)

        if self.__args.use_c2w_qkv_input_linear:
            self.__intent_c2w_attention = QKVAttention(
                self.__char_encoder_output_dim, self.__word_encoder_output_dim, self.__word_encoder_output_dim,
                self.__args.intent_c2w_attention_hidden_dim, self.__char_encoder_output_dim, self.__args.dropout_rate,
                input_linear=True, bilinear=False
            )
        else:
            self.__intent_c2w_attention = AttentiveModule(
                self.__char_encoder_output_dim, self.__word_encoder_output_dim, self.__word_encoder_output_dim,
                self.__char_encoder_output_dim, self.__args.dropout_rate,
                bilinear=self.__args.bilinear_attention
            )
        
        # self.__word_encoder = LSTMEncoder(self.__args.char_embedding_dim, self.__args.encoder_hidden_dim,
        #                                   self.__args.dropout_rate)
        # # Initialize an self-attention layer for char level
        # self.__word_attention = SelfAttention(self.__args.char_embedding_dim, self.__args.word_attention_hidden_dim,
        #                                       self.__args.attention_output_dim, self.__args.dropout_rate)

        self.__char_encoder_output_dim = self.__args.encoder_hidden_dim + self.__args.attention_output_dim

        self.__word_sent_attention = MLPAttention(self.__char_encoder_output_dim, self.__args.dropout_rate)

        # TODO: layerNorm ?
        # TODO: new fusion style ?
        self.__intent_fusion_layer = FusionLayer(self.__char_encoder_output_dim, self.__char_encoder_output_dim,
                                                 self.__args.dropout_rate, self.__args.intent_fusion_type,
                                                 class_num=self.__num_intent)

        # One-hot encoding for augment data feed.
        self.__intent_embedding = nn.Embedding(self.__num_intent, self.__num_intent)
        self.__intent_embedding.weight.data = torch.eye(self.__num_intent)
        self.__intent_embedding.weight.requires_grad = False

        # TODO: Insert an Encoder object for word-level slot ?
        # Initialize an Encoder object for word-level slot.
        self.__word_slot_encoder = LSTMEncoder(self.__char_encoder_output_dim, self.__args.slot_decoder_hidden_dim,
                                               self.__args.dropout_rate,
                                               bidirectional=not self.__args.undirectional_word_level_slot_encoder,
                                               extra_dim=self.__num_intent)

        # Initialize an Decoder object for char-level slot.
        self.__char_slot_decoder = LSTMDecoder(
            self.__char_encoder_output_dim,
            self.__args.slot_decoder_hidden_dim,
            self.__num_slot, self.__args.dropout_rate, self.__args.slot_fusion_type,
            embedding_dim=self.__args.slot_embedding_dim,
            extra_input_dim=self.__num_intent,
            extra_hidden_dim=None if self.__args.single_channel_slot else self.__args.slot_decoder_hidden_dim
        )

    def show_summary(self):
        """
        print the abstract of the defined model.
        """
        print('Model parameters are listed as follows:\n')

        print('\tdropout rate:						                    {};'.format(self.__args.dropout_rate))
        print('\tdifferentiable:						                {};'.format(self.__args.differentiable))
        print('\tunique vocabulary:						                {};'.format(self.__args.unique_vocabulary))
        print('\tgolden intent:                                         {};'.format(self.__args.golden_intent))
        print('\tsingle channel intent:                                 {};'.format(self.__args.single_channel_intent))
        print('\tsingle channel slot:                                   {};'.format(self.__args.single_channel_slot))
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
        print('\tdimension of slot decoder hidden:  	                {};\n'.format(self.__args.slot_decoder_hidden_dim))

        print('\tuse char bmes embedding:                               {};'.format(self.__args.use_char_bmes_emb))
        print('\tuse word bmes embedding:                               {};'.format(self.__args.use_word_bmes_emb))
        print('\tuse char2word encoder qkv input linear layer:          {};'.format(self.__args.use_c2w_encoder_qkv_input_linear))
        print('\tuse char2word qkv input linear layer:                  {};'.format(self.__args.use_c2w_qkv_input_linear))
        print('\tbilinear attention:                                    {};'.format(self.__args.bilinear_attention))
        print('\tintent c2w attention hidden dim:                       {};\n'.format(self.__args.intent_c2w_attention_hidden_dim))

        print('\nEnd of parameters show. Now training begins.\n\n')

    def construct_word_embedding(self, char_tensor, char_seq_lens, word_list, device):
        """
        char_tensor: [bs, mcsl, ced]
        word_list: list of list of [word_ids, word_lens]
        :return:
        """
        word_ids, char_idxs4word, char_bmes, word_lens, batch_word_idxs4char, batch_bmes4char, batch_word_num4char = \
            op.construct_word4char(word_list, only_one_none=True)

        flat_char_tensor = torch.cat([char_tensor[i][:char_seq_lens[i], :] for i in range(0, len(char_seq_lens))], dim=0)
        # [num_word, wed]
        word_tensor = self.__word_embedding(torch.tensor(word_ids, device=device))
        # [num_char4word, ced]
        char_tensor4word = torch.index_select(flat_char_tensor, dim=0, index=torch.tensor(char_idxs4word, device=device))
        if self.__args.use_char_bmes_emb:
            char_bmes4word = self.__bmes_embedding4char(torch.tensor(char_bmes, device=device))
            char_tensor4word = torch.cat([char_tensor4word, char_bmes4word], dim=-1)

        # [num_char4word, ced], [num_word] -> [num_word, max_num_char4word, ced], [num_word, max_num_char4word]
        char_tensor4word, char_mask4word = op.pad_tensor_along_batch(char_tensor4word, word_lens)

        # [num_word, max_num_char4word, ehd]
        char_lstm_hiddens4word = self.__char2word_encoder(char_tensor4word, word_lens)
        H = char_lstm_hiddens4word.shape[-1]
        # [num_word, 1, H]
        gather_index = (torch.tensor(word_lens, device=device) - 1).unsqueeze(1).unsqueeze(-1).expand(-1, -1, H)
        # [num_word, ehd]
        word_lstm_hiddens = torch.cat([torch.gather(char_lstm_hiddens4word, dim=1, index=gather_index)[:, 0, :(H // 2)],
                                       char_lstm_hiddens4word[:, 0, (H // 2):]], dim=-1)

        dropout_word_tensor = self.__dropout_layer(word_tensor)
        dropout_char_tensor4word = self.__dropout_layer(char_tensor4word)
        # [num_word, aod]
        word_attention_hiddens = self.__char2word_attention(dropout_word_tensor.unsqueeze(1),
                                                            dropout_char_tensor4word,
                                                            dropout_char_tensor4word,
                                                            mmask=char_mask4word.unsqueeze(1)).squeeze(1)
        # [num_word, wed + ehd + aod]
        word_hiddens = torch.cat([word_tensor, word_lstm_hiddens, word_attention_hiddens], dim=-1)
        word_hiddens = torch.cat([self.__word_none_embedding[None, :], word_hiddens], dim=0)

        # [total_word_num4char, wed + ehd + aod]
        word_hiddens4char = torch.index_select(word_hiddens, dim=0, index=torch.tensor(batch_word_idxs4char, device=device))
        if self.__args.use_word_bmes_emb:
            word_bmes4char = self.__bmes_embedding4word(torch.tensor(batch_bmes4char, device=device))
            word_hiddens4char = torch.cat([word_bmes4char, word_hiddens4char], dim=-1)

        # [total_word_num4char, weod], [num_char]
        # -> [num_char, max_num_word, weod], [nc, max_num_word]
        word_hiddens4char, word_mask4char = op.pad_tensor_along_batch(word_hiddens4char, batch_word_num4char)

        return word_hiddens4char, word_mask4char

    def forward(self, char_text, char_seq_lens, word_text, word_seq_lens, align_info, word_info,
                n_predicts=None, forced_slot=None, golden_intent=None):
        word_list = word_info[0]

        # Get embeddings
        char_tensor = self.__char_embedding(char_text) if not self.__args.unique_vocabulary else \
            self.__word_embedding(char_text)

        # Get mask
        device = char_tensor.device
        char_rmask, char_mmask = op.generate_mask(char_seq_lens, device)

        # Pass char encoder
        char_lstm_hiddens = self.__char_encoder(char_tensor, char_seq_lens)
        char_attention_hiddens = self.__char_attention(char_tensor, char_seq_lens, mmask=char_mmask)
        char_hiddens = torch.cat([char_attention_hiddens, char_lstm_hiddens], dim=-1)

        # [nc, ceod]
        flat_char_hiddens = torch.cat([char_hiddens[i][:char_seq_lens[i], :] for i in range(0, len(char_seq_lens))], dim=0)

        if not self.__args.single_channel_intent or not self.__args.single_channel_slot:
            # [num_char, max_num_word, weod], [nc, max_num_word]
            word_hiddens4char, word_mask4char = self.construct_word_embedding(char_tensor, char_seq_lens, word_list, device)
            # merge words for each char
            dropout_flat_char_hiddens = self.__dropout_layer(flat_char_hiddens)
            dropout_word_hiddens4char = self.__dropout_layer(word_hiddens4char)
            # [nc, ceod]
            word_output4char = self.__intent_c2w_attention(dropout_flat_char_hiddens.unsqueeze(1),
                                                           dropout_word_hiddens4char,
                                                           dropout_word_hiddens4char,
                                                           mmask=word_mask4char.unsqueeze(1)).squeeze(1)
            # [nc, ceod], [bs] -> [bs, mcsl, ceod], [bs, mcsl]
            batch_word_output4char, _ = op.pad_tensor_along_batch(word_output4char, char_seq_lens)

        # Intent Detection
        # [bs, max_char_seq_len, ceod] -> [bs, ceod]
        char_sent_output = self.__char_sent_attention(char_hiddens, rmask=char_rmask)

        if not self.__args.single_channel_intent:
            # [bs, mcsl, ceod] -> [bs, ceod]
            word_sent_output = self.__word_sent_attention(batch_word_output4char, rmask=char_rmask)

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

        # [bs, mcsl, intent_dim]
        char_feed_intent = feed_intent.unsqueeze(1).expand(-1, char_hiddens.size(1), -1)

        if not self.__args.single_channel_slot:
            # pass word-level slot encoder
            # [bs, mcsl, ceod] -> [bs, mcsl, slot_decoder_hidden_dim]
            word_slot_out4char = self.__word_slot_encoder(batch_word_output4char, char_seq_lens, extra_input=char_feed_intent)
            flat_word_slot_out4char = torch.cat([word_slot_out4char[i][:char_seq_lens[i], :]
                                                 for i in range(0, len(char_seq_lens))], dim=0)

        # Pass char-level slot decoder
        # [nc, intent_dim]
        flat_char_feed_intent = torch.cat([char_feed_intent[i][:char_seq_lens[i], :]
                                           for i in range(0, len(char_seq_lens))], dim=0)
        # [nc, num_slot], [nc, sdhd]
        pred_slot, char_slot_out = self.__char_slot_decoder(
            flat_char_hiddens, char_seq_lens,
            forced_input=forced_slot,
            extra_input=flat_char_feed_intent,
            extra_hidden=None if self.__args.single_channel_slot else flat_word_slot_out4char,
            attention_module=None)

        if n_predicts is None:
            return F.log_softmax(pred_slot, dim=1), F.log_softmax(pred_intent, dim=1)
        else:
            _, slot_index = pred_slot.topk(n_predicts, dim=1)
            _, intent_index = pred_intent.topk(n_predicts, dim=1)

            return slot_index.cpu().data.numpy().tolist(), \
                   intent_index.cpu().data.numpy().tolist()