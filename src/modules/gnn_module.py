# -*- coding: utf-8 -*-

"""
@CreateTime :       2020/4/3 15:25
@Author     :       dcteng
@File       :       gnn_module.py
@Software   :       PyCharm
@Framework  :       Pytorch
@LastModify :       2020/4/3 15:25
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.components import operation as op
from src.components.layers import FusionLayer, EmbeddingCollection, LSTMEncoder, LSTMDecoder, QKVAttention, SelfAttention, MLPAttention
from src.modules.graph import Graph

class GnnModelManager(nn.Module):

    def __init__(self, args, num_char, num_word, num_slot, num_intent, char_emb=None, word_emb=None):
        super(GnnModelManager, self).__init__()
        # hyper-parameters
        self.__num_char = num_char
        self.__num_word = num_word
        self.__num_slot = num_slot
        self.__num_intent = num_intent
        self.__args = args
        # embedding layer
        if not self.__args.unique_vocabulary:
            # Initialize an char embedding object.
            self.__char_embedding = EmbeddingCollection(self.__num_char, self.__args.char_embedding_dim,
                                                        pretrain_embedding=char_emb)
        else:
            assert self.__args.char_embedding_dim == self.__args.word_embedding_dim

        # set gnn parameters
        args.gnn_hidden_dim = self.__args.word_embedding_dim
        args.encoder_hidden_dim = self.__args.word_embedding_dim
        args.attention_output_dim = self.__args.word_embedding_dim // 2

        assert self.__args.encoder_hidden_dim == self.__args.word_embedding_dim
        assert self.__args.attention_output_dim == self.__args.word_embedding_dim // 2
        assert self.__args.word_embedding_dim == self.__args.gnn_hidden_dim

        # Initialize an word embedding object.
        self.__word_embedding = EmbeddingCollection(self.__num_word, self.__args.word_embedding_dim,
                                                    pretrain_embedding=word_emb)

        # Initialize an LSTM Encoder object for char level
        self.__char_encoder = LSTMEncoder(self.__args.char_embedding_dim, self.__args.encoder_hidden_dim,
                                          self.__args.dropout_rate)
        # Initialize an self-attention layer for char level
        self.__char_attention = SelfAttention(self.__args.char_embedding_dim, self.__args.char_attention_hidden_dim,
                                              self.__args.attention_output_dim, self.__args.dropout_rate)

        if self.__args.gnn_bidirectional:
            self.__encoder_output_dim = self.__args.gnn_hidden_dim * 2
        else:
            self.__encoder_output_dim = self.__args.gnn_hidden_dim

        # dropout layer
        self.__dropout_layer = nn.Dropout(self.__args.dropout_rate)

        # TODO: GNN Encoder
        self.__graph = Graph(args, self.__word_embedding)
        # TODO: GNN Output Encoder
        # self.__node_attention = SelfAttention(self.__encoder_output_dim, self.__args.char_attention_hidden_dim,
        #                                       self.__encoder_output_dim, self.__args.dropout_rate)
        # self.__edge_attention = SelfAttention(self.__encoder_output_dim, self.__args.char_attention_hidden_dim,
        #                                       self.__encoder_output_dim, self.__args.dropout_rate)
        # self.__node_attention = LSTMEncoder(self.__encoder_output_dim, self.__encoder_output_dim, self.__args.dropout_rate)
        # self.__edge_attention = LSTMEncoder(self.__encoder_output_dim, self.__encoder_output_dim, self.__args.dropout_rate)
        # self.__graph_encoder_output_dim = self.__encoder_output_dim + self.__args.word_embedding_dim
        self.__graph_encoder_output_dim = self.__encoder_output_dim

        # MLP Attention
        # TODO: Insert a linear layer between encoder and MLP Attention Layer ?
        if self.__args.intent_c2w_attention:
            self.__intent_c2w_attention = QKVAttention(
                self.__graph_encoder_output_dim, self.__graph_encoder_output_dim, self.__graph_encoder_output_dim,
                self.__args.intent_c2w_attention_hidden_dim, self.__graph_encoder_output_dim, self.__args.dropout_rate
            )
        else:
            self.__intent_sent_attention = MLPAttention(self.__graph_encoder_output_dim, self.__args.dropout_rate)

        self.__char_sent_attention = MLPAttention(self.__graph_encoder_output_dim, self.__args.dropout_rate)
        self.__word_sent_attention = MLPAttention(self.__graph_encoder_output_dim, self.__args.dropout_rate)

        self.__intent_fusion_layer = FusionLayer(self.__graph_encoder_output_dim, self.__graph_encoder_output_dim,
                                                 self.__args.dropout_rate, self.__args.intent_fusion_type,
                                                 class_num=self.__num_intent)

        # One-hot encoding for augment data feed.
        self.__intent_embedding = nn.Embedding(self.__num_intent, self.__num_intent)
        self.__intent_embedding.weight.data = torch.eye(self.__num_intent)
        self.__intent_embedding.weight.requires_grad = False

        # TODO: Now, lstm output dim of char-level slot decoder must be the same with that of word-level slot encoder
        # Initialize an Encoder object for word-level slot.
        self.__word_slot_encoder = LSTMEncoder(self.__graph_encoder_output_dim, self.__args.slot_decoder_hidden_dim,
                                               self.__args.dropout_rate,
                                               bidirectional=not self.__args.undirectional_word_level_slot_encoder,
                                               extra_dim=self.__num_intent)

        if self.__args.slot_c2w_attention:
            self.__slot_c2w_attention = QKVAttention(
                self.__args.slot_decoder_hidden_dim, self.__args.slot_decoder_hidden_dim, self.__args.slot_decoder_hidden_dim,
                self.__args.slot_c2w_attention_hidden_dim, self.__args.slot_decoder_hidden_dim, self.__args.dropout_rate
            )
        else:
            self.__slot_sent_attention = MLPAttention(self.__args.slot_decoder_hidden_dim, self.__args.dropout_rate)

        # Initialize an Decoder object for char-level slot.
        self.__char_slot_decoder = LSTMDecoder(
            self.__graph_encoder_output_dim,
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

        print('\ttransformer dropout rate:			                    {};'.format(self.__args.tf_drop_rate))
        print('\tcell dropout rate:                   	                {};'.format(self.__args.cell_drop_rate))
        print('\tuse gnn edge:                   	                    {};'.format(self.__args.gnn_use_edge))
        print('\tuse gnn global node:                   	            {};'.format(self.__args.gnn_use_global))
        print('\tbidirectional gnn:                   	                {};'.format(self.__args.gnn_bidirectional))
        print('\tgnn iterations:                   	                    {};'.format(self.__args.gnn_iters))
        print('\tgnn hidden dim:                   	                    {};'.format(self.__args.gnn_hidden_dim))
        print('\tgnn num head:                   	                    {};'.format(self.__args.gnn_num_head))
        print('\tgnn head dim:                   	                    {};'.format(self.__args.gnn_head_dim))
        print('\tnumber of sentence segmentation:                       {};'.format(self.__args.sent_seg_num))
        print('\tintent c2w attention:                                  {};'.format(self.__args.intent_c2w_attention))
        print('\tintent c2w attention hidden dim:                       {};'.format(self.__args.intent_c2w_attention_hidden_dim))
        print('\tslot c2w attention:                                    {};'.format(self.__args.slot_c2w_attention))
        print('\tslot c2w attention hidden dim:                         {};\n'.format(self.__args.slot_c2w_attention_hidden_dim))

        print('\nEnd of parameters show. Now training begins.\n\n')

    def forward(self, char_text, char_seq_lens, word_text, word_seq_lens, align_info, word_info,
                n_predicts=None, forced_slot=None, golden_intent=None):
        word_list = word_info[0]
        sent_segs = word_info[1]
        # Get embeddings
        char_tensor = self.__char_embedding(char_text) if not self.__args.unique_vocabulary else \
            self.__word_embedding(char_text)

        # Get mask
        device = char_tensor.device
        char_rmask, char_mmask = op.generate_mask(char_seq_lens, device)

        # TODO: take masking self-attention into account
        # Pass char encoder
        char_lstm_hiddens = self.__char_encoder(char_tensor, char_seq_lens)
        char_attention_hiddens = self.__char_attention(char_tensor, char_seq_lens, mmask=char_mmask)

        H = char_lstm_hiddens.shape[-1]
        # char_inputs_f = char_lstm_hiddens
        # char_inputs_b = char_lstm_hiddens
        char_inputs_f = torch.cat([char_lstm_hiddens[:, :, :(H // 2)], char_attention_hiddens], dim=-1)
        char_inputs_b = torch.cat([char_lstm_hiddens[:, :, (H // 2):], char_attention_hiddens], dim=-1)
        nodes, edges, edge_embs = self.__graph(word_list, char_inputs_f, char_inputs_b, char_rmask)

        # redisual connection: cat
        # nodes = torch.cat([char_lstm_hiddens, nodes], dim=-1)
        # edges = torch.cat([edge_embs, edges], dim=-1)

        if not (self.__args.single_channel_intent and self.__args.single_channel_slot):
            # construct sentence segmentation info
            word_hiddens, word_rmask, word_mmask, word_seq_lens, word_align_info, sent_seg_lens = \
                op.construct_sent_segmentation(nodes, edges, sent_segs, max_sent_seg_num=self.__args.sent_seg_num)
            # word_attention_hiddens = self.__edge_attention(word_hiddens, word_seq_lens, mmask=word_mmask)
            # word_attention_hiddens = self.__edge_attention(word_hiddens, word_seq_lens)
            # # word_hiddens += word_attention_hiddens
            # word_hiddens = torch.cat([word_hiddens, word_attention_hiddens], dim=-1)

        # encode graph output
        # nodes_attention = self.__node_attention(nodes, char_seq_lens, mmask=char_mmask)
        # nodes_attention = self.__node_attention(nodes, char_seq_lens)
        # # nodes += nodes_attention
        # nodes = torch.cat([nodes, nodes_attention], dim=-1)

        # [num_char, emb_dim]
        flat_char_hiddens = torch.cat([nodes[i][:char_seq_lens[i], :] for i in range(0, len(char_seq_lens))], dim=0)

        # MLP Attention for Intent Detection
        # [bs, max_char_seq_len, iehd] -> [bs, iehd]
        char_sent_output = self.__char_sent_attention(nodes, rmask=char_rmask)

        # Intent Prediction
        if not self.__args.single_channel_intent and self.__args.intent_c2w_attention:
            # [ns, mwsl, emb_dim] -> [nc, msn, emb_dim], [nc, msn]
            aligned_char_intent_out, char_sent_mask = op.batch_char_word_alignment(
                word_hiddens, char_seq_lens, word_seq_lens, word_align_info, sent_seg_lens)
            # [nc, emb_dim]
            dropout_flat_char_hiddens = self.__dropout_layer(flat_char_hiddens)
            dropout_aligned_char_intent_out = self.__dropout_layer(aligned_char_intent_out)
            word_sent_output = self.__intent_c2w_attention(dropout_flat_char_hiddens.unsqueeze(1),
                                                           dropout_aligned_char_intent_out,
                                                           dropout_aligned_char_intent_out,
                                                           mmask=char_sent_mask.unsqueeze(1)).squeeze(1)
            # [nc, emb_dim], [bs] -> [bs, mcsl, emb_dim], [bs, mcsl]
            word_sent_output, _ = op.pad_tensor_along_batch(word_sent_output, char_seq_lens)
            # [bs, mcsl, emb_dim] -> [bs, emb_dim]
            word_sent_output = self.__word_sent_attention(word_sent_output, rmask=char_rmask)
        elif not self.__args.single_channel_intent:
            # [num_sent, max_word_seq_len, iehd] -> [num_sent, iehd]
            word_sent_output = self.__word_sent_attention(word_hiddens, rmask=word_rmask)
            # [num_sent, iehd], [bs] -> [bs, max_sent_num, iehd], [bs, max_sent_num]
            word_sent_output, sent_rmask = op.pad_tensor_along_batch(word_sent_output, sent_seg_lens)
            # [bs, max_sent_num, iehd] -> [bs, iehd]
            word_sent_output = self.__intent_sent_attention(word_sent_output, rmask=sent_rmask)

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

        if not self.__args.single_channel_slot:
            # pass word-level slot encoder
            # [bs, intent_dim] -> [bs, max_sent_num, intent_dim]
            word_feed_intent = feed_intent.unsqueeze(1).expand(-1, max(sent_seg_lens), -1)
            # [bs, max_sent_num, intent_dim] -> [num_sent, intent_dim]
            word_feed_intent = torch.cat([word_feed_intent[i][:sent_seg_lens[i], :]
                                               for i in range(0, len(sent_seg_lens))], dim=0)
            # [num_sent, intent_dim] -> [num_sent, max_word_seq_len, intent_dim]
            word_feed_intent = word_feed_intent.unsqueeze(1).expand(-1, word_hiddens.size(1), -1)
            # [num_sent, max_word_seq_len, emb_dim] -> [num_sent, max_word_seq_len, slot_decoder_hidden_dim]
            word_slot_out = self.__word_slot_encoder(word_hiddens, word_seq_lens, extra_input=word_feed_intent)
            # [num_char, max_sent_num, slot_decoder_hidden_dim], [num_char, max_sent_num]
            aligned_char_slot_out, char_sent_mask = op.batch_char_word_alignment(
                word_slot_out, char_seq_lens, word_seq_lens, word_align_info, sent_seg_lens)

            if self.__args.slot_c2w_attention:
                # [nc, msn, sdhd], [nc, msn]
                aligned_char_slot_out = [aligned_char_slot_out, char_sent_mask]
            else:
                # [nc, msn, sdhd] -> [nc, sdhd]
                aligned_char_slot_out= self.__slot_sent_attention(aligned_char_slot_out, rmask=char_sent_mask)


        # Pass char-level slot decoder
        # [bs, max_char_seq_len, intent_dim]
        char_feed_intent = feed_intent.unsqueeze(1).expand(-1, nodes.size(1), -1)
        # [num_char, intent_dim]
        flat_char_feed_intent = torch.cat([char_feed_intent[i][:char_seq_lens[i], :]
                                           for i in range(0, len(char_seq_lens))], dim=0)
        # [num_char, num_slot], [num_char, slot_decoder_hidden_dim]
        pred_slot, char_slot_out = self.__char_slot_decoder(
            flat_char_hiddens, char_seq_lens,
            forced_input=forced_slot,
            extra_input=flat_char_feed_intent,
            extra_hidden=None if self.__args.single_channel_slot else aligned_char_slot_out,
            attention_module=None if not self.__args.slot_c2w_attention else self.__slot_c2w_attention)

        if n_predicts is None:
            return F.log_softmax(pred_slot, dim=1), F.log_softmax(pred_intent, dim=1)
        else:
            _, slot_index = pred_slot.topk(n_predicts, dim=1)
            _, intent_index = pred_intent.topk(n_predicts, dim=1)

            return slot_index.cpu().data.numpy().tolist(), \
                   intent_index.cpu().data.numpy().tolist()