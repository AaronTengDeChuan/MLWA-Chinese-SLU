# -*- coding: utf-8 -*-

"""
@CreateTime :       2020/4/3 15:13
@Author     :       dcteng
@File       :       layers.py
@Software   :       PyCharm
@Framework  :       Pytorch
@LastModify :       2020/4/3 15:13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.components.layers import NullOp, MultiHeadAtt, GloAtt, Nodes_Cell, Edges_Cell, Global_Cell


class Graph(nn.Module):
    def __init__(self, args, word_embedding):
        super(Graph, self).__init__()

        self.char_emb_dim = args.char_embedding_dim
        self.word_emb_dim = args.word_embedding_dim
        self.hidden_dim = args.gnn_hidden_dim
        self.num_head = args.gnn_num_head  # 5 10 20
        self.head_dim = args.gnn_head_dim  # 10 20
        self.iters = args.gnn_iters
        self.bmes_dim = 10
        self.length_dim = 10
        self.max_word_length = 5
        self.emb_dropout_rate = args.dropout_rate
        self.tf_dropout_rate = args.tf_drop_rate
        self.cell_dropout_rate = args.cell_drop_rate
        self.use_global = args.gnn_use_global
        self.use_edge = args.gnn_use_edge
        self.bidirectional = args.gnn_bidirectional

        if self.use_edge:
            # word embedding
            self.word_embedding = word_embedding
            # bmes embedding
            self.bmes_embedding = nn.Embedding(4, self.bmes_dim)

        # length embedding
        self.length_embedding = nn.Embedding(self.max_word_length, self.length_dim)

        self.dropout = nn.Dropout(self.emb_dropout_rate)
        self.norm = nn.LayerNorm(self.hidden_dim)
        # self.norm = NullOp()

        if self.use_edge:
            # Node aggregation module
            self.edge2node_f = nn.ModuleList(
                [MultiHeadAtt(self.hidden_dim, self.hidden_dim * 2 + self.length_dim,
                              nhead=self.num_head, head_dim=self.head_dim, dropout=self.tf_dropout_rate)
                 for _ in range(self.iters)])
            # Edge aggregation module
            self.node2edge_f = nn.ModuleList(
                [MultiHeadAtt(self.hidden_dim, self.hidden_dim + self.bmes_dim, nhead=self.num_head,
                              head_dim=self.head_dim, dropout=self.tf_dropout_rate)
                 for _ in range(self.iters)])

        else:
            # Node aggregation module
            self.edge2node_f = nn.ModuleList(
                [MultiHeadAtt(self.hidden_dim, self.hidden_dim + self.length_dim,
                              nhead=self.num_head, head_dim=self.head_dim, dropout=self.tf_dropout_rate)
                 for _ in range(self.iters)])

        if self.use_global:
            # Global Node aggregation module
            self.glo_att_f_node = nn.ModuleList(
                [GloAtt(self.hidden_dim, nhead=self.num_head, head_dim=self.head_dim, dropout=self.tf_dropout_rate)
                 for _ in range(self.iters)])

            if self.use_edge:
                self.glo_att_f_edge = nn.ModuleList(
                    [GloAtt(self.hidden_dim, nhead=self.num_head, head_dim=self.head_dim, dropout=self.tf_dropout_rate)
                     for _ in range(self.iters)])

            # Updating modules
            if self.use_edge:
                self.glo_rnn_f = Global_Cell(self.hidden_dim * 3, self.hidden_dim, dropout=self.cell_dropout_rate)
                self.node_rnn_f = Nodes_Cell(self.hidden_dim * 5, self.hidden_dim, dropout=self.cell_dropout_rate)
                self.edge_rnn_f = Edges_Cell(self.hidden_dim * 4, self.hidden_dim, dropout=self.cell_dropout_rate)
            else:
                self.glo_rnn_f = Global_Cell(self.hidden_dim * 2, self.hidden_dim, dropout=self.cell_dropout_rate)
                self.node_rnn_f = Nodes_Cell(self.hidden_dim * 4, self.hidden_dim, dropout=self.cell_dropout_rate)

        else:
            # Updating modules
            self.node_rnn_f = Nodes_Cell(self.hidden_dim * 3, self.hidden_dim, use_global=False, dropout=self.cell_dropout_rate)
            if self.use_edge:
                self.edge_rnn_f = Edges_Cell(self.hidden_dim * 2, self.hidden_dim, use_global=False, dropout=self.cell_dropout_rate)

        if self.bidirectional:

            if self.use_edge:
                # Node aggregation module
                self.edge2node_b = nn.ModuleList(
                    [MultiHeadAtt(self.hidden_dim, self.hidden_dim * 2 + self.length_dim,
                                  nhead=self.num_head, head_dim=self.head_dim, dropout=self.tf_dropout_rate)
                     for _ in range(self.iters)])
                # Edge aggregation module
                self.node2edge_b = nn.ModuleList(
                    [MultiHeadAtt(self.hidden_dim, self.hidden_dim + self.bmes_dim, nhead=self.num_head,
                                  head_dim=self.head_dim, dropout=self.tf_dropout_rate)
                     for _ in range(self.iters)])

            else:
                # Node aggregation module
                self.edge2node_b = nn.ModuleList(
                    [MultiHeadAtt(self.hidden_dim, self.hidden_dim + self.length_dim,
                                  nhead=self.num_head, head_dim=self.head_dim, dropout=self.tf_dropout_rate)
                     for _ in range(self.iters)])

            if self.use_global:
                # Global Node aggregation module
                self.glo_att_b_node = nn.ModuleList(
                    [GloAtt(self.hidden_dim, nhead=self.num_head, head_dim=self.head_dim, dropout=self.tf_dropout_rate)
                     for _ in range(self.iters)])
                if self.use_edge:
                    self.glo_att_b_edge = nn.ModuleList(
                        [GloAtt(self.hidden_dim, nhead=self.num_head, head_dim=self.head_dim, dropout=self.tf_dropout_rate)
                         for _ in range(self.iters)])

                # Updating modules
                if self.use_edge:
                    self.glo_rnn_b = Global_Cell(self.hidden_dim * 3, self.hidden_dim, self.cell_dropout_rate)
                    self.node_rnn_b = Nodes_Cell(self.hidden_dim * 5, self.hidden_dim, self.cell_dropout_rate)
                    self.edge_rnn_b = Edges_Cell(self.hidden_dim * 4, self.hidden_dim, self.cell_dropout_rate)
                else:
                    self.glo_rnn_b = Global_Cell(self.hidden_dim * 2, self.hidden_dim, self.cell_dropout_rate)
                    self.node_rnn_b = Nodes_Cell(self.hidden_dim * 4, self.hidden_dim, self.cell_dropout_rate)

            else:
                # Updating modules
                self.node_rnn_b = Nodes_Cell(self.hidden_dim * 3, self.hidden_dim, use_global=False, dropout=self.cell_dropout_rate)
                if self.use_edge:
                    self.edge_rnn_b = Edges_Cell(self.hidden_dim * 2, self.hidden_dim, use_global=False, dropout=self.cell_dropout_rate)

        if self.bidirectional:
            output_dim = self.hidden_dim * 2
        else:
            output_dim = self.hidden_dim

        self.node_layer_att_W = nn.Linear(output_dim, 1)
        if self.use_edge:
            self.edge_layer_att_W = nn.Linear(output_dim, 1)

    def construct_graph(self, batch_size, seq_len, word_list):

        if self.cuda:
            device = 'cuda'
        else:
            device = 'cpu'
        if self.use_edge:
            unk_index = torch.tensor(1, device=device)
            unk_emb = self.word_embedding(unk_index)

            bmes_emb_b = self.bmes_embedding(torch.tensor(0, device=device))
            bmes_emb_m = self.bmes_embedding(torch.tensor(1, device=device))
            bmes_emb_e = self.bmes_embedding(torch.tensor(2, device=device))
            bmes_emb_s = self.bmes_embedding(torch.tensor(3, device=device))

        sen_nodes_mask_list = []
        sen_words_length_list = []
        sen_words_mask_f_list = []
        sen_words_mask_b_list = []
        sen_words_id_list = []
        sen_word_embed_list = []
        sen_bmes_embed_list = []
        max_edge_num = -1

        for sen in range(batch_size):
            sen_nodes_mask = torch.zeros([1, seq_len], device=device).byte()
            sen_words_length = torch.zeros([1, self.length_dim], device=device)
            sen_words_mask_f = torch.zeros([1, seq_len], device=device).byte()
            sen_words_mask_b = torch.zeros([1, seq_len], device=device).byte()

            if self.use_edge:
                sen_edge_id = torch.tensor(1, device=device)[None]
                sen_word_embed = unk_emb[None, :]
                sen_bmes_embed = torch.zeros([1, seq_len, self.bmes_dim], device=device)

            num_words = sum([len(wl[0]) for wl in word_list[sen] if wl])
            for w in range(seq_len):
                if w < len(word_list[sen]) and word_list[sen][w]:
                    for word, word_len in zip(word_list[sen][w][0], word_list[sen][w][1]):

                        if word_len <= self.max_word_length:
                            word_length_index = torch.tensor(word_len-1, device=device)
                        else:
                            word_length_index = torch.tensor(self.max_word_length - 1, device=device)
                        word_length = self.length_embedding(word_length_index)
                        sen_words_length = torch.cat([sen_words_length, word_length[None, :]], 0)

                        # mask: Masked elements are marked by 1, batch_size * word_num * seq_len
                        nodes_mask = torch.ones([1, seq_len], device=device).byte()
                        words_mask_f = torch.ones([1, seq_len], device=device).byte()
                        words_mask_b = torch.ones([1, seq_len], device=device).byte()

                        words_mask_f[0, w + word_len - 1] = 0
                        sen_words_mask_f = torch.cat([sen_words_mask_f, words_mask_f], 0)

                        words_mask_b[0, w] = 0
                        sen_words_mask_b = torch.cat([sen_words_mask_b, words_mask_b], 0)

                        if self.use_edge:
                            word_index = torch.tensor(word, device=device)
                            sen_edge_id = torch.cat([sen_edge_id, word_index[None]], dim=0)
                            word_embedding = self.word_embedding(word_index)
                            sen_word_embed = torch.cat([sen_word_embed, word_embedding[None, :]], 0)

                            bmes_embed = torch.zeros([1, seq_len, self.bmes_dim], device=device)

                            for index in range(word_len):
                                nodes_mask[0, w + index] = 0
                                if word_len == 1:
                                    bmes_embed[0, w + index, :] = bmes_emb_s
                                elif index == 0:
                                    bmes_embed[0, w + index, :] = bmes_emb_b
                                elif index == word_len - 1:
                                    bmes_embed[0, w + index, :] = bmes_emb_e
                                else:
                                    bmes_embed[0, w + index, :] = bmes_emb_m

                            sen_bmes_embed = torch.cat([sen_bmes_embed, bmes_embed], 0)
                            sen_nodes_mask = torch.cat([sen_nodes_mask, nodes_mask], 0)

            if sen_words_mask_f.size(0) > max_edge_num:
                max_edge_num = sen_words_mask_f.size(0)
            sen_words_mask_f_list.append(sen_words_mask_f.unsqueeze_(0))
            sen_words_mask_b_list.append(sen_words_mask_b.unsqueeze_(0))
            sen_words_length_list.append(sen_words_length.unsqueeze_(0))
            if self.use_edge:
                assert num_words + 1 == sen_word_embed.size(0)
                sen_nodes_mask_list.append(sen_nodes_mask.unsqueeze_(0))
                sen_words_id_list.append(sen_edge_id.unsqueeze_(0))
                sen_word_embed_list.append(sen_word_embed.unsqueeze_(0))
                sen_bmes_embed_list.append(sen_bmes_embed.unsqueeze_(0))

        edges_mask = torch.zeros([batch_size, max_edge_num], device=device)
        batch_words_mask_f = torch.ones([batch_size, max_edge_num, seq_len], device=device).byte()
        batch_words_mask_b = torch.ones([batch_size, max_edge_num, seq_len], device=device).byte()
        batch_words_length = torch.zeros([batch_size, max_edge_num, self.length_dim], device=device)
        if self.use_edge:
            batch_nodes_mask = torch.zeros([batch_size, max_edge_num, seq_len], device=device).byte()
            batch_word_embed = torch.zeros([batch_size, max_edge_num, self.word_emb_dim], device=device)
            batch_bmes_embed = torch.zeros([batch_size, max_edge_num, seq_len, self.bmes_dim], device=device)
        else:
            batch_word_embed = None
            batch_bmes_embed = None
            batch_nodes_mask = None

        for index in range(batch_size):
            curr_edge_num = sen_words_mask_f_list[index].size(1)
            edges_mask[index, 0:curr_edge_num] = 1.
            batch_words_mask_f[index, 0:curr_edge_num, :] = sen_words_mask_f_list[index]
            batch_words_mask_b[index, 0:curr_edge_num, :] = sen_words_mask_b_list[index]
            batch_words_length[index, 0:curr_edge_num, :] = sen_words_length_list[index]
            if self.use_edge:
                batch_nodes_mask[index, 0:curr_edge_num, :] = sen_nodes_mask_list[index]
                batch_word_embed[index, 0:curr_edge_num, :] = sen_word_embed_list[index]
                batch_bmes_embed[index, 0:curr_edge_num, :, :] = sen_bmes_embed_list[index]

        return batch_word_embed, batch_bmes_embed, batch_nodes_mask, batch_words_mask_f, \
               batch_words_mask_b, batch_words_length, edges_mask

    def update_graph(self, word_list, word_inputs_f, word_inputs_b,  mask):
        mask = mask.float()
        B, L, _ = word_inputs_f.size()

        edge_embs, bmes_embs, nodes_mask, words_mask_f, words_mask_b, words_length, edges_mask = \
            self.construct_graph(B, L, word_list)

        _, N, _ = words_mask_f.size()

        if self.use_edge:
            edge_embs = self.dropout(edge_embs)

        # forward direction digraph
        nodes_f = self.dropout(word_inputs_f)
        nodes_f = nodes_f * mask.unsqueeze(2)
        nodes_f_cat = nodes_f[:, None, :, :]
        _, _, H = nodes_f.size()

        if self.use_edge:
            edges_f = edge_embs * edges_mask.unsqueeze(2)
            edges_f_cat = edges_f[:, None, :, :]

            if self.use_global:
                glo_f = edges_f.sum(1, keepdim=True) / edges_mask.sum(1, keepdim=True).unsqueeze_(2) + \
                        nodes_f.sum(1, keepdim=True) / mask.sum(1, keepdim=True).unsqueeze_(2)
                glo_f_cat = glo_f[:, None, :, :]

        else:
            if self.use_global:
                glo_f = (nodes_f * mask.unsqueeze(2)).sum(1, keepdim=True) / mask.sum(1, keepdim=True).unsqueeze_(2)
                glo_f_cat = glo_f[:, None, :, :]

        for i in range(self.iters):

            # Attention-based aggregation
            if self.use_edge and N > 1:
                bmes_nodes_f = torch.cat([nodes_f.unsqueeze(2).expand(B, L, N, H), bmes_embs.transpose(1, 2)], -1)
                edges_att_f = self.node2edge_f[i](edges_f, bmes_nodes_f, nodes_mask.transpose(1, 2))

            nodes_begin_f = torch.sum(nodes_f[:, None, :, :] * (1 - words_mask_b)[:, :, :, None].float(), 2)
            nodes_begin_f = torch.cat([torch.zeros([B, 1, H], device=nodes_f.device), nodes_begin_f[:, 1:N, :]], 1)

            if self.use_edge:
                nodes_att_f = self.edge2node_f[i](nodes_f, torch.cat([edges_f, nodes_begin_f, words_length], -1).unsqueeze(2), words_mask_f)
                if self.use_global:
                    glo_att_f = torch.cat([self.glo_att_f_node[i](glo_f, nodes_f, (1 - mask).byte()),
                                           self.glo_att_f_edge[i](glo_f, edges_f, (1 - edges_mask).byte())], -1)
            else:
                nodes_att_f = self.edge2node_f[i](nodes_f, torch.cat([nodes_begin_f, words_length], -1).unsqueeze(2), words_mask_f)
                if self.use_global:
                    glo_att_f = self.glo_att_f_node[i](glo_f, nodes_f, (1 - mask).byte())

            # RNN-based update
            if self.use_edge and N > 1:
                if self.use_global:
                    edges_f = torch.cat([edges_f[:, 0:1, :], self.edge_rnn_f(edges_f[:, 1:N, :],
                                         edges_att_f[:, 1:N, :], glo_att_f.expand(B, N-1, H*2))], 1)
                else:
                    edges_f = torch.cat([edges_f[:, 0:1, :], self.edge_rnn_f(edges_f[:, 1:N, :], edges_att_f[:, 1:N, :])], 1)

                edges_f_cat = torch.cat([edges_f_cat, edges_f[:, None, :, :]], 1)
                edges_f = torch.cat([edges_f[:, 0:1, :], self.norm(torch.sum(edges_f_cat[:, :, 1:N, :], 1))], 1)

            nodes_f_r = torch.cat([torch.zeros([B, 1, self.hidden_dim], device=nodes_f.device), nodes_f[:, 0:(L-1), :]], 1)

            if self.use_global:
                nodes_f = self.node_rnn_f(nodes_f, nodes_f_r, nodes_att_f, glo_att_f.expand(B, L, -1))
            else:
                nodes_f = self.node_rnn_f(nodes_f, nodes_f_r, nodes_att_f)

            nodes_f_cat = torch.cat([nodes_f_cat, nodes_f[:, None, :, :]], 1)
            nodes_f = self.norm(torch.sum(nodes_f_cat, 1))

            if self.use_global:
                glo_f = self.glo_rnn_f(glo_f, glo_att_f)
                glo_f_cat = torch.cat([glo_f_cat, glo_f[:, None, :, :]], 1)
                glo_f = self.norm(torch.sum(glo_f_cat, 1))

        nodes_cat = nodes_f_cat
        if self.use_edge:
            edges_cat = edges_f_cat

        # backward direction digraph
        if self.bidirectional:
            nodes_b = self.dropout(word_inputs_b)
            nodes_b = nodes_b * mask.unsqueeze(2)
            nodes_b_cat = nodes_b[:, None, :, :]

            if self.use_edge:
                edges_b = edge_embs * edges_mask.unsqueeze(2)
                edges_b_cat = edges_b[:, None, :, :]
                if self.use_global:
                    glo_b = edges_b.sum(1, keepdim=True) / edges_mask.sum(1, keepdim=True).unsqueeze_(2) + \
                            nodes_b.sum(1, keepdim=True) / mask.sum(1, keepdim=True).unsqueeze_(2)
                    glo_b_cat = glo_b[:, None, :, :]

            else:
                if self.use_global:
                    glo_b = nodes_b.sum(1, keepdim=True) / mask.sum(1, keepdim=True).unsqueeze_(2)
                    glo_b_cat = glo_b[:, None, :, :]

            for i in range(self.iters):

                # Attention-based aggregation
                if self.use_edge and N > 1:
                    bmes_nodes_b = torch.cat([nodes_b.unsqueeze(2).expand(B, L, N, H), bmes_embs.transpose(1, 2)], -1)
                    edges_att_b = self.node2edge_b[i](edges_b, bmes_nodes_b, nodes_mask.transpose(1, 2))

                nodes_begin_b = torch.sum(nodes_b[:, None, :, :] * (1 - words_mask_f)[:, :, :, None].float(), 2)
                nodes_begin_b = torch.cat([torch.zeros([B, 1, H], device=nodes_b.device), nodes_begin_b[:, 1:N, :]], 1)

                if self.use_edge:
                    nodes_att_b = self.edge2node_b[i](nodes_b, torch.cat([edges_b, nodes_begin_b, words_length], -1).unsqueeze(2), words_mask_b)
                    if self.use_global:
                        glo_att_b = torch.cat([self.glo_att_b_node[i](glo_b, nodes_b, (1-mask).byte()),
                                               self.glo_att_b_edge[i](glo_b, edges_b, (1-edges_mask).byte())], -1)
                else:
                    nodes_att_b = self.edge2node_b[i](nodes_b, torch.cat([nodes_begin_b, words_length], -1).unsqueeze(2), words_mask_b)
                    if self.use_global:
                        glo_att_b = self.glo_att_b_node[i](glo_b, nodes_b, (1-mask).byte())

                # RNN-based update
                if self.use_edge and N > 1:
                    if self.use_global:
                        edges_b = torch.cat([edges_b[:, 0:1, :], self.edge_rnn_b(edges_b[:, 1:N, :],
                                             edges_att_b[:, 1:N, :], glo_att_b.expand(B, N-1, H*2))], 1)
                    else:
                        edges_b = torch.cat([edges_b[:, 0:1, :], self.edge_rnn_b(edges_b[:, 1:N, :], edges_att_b[:, 1:N, :])], 1)

                    edges_b_cat = torch.cat([edges_b_cat, edges_b[:, None, :, :]], 1)
                    edges_b = torch.cat([edges_b[:, 0:1, :], self.norm(torch.sum(edges_b_cat[:, :, 1:N, :], 1))], 1)

                nodes_b_r = torch.cat([nodes_b[:, 1:L, :], torch.zeros([B, 1, self.hidden_dim], device=nodes_b.device)], 1)

                if self.use_global:
                    nodes_b = self.node_rnn_b(nodes_b, nodes_b_r, nodes_att_b, glo_att_b.expand(B, L, -1))
                else:
                    nodes_b = self.node_rnn_b(nodes_b, nodes_b_r, nodes_att_b)

                nodes_b_cat = torch.cat([nodes_b_cat, nodes_b[:, None, :, :]], 1)
                nodes_b = self.norm(torch.sum(nodes_b_cat, 1))

                if self.use_global:
                    glo_b = self.glo_rnn_b(glo_b, glo_att_b)
                    glo_b_cat = torch.cat([glo_b_cat, glo_b[:, None, :, :]], 1)
                    glo_b = self.norm(torch.sum(glo_b_cat, 1))

            nodes_cat = torch.cat([nodes_f_cat, nodes_b_cat], -1)
            if self.use_edge:
                edges_cat = torch.cat([edges_f_cat, edges_b_cat], -1)

        node_layer_att = torch.sigmoid(self.node_layer_att_W(nodes_cat))
        node_layer_alpha = F.softmax(node_layer_att, 1)
        nodes = torch.sum(node_layer_alpha * nodes_cat, 1)
        edges = None
        if self.use_edge:
            edge_layer_att = torch.sigmoid(self.edge_layer_att_W(edges_cat))
            edge_layer_alpha = F.softmax(edge_layer_att, 1)
            edges = torch.sum(edge_layer_alpha * edges_cat, 1)

        return nodes, edges, edge_embs

    def forward(self, word_list, batch_inputs_f, batch_inputs_b, mask):

        nodes, edges, edge_embs = self.update_graph(word_list, batch_inputs_f, batch_inputs_b, mask)

        return nodes, edges, edge_embs
