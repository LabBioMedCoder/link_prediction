#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : WY Liang
# @Project  : pytorch_study

# https://docs.dgl.ai/tutorials/blitz/4_link_predict.html

import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
from dgl.nn import SAGEConv
import dgl.function as fn
from sklearn.metrics import roc_auc_score


dataset = dgl.data.CoraGraphDataset()
g = dataset[0]
print(type(g))
print(g)
# Split edge set for training and testing
u, v = g.edges()    # u:头节点，v：尾节点
print(u)
print(v)
print(g.num_edges())
eids = np.arange(g.num_edges())     # 边的数量，为每条边生成一个id
eids = np.random.permutation(eids)  # 随机打乱
test_size = int(len(eids) * 0.1)    # 测试集的数量，0.1的边作为测试集
test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]] # 测试集中，每条边的头节点和尾节点
train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]] # 训练集中，每条边的头节点和尾节点
print("训练集的数量：",train_pos_u.shape)

# Find all negative edges and split them for training and testing
# 负样本边（负采样），并将负样本氛围测试集和训练集
# np.ones(len(u))  [1. 1. 1. ... 1. 1. 1.]
# sp.coo_matrix:生成矩阵，用指定数据生成矩阵，np.ones(len(u))为数据，u.numpy(), v.numpy()为数据中每个元素的位置
adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
# 生成负样本邻接矩阵
adj_neg = 1 - adj.todense() - np.eye(g.num_nodes())
neg_u, neg_v = np.where(adj_neg != 0)   # f负样本中，每条边的头节点和尾节点
print(neg_u)
print(neg_v)

neg_eids = np.random.choice(len(neg_u), g.num_edges())  # 选择与正样本数量相对的边
test_neg_u, test_neg_v = (neg_u[neg_eids[:test_size]],neg_v[neg_eids[:test_size]],) # 负样本测试集中，每条边的头节点和尾节点
train_neg_u, train_neg_v = (neg_u[neg_eids[test_size:]],neg_v[neg_eids[test_size:]],) # 负样本训练集中，每条边的头节点和尾节点

# 训练时，移除那些作为测试集的边
train_g = dgl.remove_edges(g, eids[:test_size])





# ----------- 2. create model -------------- #
# build a two-layer GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, 128, "mean")
        self.conv2 = SAGEConv(128, h_feats, "mean")

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h    # 给图添加属性矩阵
            # Compute a new edge feature named 'score' by a dot-product between the  source node feature 'h' and destination node feature 'h'.
            # apply_edges：Update the features of the specified edges by the provided function.
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata["score"][:, 0]


train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.num_nodes())
train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.num_nodes())

test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.num_nodes())
test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.num_nodes())

model = GraphSAGE(train_g.ndata["feat"].shape[1], 16)
# You can replace DotPredictor with MLPPredictor.
# pred = MLPPredictor(16)
pred = DotPredictor()


def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    )
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).numpy()
    return roc_auc_score(labels, scores)


# ----------- 3. set up loss and optimizer -------------- #
# in this case, loss will in training loop
optimizer = torch.optim.Adam(
    itertools.chain(model.parameters(), pred.parameters()), lr=0.01
)

# ----------- 4. training -------------------------------- #
all_logits = []
for e in range(100):
    # forward
    h = model(train_g, train_g.ndata["feat"])
    print(h.shape)
    pos_score = pred(train_pos_g, h)
    print(pos_score.shape)
    neg_score = pred(train_neg_g, h)
    loss = compute_loss(pos_score, neg_score)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e % 5 == 0:
        print("In epoch {}, loss: {}".format(e, loss))



with torch.no_grad():
    pos_score = pred(test_pos_g, h)
    neg_score = pred(test_neg_g, h)
    print("AUC", compute_auc(pos_score, neg_score))




