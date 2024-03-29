#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : WY Liang
# @Project  : pytorch_study

# https://zhuanlan.zhihu.com/p/383696918
# https://www.cnblogs.com/mingye7/p/14995667.html


import networkx as nx
from deepsnap.graph import Graph
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges

G = nx.karate_club_graph()
print(type(G))
data = Graph(G)  # 将 networkx中的graph对象转化为torch_geometric的Data对象
data.num_features = 3
data.edge_attr = None

# 构造节点特征矩阵（原网络不存在节点特征）
data.x = torch.ones((data.num_nodes, data.num_features), dtype=torch.float32)
print(data.x)

# 该函数将自动地采样得到负样本，并将正负样本分成训练集、验证集和测试集三个集合
data = train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1)
print(data)
print(data.edge_label_index)
# print(data.test_neg_edge_index)
# print(data.test_pos_edge_index)
print(data.train_pos_edge_index)  # 只是用生成的训练集中的正样本，训练集中的负样本的话在每个epoch中进行采样
# print(data.val_neg_edge_index)
# print(data.val_pos_edge_index)

# 构造一个简单的图卷积神经网络（两层），包含编码（节点嵌入）、解码（分数预测）等操作
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.num_features, 128)
        self.conv2 = GCNConv(128, 64)

    def encode(self):
        x = self.conv1(data.x, data.train_pos_edge_index)
        x = x.relu()
        x = self.conv2(x, data.train_pos_edge_index)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)  # 将正样本与负样本拼接 shape:[2,272]
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


# 将模型和数据送入设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
# 指定优化器
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

# 将训练集中的正边标签设置为1，负边标签设置为0
def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

# 训练函数，每次训练重新采样负边，计算模型损失，反向传播误差，更新模型参数
def train():
    model.train()
    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1),  # 负采样数量与正样本数量相同正样本
        force_undirected=True,
    )  # 得到负采样shape: [2,136]
    neg_edge_index = neg_edge_index.to(device)
    optimizer.zero_grad()
    z = model.encode()  # 利用正样本训练学习得到每个节点的特征 shape:[34, 64]
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)  # [272] 利用正样本和负样本 按位相乘 求和  (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
    print(link_logits.shape)
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index)  # [272] 前136个是1，后136个是0
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)  # binary_cross_entropy_with_logits会自动计算link_logits的sigmoid
    loss.backward()
    optimizer.step()
    return loss

# 测试函数，评估模型在验证集和测试集上的预测准确率
@torch.no_grad()
def test():
    model.eval()
    perfs = []
    for prefix in ["val", "test"]:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        z = model.encode()
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)
        link_probs = link_logits.sigmoid()
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        perfs.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))
    return perfs

# 训练模型，每次训练完，输出模型在验证集和测试集上的预测准确率
best_val_perf = test_perf = 0
for epoch in range(1, 11):
    train_loss = train()
    val_perf, tmp_test_perf = test()
    if val_perf > best_val_perf:
        best_val_perf = val_perf
        test_perf = tmp_test_perf
    log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_loss, best_val_perf, test_perf))

# 利用训练好的模型计算网络中剩余所有边的分数
z = model.encode()
final_edge_index = model.decode_all(z)
