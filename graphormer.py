# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.



import torch
import torch.nn as nn

from multihead_attention import MultiheadAttention
import networkx as nx
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import GCNConv,global_mean_pool,global_add_pool,PointTransformerConv
import numpy as np
import torch.nn.functional as F

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


class GraphormerLayer(nn.Module):
    def __init__(self, embedding_dim, num_attention_heads, ffn_embedding_dim, dropout):
        super(GraphormerLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.ffn_embedding_dim = ffn_embedding_dim
        
        self.dropout_module = nn.Dropout(p=dropout)
        self.activation_dropout_module = nn.Dropout(p=dropout)
        
        self.activation_fn = nn.ReLU()
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)
        
        self.fc1 = nn.Linear(self.embedding_dim, self.ffn_embedding_dim)
        self.fc2 = nn.Linear(self.ffn_embedding_dim, self.embedding_dim)
        
        self.self_attn = MultiheadAttention(embedding_dim, num_attention_heads, dropout=dropout)

    def forward(self, x, self_attn_bias = None):
        residual = x
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_bias=self_attn_bias,
            need_weights=False,
        )
        
        x = self.dropout_module(x)
        x = residual + x

        x = self.self_attn_layer_norm(x)

        residual = x

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = residual + x

        x = self.final_layer_norm(x)
        return x

class Graphormer_d(nn.Module):
    def __init__(self, args, output_channels=40, embedding_dim = 64, num_attention_heads = 8, num_degree = 20, num_encoder_layers = 4, ffn_embedding_dim = 128):
        super(Graphormer_d, self).__init__()
        self.args = args
        self.k = args.k
        
        self.layers = nn.ModuleList([])
        self.layers.extend([GraphormerLayer(embedding_dim, num_attention_heads, ffn_embedding_dim, args.dropout)
                for _ in range(num_encoder_layers)])
        self.GCN_layers = nn.ModuleList([])
        self.GCN_layers.extend([ GCNConv(embedding_dim, embedding_dim)
                for _ in range(num_encoder_layers)])
        self.Bn_layers = nn.ModuleList([])
        self.Bn_layers.extend([ nn.BatchNorm1d(embedding_dim)
                for _ in range(num_encoder_layers)])
        
        self.edge_dis_encoder = nn.Embedding(256, num_attention_heads)
        self.fc_in = GCNConv(3,embedding_dim)
        self.fc_out = nn.Linear(embedding_dim,40)
        
        self.bn1 = nn.BatchNorm1d(64)        
        self.conv1 = GCNConv(3, 64)
        
        self.linear1 = nn.Linear(128, 128, bias=False)
        self.bn6 = nn.BatchNorm1d(128)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(128, 128)
        self.bn7 = nn.BatchNorm1d(128)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(128, output_channels)
    
    def get_graph(self,x):
        batch_size = self.batch_size
        num_points = self.num_points
        
        x = x.reshape(batch_size, num_points, -1)
        x = x.transpose(2,1)
        x = x.view(batch_size, -1, num_points)
        
        idx = knn(x, k=self.k)   # (batch_size, num_points, k)
        device = torch.device('cuda')
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
        idx = idx + idx_base
        
        base = torch.arange(0, batch_size*num_points, device=device).unsqueeze(-1).repeat(1,self.k)
        base = base.view(-1)
        
        idx = idx.view(-1)
        edge_index = torch.stack((base,idx))
        
        x = x.transpose(2, 1).view(batch_size*num_points,-1)
        return x,edge_index    
    
    def get_attn_bias(self,edge_index,batch):
        adj = to_dense_adj(edge_index, batch)
        dist_matrices = []
        for i in range(adj.shape[0]):
            adj_matrix = adj[i].cpu()
            adj_matrix = adj_matrix.numpy()
            G = nx.from_numpy_matrix(adj_matrix)
            shortest_path_distance = nx.floyd_warshall_numpy(G)
            shortest_path_distance[shortest_path_distance < 0] = -1
            shortest_path_distance[shortest_path_distance > 255] = -1
            #print(shortest_path_distance)
            dist_matrices.append(self.edge_dis_encoder.weight[shortest_path_distance])
        dist_matrices = torch.stack(dist_matrices)
        return dist_matrices
    
    def forward(self, x):
        device = torch.device('cuda')        
        self.batch_size = x.shape[0]
        self.num_points = x.shape[2]
        batch = torch.arange(0, self.batch_size, device=device).unsqueeze(-1).repeat(1,self.num_points).view(-1)
        
        x = x.transpose(2, 1)
        
        x,edge_index = self.get_graph(x)
        attn_bias = self.get_attn_bias(edge_index,batch)
        
        x = self.conv1(x,edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x,negative_slope=0.2)
        
        for i in range(len(self.layers)):
            x = self.GCN_layers[i](x,edge_index)
            x = self.Bn_layers[i](x)
            x = F.leaky_relu(x,negative_slope=0.2)
            
            x0 = x.clone().view(self.batch_size, self.num_points, -1).transpose(0,1)
            x0 = self.layers[i](x0,self_attn_bias = attn_bias)
            x0 = x0.transpose(0,1).reshape(self.batch_size*self.num_points, -1)
            x = x + x0

        x1 = global_mean_pool(x, batch)
        x2 = global_add_pool(x, batch)
        
        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        
        return x
    
class Graphormer_r(nn.Module):
    def __init__(self, args, output_channels=40, embedding_dim = 64, num_attention_heads = 8, num_degree = 20, num_encoder_layers = 4, ffn_embedding_dim = 128):
        super(Graphormer_r, self).__init__()
        self.args = args
        self.k = args.k

        
        self.layers = nn.ModuleList([])
        self.layers.extend([GraphormerLayer(embedding_dim, num_attention_heads, ffn_embedding_dim, args.dropout)
                for _ in range(num_encoder_layers)])
        self.GCN_layers = nn.ModuleList([])
        self.GCN_layers.extend([ GCNConv(embedding_dim, embedding_dim)
                for _ in range(num_encoder_layers)])
        self.Bn_layers = nn.ModuleList([])
        self.Bn_layers.extend([ nn.BatchNorm1d(embedding_dim)
                for _ in range(num_encoder_layers)])
        
        self.edge_dis_encoder = nn.Embedding(256, num_attention_heads)
        self.fc_in = GCNConv(3,embedding_dim)
        self.fc_out = nn.Linear(embedding_dim,40)
        
        self.bn1 = nn.BatchNorm1d(64)        
        self.conv1 = GCNConv(3, 64)
        
        self.bn5 = nn.BatchNorm1d(512)  
        self.conv5 = GCNConv(320, 512)
        
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)
    
    def get_graph(self,x):
        batch_size = self.batch_size
        num_points = self.num_points
        
        x = x.reshape(batch_size, num_points, -1)
        x = x.transpose(2,1)
        x = x.view(batch_size, -1, num_points)
        
        idx = knn(x, k=self.k)   # (batch_size, num_points, k)
        device = torch.device('cuda')
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
        idx = idx + idx_base
        
        base = torch.arange(0, batch_size*num_points, device=device).unsqueeze(-1).repeat(1,self.k)
        base = base.view(-1)
        
        idx = idx.view(-1)
        edge_index = torch.stack((base,idx))
        
        x = x.transpose(2, 1).view(batch_size*num_points,-1)
        return x,edge_index    
    
    def get_attn_bias(self,edge_index,batch):
        adj = to_dense_adj(edge_index, batch)
        dist_matrices = []
        for i in range(adj.shape[0]):
            adj_matrix = adj[i].cpu()
            adj_matrix = adj_matrix.numpy()
            G = nx.from_numpy_matrix(adj_matrix)
            shortest_path_distance = nx.floyd_warshall_numpy(G)
            shortest_path_distance[shortest_path_distance < 0] = -1
            shortest_path_distance[shortest_path_distance > 255] = -1
            #print(shortest_path_distance)
            dist_matrices.append(self.edge_dis_encoder.weight[shortest_path_distance])
        dist_matrices = torch.stack(dist_matrices)
        return dist_matrices
    
    def forward(self, x):
        device = torch.device('cuda')        
        self.batch_size = x.shape[0]
        self.num_points = x.shape[2]
        batch = torch.arange(0, self.batch_size, device=device).unsqueeze(-1).repeat(1,self.num_points).view(-1)
        
        x = x.transpose(2, 1)
        
        x,edge_index = self.get_graph(x)
        attn_bias = self.get_attn_bias(edge_index,batch)
        
        data_out = []
        
        x = self.conv1(x,edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x,negative_slope=0.2)
        data_out.append(x)
        
        for i in range(len(self.layers)):
            x = self.GCN_layers[i](x,edge_index)
            x = self.Bn_layers[i](x)
            x = F.leaky_relu(x,negative_slope=0.2)
            
            x0 = x.clone().view(self.batch_size, self.num_points, -1).transpose(0,1)
            x0 = self.layers[i](x0,self_attn_bias = attn_bias)
            x0 = x0.transpose(0,1).reshape(self.batch_size*self.num_points, -1)
            x = x + x0
            data_out.append(x)
            
        x = torch.stack(data_out,dim = 2)
        x = x.reshape(x.shape[0],-1)
        
        x = self.conv5(x,edge_index)
        x = self.bn5(x)
        x = F.leaky_relu(x,negative_slope=0.2)

        x1 = global_mean_pool(x, batch)
        x2 = global_add_pool(x, batch)
        
        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        
        return x