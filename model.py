#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,global_mean_pool,global_add_pool,PointTransformerConv


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x
    
class GCN_d(nn.Module):
    def __init__(self, args, output_channels=40):
        super(GCN_d, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1024)
        
        self.conv1 = GCNConv(3, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 256)
        self.conv4 = GCNConv(256, 512)
        self.conv5 = GCNConv(512, 1024)
        
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
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
        
    def forward(self, x):
        self.batch_size = x.shape[0]
        device = torch.device('cuda')
        self.num_points = x.shape[2]
        batch = torch.arange(0, self.batch_size, device=device).unsqueeze(-1).repeat(1,self.num_points).view(-1)
        x = x.transpose(2, 1)
        
        x,edge_index = self.get_graph(x)
        x = self.conv1(x,edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x,negative_slope=0.2)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x,edge_index)
        x = self.bn2(x)
        x = F.leaky_relu(x,negative_slope=0.2)
        x = F.dropout(x, training=self.training)
        
        x = self.conv3(x,edge_index)
        x = self.bn3(x)
        x = F.leaky_relu(x,negative_slope=0.2)
        x = F.dropout(x, training=self.training)
        
        x = self.conv4(x,edge_index)
        x = self.bn4(x)
        x = F.leaky_relu(x,negative_slope=0.2)
        x = F.dropout(x, training=self.training)
        
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
    
class GCN_s(nn.Module):
    def __init__(self, args, output_channels=40):
        super(GCN_s, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(1024)
        self.conv1 = GCNConv(3, 128)
        self.conv2 = GCNConv(128, 1024)
        
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
        
    def forward(self, x):
        self.batch_size = x.shape[0]
        device = torch.device('cuda')
        self.num_points = x.shape[2]
        batch = torch.arange(0, self.batch_size, device=device).unsqueeze(-1).repeat(1,self.num_points).view(-1)
        x = x.transpose(2, 1)
        
        x,edge_index = self.get_graph(x)
        x = self.conv1(x,edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x,negative_slope=0.2)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x,edge_index)
        x = self.bn2(x)
        x = F.leaky_relu(x,negative_slope=0.2)
        x1 = global_mean_pool(x, batch)
        x2 = global_add_pool(x, batch)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        
        return x
    
class GCN_r(nn.Module):
    def __init__(self, args, output_channels=40):
        super(GCN_r, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(1024)
        
        self.conv1 = GCNConv(3, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 128)
        self.conv4 = GCNConv(128, 256)
        self.conv5 = GCNConv(512, 1024)
        
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
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
        
    def forward(self, x):
        self.batch_size = x.shape[0]
        device = torch.device('cuda')
        self.num_points = x.shape[2]
        batch = torch.arange(0, self.batch_size, device=device).unsqueeze(-1).repeat(1,self.num_points).view(-1)
        x = x.transpose(2, 1)
        
        x,edge_index = self.get_graph(x)
        x = self.conv1(x,edge_index)
        x = self.bn1(x)
        x1 = F.leaky_relu(x,negative_slope=0.2)

        x = self.conv2(x1,edge_index)
        x = self.bn2(x)
        x2 = F.leaky_relu(x,negative_slope=0.2)
        
        x = self.conv3(x2,edge_index)
        x = self.bn3(x)
        x3 = F.leaky_relu(x,negative_slope=0.2)
        
        x = self.conv4(x3,edge_index)
        x = self.bn4(x)
        x4 = F.leaky_relu(x,negative_slope=0.2)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x,edge_index)
        x = self.bn5(x)
        x = F.leaky_relu(x,negative_slope=0.2)
        
        xa = global_mean_pool(x, batch)
        xb = global_add_pool(x, batch)
        
        x = torch.cat((xa, xb), 1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        print(x.shape)
        return x
    
    
class PointTransformer(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointTransformer, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        
        self.conv1 = PointTransformerConv(3, 64)
        self.conv2 = PointTransformerConv(64, 64)
        self.conv3 = PointTransformerConv(64, 128)
        self.conv4 = PointTransformerConv(256, 256)
        
        self.linear1 = nn.Linear(512, 512, bias=False)
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
        
    def forward(self, x):
        self.batch_size = x.shape[0]
        device = torch.device('cuda')
        self.num_points = x.shape[2]
        batch = torch.arange(0, self.batch_size, device=device).unsqueeze(-1).repeat(1,self.num_points).view(-1)
        x = x.transpose(2, 1)
        pos = x.clone().detach().view(self.batch_size*self.num_points,-1)

        x,edge_index = self.get_graph(x)
        x = self.conv1(x,pos,edge_index)
        x = self.bn1(x)
        x1 = F.leaky_relu(x,negative_slope=0.2)

        x = self.conv2(x1,pos,edge_index)
        x = self.bn2(x)
        x2 = F.leaky_relu(x,negative_slope=0.2)
        
        x = self.conv3(x2,pos,edge_index)
        x = self.bn3(x)
        x3 = F.leaky_relu(x,negative_slope=0.2)
        
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv4(x,pos,edge_index)
        x = self.bn4(x)
        x = F.leaky_relu(x,negative_slope=0.2)
        
        xa = global_mean_pool(x, batch)
        xb = global_add_pool(x, batch)
        
        x = torch.cat((xa, xb), 1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        
        return x