# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 13:33:26 2023

@author: admin
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Neural network
class LipNet(torch.nn.Module):
    def __init__(
        self,
        dropout_p,
        conv_sizes,
        conv_kernel_sizes,
        gru_size,
        act_fn_maxpool,
        act_fn_dense,
        img_T,
        img_H,
        img_W,
        vocabsize,
    ):
        super().__init__()
        self.dropout_p  = dropout_p
        self.conv_sizes = conv_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.gru_size = gru_size
        self.act_fn_maxpool = act_fn_maxpool
        self.act_fn_dense = act_fn_dense
        self.img_T =img_T
        self.img_H =img_H
        self.img_W = img_W
        self.vocabsize = vocabsize

        self.cnnetwork = torch.nn.ModuleList()
        for idx in range(len(self.conv_sizes)-1):
            self.cnnetwork.append(
                nn.Conv3d(self.conv_sizes[idx], self.conv_sizes[idx+1], (1,self.conv_kernel_sizes[idx],self.conv_kernel_sizes[idx]))
            )

            self.cnnetwork.append(
                nn.Dropout3d(self.dropout_p)
            )


        self.gru1 = nn.GRU(self.calc_inputs(),self.gru_size,batch_first=True)
        self.dense = nn.Linear(self.gru_size*img_T,self.vocabsize)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)

    def calc_inputs(self):
        for idx in range(len(self.conv_sizes)-1):
            self.img_H = self.img_H - self.conv_kernel_sizes[idx] + 1
            self.img_W = self.img_W - self.conv_kernel_sizes[idx] + 1
        return int((self.conv_sizes[-1])*self.img_H*self.img_W)

    def forward(self, x):

        for layer in self.cnnetwork:
            if isinstance(layer, torch.nn.Dropout3d):
                x = self.act_fn_maxpool(layer(x))
            else:
                x = layer(x)

        # (B, C, T, H, W)->(B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # (B, T, C, H, W)->(B, T, C*H*W)
        x = x.view(x.size(0), x.size(1), -1)

        self.gru1.flatten_parameters()
        x, h = self.gru1(x)
        x = self.dropout(x)

        x = x.reshape(x.size(0), -1)
        x = self.dense(x)
        x = F.softmax(x, dim=-1)
        return x