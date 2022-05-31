import torch
import torch.nn as nn


def topk_with_distance(src, dis, k):
    """
    find for every data in src its k-nearest neighbor from dis with index and distance 
    params:
        src: (B, N, C) 
        dis: (B, M, C)
        k: numbers to select from dis
    return:
        dis_index_distance: (B, N, k, 2) 
    """
    ...
    dis_index_distance = ...

    return dis_index_distance


def nonconformity_measure(index_distance, label):
    """
    params:
        cali_index_distance: (B, N, k, 2) 
        label: (B, N, L)
    return:
        nonconformity: (B, N) 
    """
    ...
    nonconformity = ...
    
    return nonconformity


class kNN(nn.Module):
    """
    params:
        input_channel:
        output_channel:
    inputs:
        train_feature: (B, N, C)
        train_label: (B, N, L), one-hot code
    return:
        nonconformity: (B, N)
    """
    def __init__(self, input_channel, output_channel, k):
        super(kNN, self).__init__()
        self.k = k
    
    def forward(self, train_feature, train_label):
        index_distance = topk_with_distance(train_feature, train_feature, self.k) # (B, N, k, 2)
        nonconformity = nonconformity_measure(index_distance, train_label) # (B, N)
        return nonconformity







