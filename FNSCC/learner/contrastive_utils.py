from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

class PairConLossWithNeighbors(nn.Module):
    def __init__(self, temperature=0.05, k=5):
        super(PairConLossWithNeighbors, self).__init__()
        self.temperature = temperature
        self.k = k
        self.eps = 1e-08
        print(f"\n Initializing PairConLossWithNeighbors \n")

    def compute_similarity(self, features):
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
        return similarity_matrix

    def get_k_nearest_neighbors(self, similarity_matrix):
        _, neighbors_indices = torch.topk(similarity_matrix, k=self.k + 1, dim=-1)
        return neighbors_indices[:, 1:]

    def forward(self, features_1, features_2):
        device = features_1.device
        batch_size = features_1.shape[0]

        features = torch.cat([features_1, features_2], dim=0)
        similarity_matrix = self.compute_similarity(features)
        neighbors_indices = self.get_k_nearest_neighbors(similarity_matrix)

        pos = torch.exp(torch.sum(features_1 * features_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)  # #(2*batch,1)

        neg = torch.exp(similarity_matrix / self.temperature) #(2*batch,2*batch)
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(device)
        neg = neg.masked_fill(mask, 0)  # (2*batch,2*batch)


        # for i in range(2 * batch_size):
        #     neg[i, neighbors_indices[i]] = 0

        batch_indices = torch.arange(2 * batch_size).unsqueeze(1).to(device)  # (2*batch, 1)
        neg[batch_indices, neighbors_indices] = 0

        Ng = neg.sum(dim=-1)

        loss_pos = (- torch.log(pos / (Ng + pos))).mean()

        return {"loss": loss_pos}
