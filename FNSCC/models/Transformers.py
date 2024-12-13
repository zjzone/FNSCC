"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from transformers import BertPreTrainedModel
# from transformers import AutoModel, AutoTokenizer

class SCCLBert(nn.Module):
    def __init__(self, bert_model, tokenizer, cluster_centers=None, alpha=1.0):
        super(SCCLBert, self).__init__()
        
        self.tokenizer = tokenizer
        self.bert = bert_model
        self.emb_size = self.bert.config.hidden_size #768
        self.alpha = alpha

        # Instance-CL head
        #It consists of a single-layer MLP, and its activation function uses ReLU. The input dimension is 768 and the output dimension is 128.
        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.emb_size))
        # Clustering head
        # The cluster center is automatically included in the parameter list when using the optimizer, and it will be automatically optimized and updated during the training process
        initial_cluster_centers = torch.tensor(
            cluster_centers, dtype=torch.float, requires_grad=True)
        self.cluster_centers = Parameter(initial_cluster_centers)

    
    def forward(self, input_ids, attention_mask, task_type="virtual"):
        if task_type == "evaluate":
            return self.get_mean_embeddings(input_ids, attention_mask)
        
        elif task_type == "virtual":
            input_ids_1, input_ids_2 = torch.unbind(input_ids, dim=1)
            attention_mask_1, attention_mask_2 = torch.unbind(attention_mask, dim=1) 
            
            mean_output_1 = self.get_mean_embeddings(input_ids_1, attention_mask_1)
            mean_output_2 = self.get_mean_embeddings(input_ids_2, attention_mask_2)
            return mean_output_1, mean_output_2
        
        elif task_type == "explicit":
            input_ids_1, input_ids_2, input_ids_3 = torch.unbind(input_ids, dim=1)  #input_ids
            attention_mask_1, attention_mask_2, attention_mask_3 = torch.unbind(attention_mask, dim=1) #attention_mask

            mean_output_1 = self.get_mean_embeddings(input_ids_1, attention_mask_1) #(32,768)
            mean_output_2 = self.get_mean_embeddings(input_ids_2, attention_mask_2)
            mean_output_3 = self.get_mean_embeddings(input_ids_3, attention_mask_3)
            return mean_output_1, mean_output_2, mean_output_3
        
        else:
            raise Exception("TRANSFORMER ENCODING TYPE ERROR! OPTIONS: [EVALUATE, VIRTUAL, EXPLICIT]")

    def get_mean_embeddings(self, input_ids, attention_mask):
        bert_output = self.bert.forward(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        mean_output = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return mean_output

    # Calculate the probability distribution of a given embedding vector embeddings belonging to each cluster center
    def get_cluster_prob(self, embeddings):
        norm_squared = torch.sum((embeddings.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True) #(batch,num_classes)

    def local_consistency(self, embd0, embd1, embd2, criterion):
        p0 = self.get_cluster_prob(embd0)
        p1 = self.get_cluster_prob(embd1)
        p2 = self.get_cluster_prob(embd2)
        
        lds1 = criterion(p1, p0)
        lds2 = criterion(p2, p0)
        return lds1+lds2
    
    def contrast_logits(self, embd1, embd2=None):
        feat1 = F.normalize(self.contrast_head(embd1), dim=1)
        if embd2 != None:
            feat2 = F.normalize(self.contrast_head(embd2), dim=1)
            return feat1, feat2
        else: 
            return feat1

    def compute_similarity(self, embeddings): # (batch, batch)
        similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
        return similarity_matrix

    def get_k_nearest_neighbors(self, embeddings, k): #(batch, k)
        similarity_matrix = self.compute_similarity(embeddings)
        _, neighbors_indices = torch.topk(similarity_matrix, k=k + 1)
        return neighbors_indices[:, 1:]  # Remove itself

    def update_cluster_prob_with_neighbors(self, embeddings, k, lambda_param):
        neighbors_indices = self.get_k_nearest_neighbors(embeddings, k)
        sim_matrix = self.compute_similarity(embeddings)

        qij = self.get_cluster_prob(embeddings)

        new_qij = torch.zeros_like(qij)

        for i in range(embeddings.size(0)):
            neighbor_indices = neighbors_indices[i]
            sim_sum = 0.0
            weighted_q_sum = 0.0
            density_sum = 0.0

            # Calculate the fuzzy membership weight and importance weight of neighbors
            for neighbor_idx in neighbor_indices:
                sim = sim_matrix[i, neighbor_idx]
                density = torch.exp(-torch.norm(embeddings[i] - embeddings[neighbor_idx]) ** 2)
                density_sum += density
                weighted_q_sum += sim * density * qij[neighbor_idx]
                sim_sum += sim * density

            adaptive_lambda = lambda_param * density_sum / (density_sum + 1e-8)
            #adaptive_lambda = lambda_param / (density_sum + 1e-8)

            numerator = qij[i] + adaptive_lambda * weighted_q_sum
            denominator = 1 + adaptive_lambda * sim_sum

            new_qij[i] = numerator / denominator

        new_qij = new_qij / torch.sum(new_qij, dim=1, keepdim=True)
        return new_qij
