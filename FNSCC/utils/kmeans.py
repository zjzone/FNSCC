"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from utils.metric import Confusion
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from numpy import where

def get_mean_embeddings(bert, input_ids, attention_mask):
        #(batch,max_length,768)
        input_ids = input_ids.to('cuda')
        attention_mask = attention_mask.to('cuda')
        bert_output = bert.forward(input_ids=input_ids, attention_mask=attention_mask)
        device = bert_output[0].device
        attention_mask = attention_mask.to(device).unsqueeze(-1) #(batch,max_length,1)
        mean_output = torch.sum(bert_output[0] * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return mean_output #tensor(batch,768)

    
def get_batch_token(tokenizer, text, max_length):
    token_feat = tokenizer.batch_encode_plus(
        text, 
        max_length=max_length, 
        return_tensors='pt', 
        padding='max_length', 
        truncation=True
    )
    return token_feat

def acc_cal(y_true, y_pred):  #acc
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
def get_kmeans_centers(bert, tokenizer, train_loader, num_classes, max_length):
    for i, batch in enumerate(train_loader):

        text, label = batch['text'], batch['label']
        tokenized_features = get_batch_token(tokenizer, text, max_length)
        corpus_embeddings = get_mean_embeddings(bert, **tokenized_features) #tensor(batch,768)
        
        if i == 0:
            all_labels = label

            all_embeddings = corpus_embeddings.detach().to("cpu").numpy()
        else:
            all_labels = torch.cat((all_labels, label), dim=0)
            all_embeddings = np.concatenate((all_embeddings, corpus_embeddings.detach().to("cpu").numpy()), axis=0)

    # Perform KMeans clustering
    confusion = Confusion(num_classes)
    clustering_model = KMeans(n_clusters=num_classes)
    clustering_model.fit(all_embeddings)
    cluster_assignment = clustering_model.labels_
    true_labels = all_labels
    pred_labels = torch.tensor(cluster_assignment)

    # 可视化
    # label = pred_labels.numpy()
    # if True:
    #     tsne = TSNE(n_components=2, metric="euclidean", random_state=0)  # Tsne可视化降维到2维
    #     X = tsne.fit_transform(all_embeddings)
    #     for cluster_label in np.unique(label):
    #         # 获取此群集的示例的行索引
    #         row_ix = where(label == cluster_label)
    #         # 创建这些样本的散布
    #         if cluster_label == 1:
    #             c = "b"
    #         elif cluster_label == 2:
    #             c = "g"
    #         elif cluster_label == 3:
    #             c = "y"
            # elif cluster_label == 4:
            #     c = "c"
            # elif cluster_label == 5:
            #     c = "m"  # 紫
            # elif cluster_label == 6:
            #     c = "r"  # 紫
            # elif cluster_label == 7:
            #     c = "k"  # 紫
            # else:
            #     c = "#FF5733"
            # plt.scatter(X[row_ix, 0], X[row_ix, 1], s=2, c=c)

        #plt.title("t-SNE Clustering Visualization")
        # plt.savefig("origin_tsne_visualization.pdf")
        # plt.close()

    print("all_embeddings:{}, true_labels:{}, pred_labels:{}".format(all_embeddings.shape, len(true_labels), len(pred_labels)))
    print(type(pred_labels),type(true_labels))
    print(len(pred_labels), len(true_labels))
    confusion.add(pred_labels, true_labels)
    confusion.optimal_assignment(num_classes)
    print("Iterations:{}, Clustering ACC:{:.3f}, centers:{}".format(clustering_model.n_iter_, confusion.acc(),clustering_model.cluster_centers_.shape))
    print('[Representation] Clustering scores:',confusion.clusterscores())
    return clustering_model.cluster_centers_



