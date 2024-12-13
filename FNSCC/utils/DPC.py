import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from utils.metric import Confusion
from sklearn.cluster import KMeans


def get_mean_embeddings(bert, input_ids, attention_mask):

    bert_output = bert.forward(input_ids=input_ids, attention_mask=attention_mask)
    device = bert_output[0].device
    attention_mask = attention_mask.to(device).unsqueeze(-1)  # (batch,max_length,1)
    mean_output = torch.sum(bert_output[0] * attention_mask, dim=1) / torch.sum(attention_mask,
                                                                                dim=1)
    return mean_output  # tensor(batch,768)


def get_batch_token(tokenizer, text, max_length):
    token_feat = tokenizer.batch_encode_plus(
        text,
        max_length=max_length,
        return_tensors='pt',
        padding='max_length',
        truncation=True
    )
    return token_feat



def acc_cal(y_true, y_pred):  # acc
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def getDistanceMatrix(datas):
    N,D=np.shape(datas)
    dists=np.zeros([N,N])

    for i in range(N):
        for j in range(N):
            vi=datas[i,:]
            vj=datas[j,:]
            dists[i,j]=np.sqrt(np.dot((vi-vj),(vi-vj)))
    return dists

def select_dc(dists):
    # N=np.shape(dists)[0]
    # tt=np.reshape(dists,N*N)
    # percent=2.0
    # position=int(N*(N-Acc_Test)*percent/100)
    # dc=np.sort(tt)[position+N]

    N=np.shape(dists)[0]
    max_dis=np.max(dists)
    min_dis=np.min(dists)
    dc=(max_dis+min_dis)/2

    while True:
        n_neighs=np.where(dists<dc)[0].shape[0]-N
        rate=n_neighs/(N*(N-1))

        if rate>=0.01 and rate<=0.02:
            break
        if rate<0.01:
            min_dis=dc
        else:
            max_dis=dc

        dc=(max_dis+min_dis)/2
        if max_dis-min_dis<0.0001:
            break
    return dc

def get_density(dists,dc,method=None):
    N=np.shape(dists)[0]
    rho=np.zeros(N)

    for i in range(N):
        if method==None:
            rho[i]=np.where(dists[i,:]<dc)[0].shape[0]-1
        else:
            rho[i]=np.sum(np.exp(-(dists[i,:]/dc)**2))-1
    return rho

def get_deltas(dists,rho):
    N=np.shape(dists)[0]
    deltas=np.zeros(N)
    nearest_neiber=np.zeros(N)

    index_rho=np.argsort(-rho)

    for i,index in enumerate(index_rho):

        if i==0:
            continue

        index_higher_rho=index_rho[:i]

        deltas[index]=np.min(dists[index,index_higher_rho])


        index_nn=np.argmin(dists[index,index_higher_rho])
        nearest_neiber[index]=index_higher_rho[index_nn].astype(int)

    deltas[index_rho[0]]=np.max(deltas)
    return deltas,nearest_neiber

def find_centers_K(rho,deltas,K):
    rho_delta=rho*deltas
    centers=np.argsort(-rho_delta)
    return centers[:K]

def get_kmeans_centers(bert, tokenizer, train_loader, num_classes, max_length):
    for i, batch in enumerate(train_loader):

        text, label = batch['text'], batch['label']
        tokenized_features = get_batch_token(tokenizer, text, max_length)
        corpus_embeddings = get_mean_embeddings(bert, **tokenized_features)

        if i == 0:
            all_labels = label

            all_embeddings = corpus_embeddings.detach().to("cpu").numpy()
        else:
            all_labels = torch.cat((all_labels, label), dim=0)
            all_embeddings = np.concatenate((all_embeddings, corpus_embeddings.detach().to("cpu").numpy()),
                                            axis=0)

    dists = getDistanceMatrix(all_embeddings)
    dc = select_dc(dists)
    rho = get_density(dists, dc, method="gaussain")
    deltas, nearest_neiber = get_deltas(dists, rho)
    centers = find_centers_K(rho, deltas, num_classes)
    print(all_embeddings[centers].shape)
    return np.array(all_embeddings[centers])