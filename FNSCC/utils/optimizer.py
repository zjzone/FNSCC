"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import os
import torch 
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer

BERT_CLASS = {
    "distilbert": 'distilbert-base-uncased', 
}

SBERT_CLASS = {
    "distilbert": 'distilbert-base-nli-stsb-mean-tokens',
}

def get_optimizer(model, args):
    #BERT 的参数使用基础学习率，而对比学习头和簇中心使用放大的学习率，通常是为了加快这些模块的训练速度。
    optimizer = torch.optim.Adam([ #采用Adam优化器
        {'params':model.bert.parameters()},  #bert模型主干，学习率为5e-6
        {'params':model.contrast_head.parameters(), 'lr': args.lr*args.lr_scale}, #contrast_head使用不同的学习率
        {'params':model.cluster_centers, 'lr': args.lr*args.lr_scale}#cluster_centers使用和contrast_head一样的学习率
    ], lr=args.lr)
    
    print(optimizer)
    return optimizer 
    

def get_bert(args):
    
    if args.use_pretrain == "SBERT":
        bert_model = get_sbert(args)
        tokenizer = bert_model[0].tokenizer
        model = bert_model[0].auto_model
        print("..... loading Sentence-BERT !!!")
    else:
        config = AutoConfig.from_pretrained(BERT_CLASS[args.bert])
        model = AutoModel.from_pretrained(BERT_CLASS[args.bert], config=config)
        tokenizer = AutoTokenizer.from_pretrained(BERT_CLASS[args.bert])
        print("..... loading plain BERT !!!")

    return model, tokenizer


def get_sbert(args):
    sbert = SentenceTransformer(SBERT_CLASS[args.bert])
    return sbert

# "../huggingface/hub/models--distilbert-base-uncased/snapshots/disbert"






