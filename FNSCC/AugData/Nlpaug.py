import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import csv

def data_augmentation(sentences, aug_p=0.3):
    # Contextual Augmenter
    augmenter1 = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute",aug_min=1, aug_p=aug_p, device='cuda')
    augmenter2 = naw.ContextualWordEmbsAug(model_path='roberta-base', action="substitute", aug_min=1, aug_p=aug_p, device='cuda')

    augmented_sentences_1 = []
    augmented_sentences_2 = []
    for sentence in sentences:
        if isinstance(sentence, str):

            aug_sentence1 = augmenter1.augment(sentence)
            aug_sentence2 = augmenter2.augment(sentence)

            augmented_sentences_1.extend(aug_sentence1)
            augmented_sentences_2.extend(aug_sentence2)
        else:
            raise ValueError("Input sentences must be a list of strings")


    return augmented_sentences_1,augmented_sentences_2

def read_csv_column_as_list(file_path, column_number):
    column_data = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)

        for row in reader:
            if len(row) > column_number and row[column_number]!='label':
                column_data.append(row[column_number])
    return column_data

def remap_labels_to_range(labels):
    unique_labels = np.unique(labels)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    remapped_labels = np.array([label_mapping[label] for label in labels])

    return remapped_labels, label_mapping

if __name__ == '__main__':
    csv_data = ['AgNews', 'GooglenewsS', 'GooglenewsT',
                'GooglenewsTS','SearchSnippets','Tweet']

    for i in csv_data:
        file='./Dataset/source/'+i+'.csv'
        data = pd.read_csv(file)
        column_number = 1
        column_data = read_csv_column_as_list(file, column_number)
        aug_data1, aug_data2 = data_augmentation(column_data[1:])
        data['text1'] = aug_data1
        data['text2'] = aug_data2

        data.to_csv('./Dataset/aug_20_bert+robert/'+i+'.csv', index=False)
        column_number = 0
        column_data = read_csv_column_as_list('./Dataset/aug_20_bert+robert/'+i+'.csv', column_number)
        new_labels, mapping = remap_labels_to_range(column_data)
        data["label"] = new_labels

        data.to_csv('./Dataset/aug_20_bert+robert/'+i+'.csv', index=False)
        print('--------'+file+'Completed--------')