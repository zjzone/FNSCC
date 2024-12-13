import pandas as pd
import csv
import numpy as np

def remap_labels_to_range(labels):
    unique_labels = np.unique(labels)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    remapped_labels = np.array([label_mapping[label] for label in labels])

    return remapped_labels, label_mapping

def read_csv_column_as_list(file_path, column_number):
    column_data = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)

        for row in reader:
            if len(row) > column_number and row[column_number]!='label':
                column_data.append(int(row[column_number]))
    return column_data

column_number = 0
column_data = read_csv_column_as_list("AgNews.csv", column_number)

new_labels, mapping = remap_labels_to_range(column_data)
print(new_labels)

data=pd.read_csv("AgNews.csv")
data["label"]=new_labels
data.to_csv('AgNews.csv', index=False)