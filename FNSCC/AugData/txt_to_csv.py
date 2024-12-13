import pandas as pd

txt_data=['AgNews.txt','GooglenewsS.txt','GooglenewsT.txt','GooglenewsTS.txt','SearchSnippets.txt',
      'Tweet.txt']
csv_data=['AgNews.csv','GooglenewsS.csv','GooglenewsT.csv','GooglenewsTS.csv','SearchSnippets.csv',
      'Tweet.csv']

import csv

for i in range(len(txt_data)):

    txt_file = './Dataset/'+txt_data[i]
    csv_file = './Dataset/'+csv_data[i]
    # Specify column names
    column_names = ['label', 'text']
    # Use tab as separator
    df = pd.read_csv(txt_file, sep='\t', header=None, names=column_names,index_col=False)
    df.to_csv(csv_file, index=False)
