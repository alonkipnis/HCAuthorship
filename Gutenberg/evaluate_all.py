import pandas as pd
import numpy as np
import os
import nltk
from sklearn.model_selection import KFold
import sys
from evaluate_classifier import evaluate_classifier

data_cloud_path = '/scratch/users/kipnisal/Data/Gutenberg'
data_local_path = '/Users/kipnisal/Data/Gutenberg/Data'

cloud_lib_path = '/scratch/users/kipnisal/Lib/AuthorshipAttribution'
local_lib_path = '../Authorship'

cloud_vocab_file = '/Users/kipnisal/authorship/HCAuthorship/google-books-common-words.txt'
local_vocab_file = '../google-books-common-words.txt'

try :
    os.listdir(cloud_path)
    data_path = data_cloud_path
    lib_path = cloud_lib_path
    vocab_file = local_vocab_file
    print('Running remotely')
except :
    print('Running locally')
    data_path = data_local_path
    lib_path = local_lib_path
    vocab_file = local_vocab_file


sys.path.append(lib_path)
from AuthAttLib.AuthAttLib import to_docTermCounts
from AuthAttLib.FreqTable import FreqTable, FreqTableClassifier

clf_names=[
'freq_table_chisq',
'freq_table_cosine',
'freq_table_LL',
'freq_table_CR',
'freq_table_HC',
'multinomial_NB',
'KNN_5',
'KNN_2',
'logistic_regression',
'SVM',
]

vocab_sizes=[250, 1000, 3000]
df = pd.DataFrame()


n_split = 10
no_clf = 9
no_vocab_sizes = 3
for ic in range(no_clf) :
    for iv in range(no_vocab_sizes) :
        vocab_size = vocab_sizes[iv] 
        clf_name = clf_names[ic]
        print("classifier = {}".format(clf_name))
        print("vocab size = {}".format(vocab_size))
        acc = evaluate_classifier(clf_name, vocab_size, n_split)
        print("avg. accuracy = {}".format(np.mean(acc)))
        df = df.append({'clf_name' : clf_name,
            'vocab_size' : vocab_size,
            'accuracy' : np.mean(acc),
            'std' : np.std(acc),
            'n_split' : n_split
            }, ignore_index = True)        
        df.to_csv('results.csv')
