import pandas as pd
import numpy as np
import os
import nltk
from sklearn.model_selection import KFold
import sys
from evaluate_classifier import evaluate_classifier
from dask.distributed import Client

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
'NeuralNet'
]

vocab_sizes=[250, 1000, 3000]
vocab_sizes=[100]
n_split = 10


if __name__ == '__main__':
    client = Client()



    lo_fut = []
    lo_params = [(clf_name, vocab_size, n_split) for clf_name in clf_names for vocab_size in vocab_sizes]
    for param in lo_params[:2] :
        lo_fut += [client.map(evaluate_classifier, param[0], param[1], param[2])]
            
    res = client.gather(lo_fut)

    df = pd.DataFrame()
    for r,param in zip(res,lo_params) :
        df = df.append({'clf_name' : param[0],
            'vocab_size' : param[1],
            'accuracy' : np.mean(r),
            'std' : np.std(r),
            'n_split' : param[2]
            }, ignore_index = True)        
    df.to_csv('results.csv')





