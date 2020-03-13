import pandas as pd
import numpy as np
import os
import nltk
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import sys
from evaluate_classifier import evaluate_classifier

data_filename = 'Gutenberg_reduced_100.csv'

cloud_path = '/scratch/users/kipnisal/Gutenberg/'
local_path = '/Users/kipnisal/Data/Gutenberg/'

cloud_lib_path = '/scratch/users/kipnisal/Lib/AuthorshipAttribution/'
local_lib_path = '/Users/kipnisal/Documents/Authorship/'

try :
    os.listdir(cloud_path)
    path = cloud_path
    lib_path = cloud_lib_path
except :
    print('Using local path')
    path = local_path
    lib_path = local_lib_path

sys.path.append(lib_path)
from AuthAttLib import to_docTermCounts
from FreqTable import FreqTable, FreqTableClassifier

clf_names=[
'freq_table_chisq',
'freq_table_cosine',
'freq_table_LL',
#'freq_table_modLL',
#'freq_table_FT',
#'freq_table_Neyman',
'freq_table_CR',
'freq_table_HC',
'multinomial_NB',
'KNN_5',
'KNN_2',
#'logistic_regression',
'SVM'
]

vocab_sizes=[250, 1000, 3000]
df = pd.DataFrame()

n_split=10
no_clf = 9
no_vocab_sizes = 3
for ic in range(no_clf) :
    for iv in range(no_vocab_sizes) :
        vocab_size = vocab_sizes[iv] 
        clf_name = clf_names[ic]
        print("classifier = {}".format(clf_name))
        print("vocab size = {}".format(vocab_size))
        acc = evaluate_classifier(clf_name, vocab_size, n_split)
        print("accuracy = {}".format(acc))
        df = df.append({'clf_name' : clf_name,
            'vocab_size' : vocab_size,
            'accuracy' : acc, 
            'n_split' : n_split
            }, ignore_index = True)        
        df.to_csv('results.csv')
