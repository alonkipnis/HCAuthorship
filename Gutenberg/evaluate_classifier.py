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
from sklearn.neural_network import MLPClassifier
import glob
import sys
from typing import List


data_cloud_path = '/scratch/users/kipnisal/Data/Gutenberg'
data_local_path = '/Users/kipnisal/Data/Gutenberg/Data'

cloud_lib_path = '/scratch/users/kipnisal/Lib/AuthorshipAttribution'
local_lib_path = '/Users/kipnisal/Data/Gutenberg/HCAuthorship'

cloud_vocab_file = '/Users/kipnisal/authorship/HCAuthorship/google-books-common-words.txt'
local_vocab_file = '../google-books-common-words.txt'

try :
    os.listdir(data_cloud_path)
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


most_common_list = pd.read_csv(vocab_file, sep = '\t', header=None
                              ).iloc[:,0].str.lower().tolist()

def get_n_most_common_words(n = 5000) :
    return most_common_list[:n]

def get_counts_labels(df, vocab) :
#prepare data:
    X = []
    y = []
    for r in df.iterrows() :
        dt = to_docTermCounts([r[1].text], 
                            vocab=vocab
                             )
        X += [FreqTable(dt[0], dt[1])._counts]
        y += [r[1].author]
    
    return X, y


lo_classifiers = {
            'freq_table_chisq' : FreqTableClassifier,
            'freq_table_cosine' : FreqTableClassifier,
            'freq_table_LL' : FreqTableClassifier,
            'freq_table_modLL' : FreqTableClassifier,
            'freq_table_FT' : FreqTableClassifier,
            'freq_table_Neyman' : FreqTableClassifier,
            'freq_table_CR' : FreqTableClassifier,
            'freq_table_HC' : FreqTableClassifier,
            'multinomial_NB' : MultinomialNB,
            'KNN_5' : KNeighborsClassifier,
            'KNN_2' : KNeighborsClassifier,
            'logistic_regression' : LogisticRegression,
            'SVM' : LinearSVC,
            'NeuralNet' : MLPClassifier,
                }

lo_args = {'multinomial_NB' : {},
           'freq_table_HC' : {'metric' : 'HC',
                          'alpha' : 0.2},
           'freq_table_chisq' : {'metric' : 'chisq'},
           'freq_table_cosine' : {'metric' : 'cosine'},
           'freq_table_LL' : {'metric' : 'log-likelihood'},
           'freq_table_modLL' : {'metric' : "mod-log-likelihood"},
           'freq_table_FT' : {'metric' : "freeman-tukey"},
           'freq_table_Neyman' : {'metric' : "neyman"},
           'freq_table_CR' : {'metric' : "cressie-read"},
           'KNN_5' : {'metric' : 'cosine',
                          'n_neighbors' : 5},
           'KNN_2' : {'metric' : 'cosine',
                          'n_neighbors' : 2},
            'SVM' : {'loss' : 'hinge'},
            'NeuralNet' : {'hidden_layer_sizes' : (1000, 500, 500, 500, 500),
            'early_stopping' : True,
             'alpha' : 1, 'max_iter' : 1000, },
            }

def read_data(data_path) :
    fn = glob.glob(data_path + '/Gutenberg*.csv')
    print(f"Found {len(fn)} data files")
    if len(fn) != 1 :
        print(f"Could not read data from {data_path}")
        exit(1)
    return pd.read_csv(fn)

def evaluate_classifier(clf_name, vocab_size, n_split) -> List :

    clf = lo_classifiers[clf_name](**lo_args[clf_name])

    kf = KFold(n_splits=n_split, shuffle=True)

    #load data:
    data_df = read_data(data_path)
    vocab = get_n_most_common_words(vocab_file)

    X, y = get_counts_labels(data_df, vocab)

    acc = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
                
        clf.fit(X_train,y_train)
        acc += [clf.score(X_test, y_test)]

    return acc