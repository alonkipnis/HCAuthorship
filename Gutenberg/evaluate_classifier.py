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
from sklearn.ensemble import RandomForestClassifier
import glob
import sys
from typing import List
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


sys.path.append("../")
from AuthAttLib.FreqTable import FreqTableClassifier

from count_words import get_word_counts_from_pkl_file

lo_classifiers = {
            'freq_table_chisq' : FreqTableClassifier,
            'freq_table_cosine' : FreqTableClassifier,
            'freq_table_LL' : FreqTableClassifier,
            'freq_table_modLL' : FreqTableClassifier,
            'freq_table_FT' : FreqTableClassifier,
            'freq_table_Neyman' : FreqTableClassifier,
            'freq_table_CR' : FreqTableClassifier,
            'freq_table_HC' : FreqTableClassifier,
            'freq_table_HC_org' : FreqTableClassifier,
            'multinomial_NB' : MultinomialNB,
            'random_forest' : RandomForestClassifier,
            'KNN_5' : KNeighborsClassifier,
            'KNN_2' : KNeighborsClassifier,
            'logistic_regression' : LogisticRegression,
            'SVM' : LinearSVC,
            'NeuralNet' : MLPClassifier,
            'logistic_regression' : LogisticRegression,
                }


lo_args = {'multinomial_NB' : {},
           'freq_table_HC' : {'metric' : 'HC',
                          'gamma' : 0.2},
           'freq_table_HC_org' : {'metric' : 'HC',
                          'gamma' : 0.2, 'HCtype' : 'original'},
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
            'NeuralNet' : {'alpha' : 1, 'max_iter' : 1000, 'hidden_layer_sizes' : (128,64,64)},
            'random_forest' : {'max_depth' : 10, 'n_estimators' : 30,
            'max_features' : 10},
            'logistic_regression' : {},
            }



def evaluate_classifier(clf_name, data, n_split) -> List :
    """
    Mean accuracy and std over n_split runs

    """
    clf = lo_classifiers[clf_name](**lo_args[clf_name])

    kf = KFold(n_splits=n_split, shuffle=True)

    #load data:

    X, y = data
    #X, y = get_counts_labels_from_folder(data_path, vocab)    

    acc = []
    
    for train_index, test_index in tqdm(kf.split(X)):
        X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
                
        clf.fit(X_train,y_train)
        acc += [clf.score(X_test, y_test)]

    return np.mean(acc), np.std(acc)


def main() :
  parser = argparse.ArgumentParser()
  parser = argparse.ArgumentParser(description='Evaluate classifier on'
  ' Authorship challenge')
  parser.add_argument('-i', type=str, help='data file (pkl)')
  parser.add_argument('-n', type=int, help='n split (integer)', default=10)
  parser.add_argument('-c', type=str, help='classifier name (one of '\
    + str(lo_classifiers) +')', default='freq_table_HC')
  args = parser.parse_args()
  if not args.i:
      print('ERROR: The data file is required')
      parser.exit(1)
  else :
    input_filename = args.i

 
  print(f"Loading data from {args.i}..,", end=' ')
  data = get_word_counts_from_pkl_file(args.i)
  print(f"Found {len(data[0])} samples.")

  print(f'Evaluating classifier {args.c}')
  print(f'\tnumber of train/val splits = {args.n}')
  acc, std = evaluate_classifier(args.c, data, args.n)
  print(f"Average accuracy = {acc}")
  print(f"STD = {std}")

if __name__ == '__main__':
  main()
