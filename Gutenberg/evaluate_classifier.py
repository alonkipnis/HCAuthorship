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
from AuthAttLib.AuthAttLib import to_docTermCounts
from AuthAttLib.FreqTable import FreqTable, FreqTableClassifier


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



vocab_file = '../google-books-common-words.txt'
def get_vocab(vocab_file, n = 5000) :
    most_common_list = pd.read_csv(vocab_file, sep = '\t', header=None
                              ).iloc[:,0].str.lower().tolist()

    return most_common_list[:n]


def get_counts_labels_from_file_by_line(data_path, vocab) :
    X = []
    y = []
    fn = glob.glob(data_path)
    if len(fn) == 0 :
        print(f"Did not find any files in {data_path}")
        exit(1)

    print(f"Reading data from {fn[0]}...", end=' ')
    X = []
    Y = []
    df = pd.read_csv(fn[0], chunksize=500)
    for chunk in tqdm(df, unit = " chunks per") :
        for r in chunk.iterrows() :
            dt = to_docTermCounts([r[1].text], 
                                vocab=vocab
                                 )
            X += [FreqTable(dt[0], dt[1])._counts]
            y += [r[1].author]

    return X, y


def get_counts_labels_from_folder(data_folder_path, vocab) :
    X = []
    y = []
    print(f"Reading data from {data_folder_path}....", end=" ")
    lo_files = glob.glob(data_folder_path + '/*.csv')
    print(f"Found {len(lo_files)} files.")
    for fn in lo_files :
        try :
            dfr = pd.read_csv(fn)
            dt = to_docTermCounts(dfr.text, vocab=vocab)
            X += [FreqTable(dt[0], dt[1])._counts]
            y += dfr.author.values[0]
        except :
            print(f"Could not read {fn}.")
    return X, y


def get_counts_labels_from_file(data_path, vocab) :
    X = []
    y = []
    fn = glob.glob(data_path)
    if len(fn) == 0 :
        print(f"Did not find any files in {data_path}")
        exit(1)
    print(f"Reading data from {fn[0]}...", end=' ')
    df = pd.read_csv(fn[0])
    print("Done.")

    X = []
    y = []
    for r in df.iterrows() :
        dt = to_docTermCounts([r[1].text], 
                            vocab=vocab
                             )
        X += [FreqTable(dt[0], dt[1])._counts]
        y += [r[1].author]
    
    return X, y

def get_counts_labels(*args) :
    return get_counts_labels_from_file_by_line(*args)


def evaluate_classifier(clf_name, data_path, vocab_size, n_split) -> List :
    """
    Mean accuracy and std over n_split runs

    """
    clf = lo_classifiers[clf_name](**lo_args[clf_name])

    kf = KFold(n_splits=n_split, shuffle=True)

    #load data:

    vocab = get_vocab(vocab_file, vocab_size)
    #data_df = read_data(data_path)
    X, y = get_counts_labels(data_path, vocab)
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
  parser.add_argument('-i', type=str, help='data file (csv)')
  parser.add_argument('-n', type=int, help='n split (integer)', default=10)
  parser.add_argument('-s', type=int, help='vocabulary size (integer)', default=500)
  parser.add_argument('-c', type=str, help='classifier name (one of '\
    + str(lo_classifiers) +')', default='freq_table_HC')
  args = parser.parse_args()
  if not args.i:
      print('ERROR: The data file is required')
      parser.exit(1)
  else :
    input_filename = args.i

 
  print('Evaluating classifier {}'.format(args.c))
  print('\tdata file = {}'.format(input_filename))
  print('\tnumber of train/val splits = {}'.format(args.n))
  print('\tvocabulary size = {}'.format(args.s))
  acc, std = evaluate_classifier(args.c, args.i, args.s, args.n)
  print("Average accuracy = {}".format(acc))
  
  print("STD = {}".format(std))

if __name__ == '__main__':
  main()
