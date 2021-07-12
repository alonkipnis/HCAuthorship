
import pandas as pd
import numpy as np
import os
import nltk
from sklearn.model_selection import KFold
import sys
import argparse
from evaluate_classifier import evaluate_classifier
from dask.distributed import Client

clf_names=[
#'freq_table_chisq',
#'freq_table_cosine',
#'freq_table_LL',
#'freq_table_CR',
#'freq_table_HC',
#'multinomial_NB',
#'KNN_5',
#'freq_table_cosine',
#'freq_table_LL',
#'freq_table_CR',
#'freq_table_HC',
#'multinomial_NB',
#'KNN_5',
#'KNN_2',
'logistic_regression',
'NeuralNet',
'random_forest'
#'SVM',
]


def main() :
  parser = argparse.ArgumentParser()
  parser = argparse.ArgumentParser(description='Evaluate hardcoded list of '
  'classifiers Authorship challenge')
  parser.add_argument('-i', type=str, help='data file (csv)')
  parser.add_argument('-n', type=str, help='n split (integer)', default=10)
  args = parser.parse_args()
  if not args.i:
      print('ERROR: The data file is required')
      parser.exit(1)
  else :
    data_path = args.i

  print('\tData: {}'.format(data_path))
  print('\tNumber of train/val splits = {}'.format(args.n))

  vocab_sizes=[250, 1000, 3000]
  df = pd.DataFrame()

  n_split = args.n
  no_vocab_sizes = len(vocab_sizes)
  no_clf = len(clf_names)


  client = Client(memory_limit='8GB')
    #print(client)
 
  def evaluate_classifier_d(params) :
    return evaluate_classifier(params[0], params[1], params[2], params[3])
  
  # lo_fut = []
  # lo_params = [(clf_name, vocab_size, n_split) for clf_name in clf_names for vocab_size in vocab_sizes]
  # for params in lo_params[:2] :
  #   #lo_fut += [client.submit(evaluate_classifier_d, params)]
  #   lo_fut += [evaluate_classifier_d(params)]

  lo_fut = []
  lo_params = []
  for ic in range(no_clf) :
    for iv in range(no_vocab_sizes) :
        vocab_size = vocab_sizes[iv] 
        clf_name = clf_names[ic]
        print("classifier = {}".format(clf_name))
        print("vocab size = {}".format(vocab_size))

        params = (clf_name, data_path, vocab_size, n_split)
        lo_params += [params]
        lo_fut += [client.submit(evaluate_classifier_d, params)]

  res = client.gather(lo_fut)
  
  for r,param in zip(res, lo_params) :
      df = df.append({'clf_name' : param[0],
            'vocab_size' : param[2],
            'accuracy' : r[0],
            'std' : r[1],
            'n_split' : param[3]
            }, ignore_index = True)        
      df.to_csv('results.csv')

if __name__ == '__main__':
  main()

