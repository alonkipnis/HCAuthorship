import pandas as pd
import numpy as np
import os
import glob
import sys
from typing import List
import argparse
from tqdm import tqdm
import warnings
import pickle
import logging


logging.basicConfig(level=logging.INFO)

sys.path.append("../")
from AuthAttLib.AuthAttLib import to_docTermCounts
from AuthAttLib.FreqTable import FreqTable, FreqTableClassifier


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

    print(f"Reading data from {fn[0]}:")
    X = []
    Y = []
    df = pd.read_csv(fn[0], chunksize=500)
    for chunk in tqdm(df, unit = " chunk") :
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

def get_word_counts_from_pkl_file(filename) :
    with open(filename, "rb" ) as f :
        X,y = pickle.load(f)
    return X,y



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

    print(f"Reading data from {fn[0]}:")
    X = []
    Y = []
    df = pd.read_csv(fn[0], chunksize=500)
    for chunk in tqdm(df, unit = " chunk") :
        for r in chunk.iterrows() :
            dt = to_docTermCounts([r[1].text], 
                                vocab=vocab
                                 )
            X += [np.squeeze(np.asarray(dt[0].todense()))]
            y += [r[1].author]

    return X, y

def counts_words(*args) :
    return get_counts_labels_from_file_by_line(*args)

def get_word_counts_from_pkl_file(filename) :
    with open(filename, "rb" ) as f :
        X,y = pickle.load(f)
    return X,y

def main() :
  parser = argparse.ArgumentParser()
  parser = argparse.ArgumentParser(description='Evaluate classifier on'
  ' Authorship challenge')
  parser.add_argument('-i', type=str, help='data file (csv)')
  parser.add_argument('-v', type=str, help='vocabulary file', default='../google-books-common-words.txt')
  parser.add_argument('-s', type=int, help='vocabulary size (integer)', default=500)
  parser.add_argument('-o', type=str, help='output file', default=f'./counts.pkl')

  args = parser.parse_args()
  if not args.i:
      print('ERROR: The data file is required')
      parser.exit(1)
  else :
    input_filename = args.i

  dirname = os.path.dirname(args.o)
  out_filename = dirname + '/' + os.path.basename(args.o).split('.')[0] + f'_{args.s}.pkl'

  vocab = get_vocab(args.v, args.s)
  logging.info(f'Retained vocabulary of size = {len(vocab)}')
  logging.info(f'Reading text data from {args.i}')
  X, y = counts_words(args.i, vocab)
  pickle.dump([X,y], open(out_filename, "wb"))
  logging.info(f"Stored {len(X)} samples to {out_filename}.")

  
if __name__ == '__main__':
  main()
