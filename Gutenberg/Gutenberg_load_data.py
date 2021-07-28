import pandas as pd
import os
import glob
import sys
import argparse
from tqdm import tqdm

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


def main() :
  parser = argparse.ArgumentParser()
  parser = argparse.ArgumentParser(description='Read data')
  parser.add_argument('-i', type=str, help='data file (csv)')
  args = parser.parse_args()
  if not args.i:
      print('ERROR: The data file is required')
      parser.exit(1)
  else :
    input_filename = args.i

  vocab_size = 500
  vocab = get_vocab(vocab_file, vocab_size)
  print('\tdata file = {}'.format(input_filename))
  X,y = get_counts_labels_from_file_by_line(input_filename, vocab)
  print(f"X[:1] = {X[:1]}")
  print(f"y[:1] = {y[:1]}")

if __name__ == '__main__':
  main()

