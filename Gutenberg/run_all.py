import pandas as pd
import numpy as np
import os
import nltk
from sklearn.model_selection import KFold
import sys
import argparse
from evaluate_classifier import evaluate_classifier
from count_words import get_word_counts_from_pkl_file

clf_names=[
'freq_table_HC',
'freq_table_HC_org',
#'freq_table_chisq',
#'freq_table_cosine',
#'freq_table_LL',
#'freq_table_CR',
#'freq_table_cosine',
#'multinomial_NB',
#'KNN_5',
#'KNN_2',
#'logistic_regression',
#'NeuralNet',
#'NeuralNet',
#'SVM',
]


def main() :
	parser = argparse.ArgumentParser()
	parser = argparse.ArgumentParser(description='Evaluate hardcoded list of '
	'classifiers in Authorship challenge')
	parser.add_argument('-i', type=str, help='data file (csv)')
	parser.add_argument('-n', type=int, help='n split (integer)', default=20)
	args = parser.parse_args()
	if not args.i:
			print('ERROR: The data file is required')
			parser.exit(1)
	else :
		data_path = args.i

	print('\tData: {}'.format(data_path))
	print('\tNumber of train/val splits = {}'.format(args.n))

	vocab_sizes=[# 250,
	# 1000,
 		3000]
	
	n_split = args.n
	no_vocab_sizes = len(vocab_sizes)
	no_clf = len(clf_names)

	df = pd.DataFrame()
	for ic in range(no_clf) :
		for iv in range(no_vocab_sizes) :
				vocab_size = vocab_sizes[iv]
				clf_name = clf_names[ic]
				print("classifier = {}".format(clf_name))
				fn = args.i.split('.pkl')[0]
				data_file = fn + f'_{vocab_size}.pkl'

				print(f"Loading data from {data_file}..,", end=' ')
				try :
					data = get_word_counts_from_pkl_file(data_file)
				except : 
					print("Could not load data file")
					continue
				print(f"Found {len(data[0])} samples.")

				acc, std = evaluate_classifier(clf_name, data, n_split)
				print("Average accuracy = {}".format(acc))  
				print("STD = {}".format(std))

				df = df.append({'clf_name' : clf_name,
						'dataset' : data_file,
						'accuracy' : acc,
						'std' : std,
						'n_split' : n_split
						}, ignore_index = True)        
				df.to_csv('results_HC_org_vs_star.csv')

if __name__ == '__main__':
	main()
