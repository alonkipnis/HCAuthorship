"""
Filter catalog file according to various criteria

"""

import pandas as pd
import numpy as np

def from_external_list(ds, file_to_list) :
	"""
	Args:
	-----
	ds : original catalog
	file_to_list : path to external list

	# reduce catalog based on an external list
	# external list has fields 'author' and 'doc_id'
	"""
	ls = pd.read_csv(file_to_list)
	ls = ls.filter(['author', 'doc_id']).drop_duplicates()
	ds_red = ds[ds.title.isin(ls.doc_id)]
	ds_red.loc[:,'dup'] = ds_red.groupby(['author', 'title'])['title'].transform('count')
	ds_red = ds_red.filter(['title', 'author']).drop_duplicates().join(ds_red, how = 'left', rsuffix='_r')\
	      .filter(['title', 'author', 'id', 'formats', 'LCC', 'type', 'subjects',
	               'authoryearofbirth', 'authoryearofdeath', 'language', 'count'])
	return ds_red

catalog_file = './Gutenberg_full_catalog.csv'
#def filter_catalog(full_catalog_file) :

ds_raw = pd.read_csv(catalog_file)

NIN_TITLES_PER_AUTHOR = 10
LIST_OF_NO_AUTHORS = ['(name unknown)', 'A. L. O. E.', 'A-No. 1']
LANGUAGE = "['en']"

ds = ds_raw.dropna()
ds = ds[(ds.language == "['en']") & (ds['type'] == 'Text')]
for nm in ['(name unknown)', 'A. L. O. E.', 'A-No. 1'] :
    ds = ds[~ds.author.str.contains(nm)]

ds = ds[ds.formats.str.contains('text/plain') | 
        ds.formats.str.contains('text/html')]

ds.loc[:,'authoryearofbirth'] = ds.authoryearofbirth.astype(int, errors='ignore')
ds.loc[:,'authoryearofdeath'] = ds.authoryearofdeath.astype(int, errors='ignore')
    
ds.loc[:,'count'] = ds.groupby('author')['author'].transform('count')
ds_red = ds[ds['count'] >= 10]

ds_red = ds_red.filter(['id', 'title', 'author', 'count', 'url', 'formats',
                        'authoryearofbirth', 'authoryearofdeath', 'subjects'])

ds_red.to_csv('./reduced_catalog.csv')