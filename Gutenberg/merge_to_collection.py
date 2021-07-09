import os
import pandas as pd
import argparse
import glob
from tqdm import tqdm

"""
Merge many files downloaded from the Gutenberg Project into a single 
file. 

The input is the folder containing many files with prefix Gut* in 
.csv format. Each file corresponds to one title. 

Another file containg the list of titles to include in the collection

The output is one .csv file in which each row corresponds to 

"""

def get_collection(path) :

  lo_files = glob.glob(path + '/Gut*')

  df = pd.DataFrame()
  for fn in lo_files :
     try : 
       rec = pd.read_csv(fn)
       rec.loc[:,'len'] = len(rec['text'].values[0].split())
       rec = rec.drop(['text', 'Unnamed: 0'], axis=1)

       rec.loc[:,'Gut_id'] = os.path.basename(fn).split('.')[0]

       df = df.append(rec, sort = False)
     except KeyboardInterrupt :
       exit(1)
     except:
       print('could not parse {}'.format(fn))
      
  return df

def generate_collection_based_on_metadata(path_to_data, path_to_list, 
  out_filename) : 
  meta_data = pd.read_csv(path_to_list)
  print(f"Found {len(meta_data)} titles in list")

  lo_files = glob.glob(path_to_data + '/Gut*')
  print(f"Found {len(lo_files)} files in {path_to_data}")
  print("Reading and merging...")
  df = pd.DataFrame()

  for fn in lo_files :
     try :
       rec = pd.read_csv(fn)
       doc_id = rec['doc_id'].values[0]
       if doc_id in meta_data['doc_id'].tolist() :
         rec.loc[:,'len'] = len(rec['text'].values[0].split())
         rec = rec.drop(['Unnamed: 0'], axis=1)
         rec.loc[:,'Gut_id'] = os.path.basename(fn).split('.')[0]
         df = df.append(rec, sort = False)

     except KeyboardInterrupt :
       exit(1)
     except:
       print(f'could not parse {fn} with title {doc_id}')
  df.to_csv(out_filename)
  print(f"Included {len(df)} in collection. Stored in {out_filename}")

def save_as_txt(path_to_data, path_to_list, out_folder) : 
  meta_data = pd.read_csv(path_to_list)
  print(f"Found {len(meta_data)} titles in list")

  lo_files = glob.glob(path_to_data + '/Gut*')
  print(f"Found {len(lo_files)} files in {path_to_data}")
  print("Reading and merging...")
  df = pd.DataFrame()

  for fn in tqdm(lo_files) :
      try :
       rec = pd.read_csv(fn)
       doc_id = rec['doc_id'].values[0]
       if doc_id in meta_data['doc_id'].tolist() :
         rec.loc[:,'len'] = len(rec['text'].values[0].split())
         rec = rec.drop(['Unnamed: 0'], axis=1)
         filename = os.path.basename(fn).split('.')[0] + '.txt'
         rec.loc[:,'filename'] = filename
         with open(out_folder + '/' + filename, 'wt') as f:
            f.write(rec['text'].values[0])
         df = df.append(rec.drop(['text'], axis=1))
      except KeyboardInterrupt :
         exit(1)
      except:
         print(f'could not parse {fn} with title {doc_id}')

  out_filename = out_folder + '/list_of_title.csv'
  df.to_csv(out_filename)
  print(f"Included {len(df)} in collection. Stored in {out_filename}")


def main() :
  parser = argparse.ArgumentParser()
  parser = argparse.ArgumentParser(description='Download title from catalog')
  parser.add_argument('-i', type=str, help='data folder')
  parser.add_argument('-l', type=str, help='list file')
  parser.add_argument('-o', type=str, help='output file', default='collection_info.csv')
  args = parser.parse_args()
  if not args.i:
      print('ERROR: The data folder is requires')
      parser.exit(1)

  #generate_collection_based_on_metadata(args.i, args.l, args.o)
  save_as_txt(args.i, args.l, args.o)

if __name__ == '__main__':
  main()
  #load database file

  