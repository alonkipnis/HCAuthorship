#!/usr/bin/env python
# coding: utf-8

"""
Download titles from Gutenberg Project accoding to a catalog file
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

#import auxiliary functions for python
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers

import urllib3  
import argparse
from bs4 import BeautifulSoup

def download_etext(formats) :
    dc = dict(eval(formats))
    good_keys = [k for k in dc if 'text' in k]
    
    txt = None
    for k in good_keys :
        try : 
            url = dc[k]
            typ = k.split(" ")[0]
            http = urllib3.PoolManager()
            response = http.request('GET', url)
            txt = response.data.decode()
            
            if 'text/html' in typ :
                txt = BeautifulSoup(txt, "html.parser").text
                
            break # until succeed
        except KeyboardInterrupt:
                exit(1)
        except :
            None
    return txt

#download full texts
def download_full_texts(db, out_path) :
    data = pd.DataFrame()
    exception_list = []
    acc =0 
    for r in tqdm(db.iterrows()) :
            try :
              # checking if title already exists
              open(out_path + "Gut_{}.csv".format(r[0]))
              continue
            except KeyboardInterrupt:
                exit(1)
            except FileNotFoundError:
              print("Trying to download {} ...".format(r[1].title), end =" ")  
              try : 
                text = strip_headers(down_etext(r[1].id)).strip()
              except KeyboardInterrupt:
                exit(1)
              except NameError:
                try :
                    text = strip_headers(download_etext(r[1].formats)).strip()
                    print("succeeded.")
                except (NameError, AttributeError):
                    print("could not find {}".format(r[1].title))
                    exception_list.append(r[1].title)
                    continue
            
                data = pd.DataFrame(
                        {'doc_id' :  r[1].title,
                         'author' : r[1].author,
                          'text' : text,
                          'authoryearofbirth' : r[1].authoryearofbirth,
                          'authoryearofdeath' : r[1].authoryearofdeath,
                          'no_words' : len(text.split())
                          }, index = [r[0]])

                filename = "Gut_{}.csv".format(r[0])
                data.to_csv(out_path + '/' + filename)
    return data, exception_list

if __name__ == '__main__':
  #load database file

  parser = argparse.ArgumentParser()
  parser = argparse.ArgumentParser(description='Download title from catalog')
  parser.add_argument('-i', type=str, help='catalog file (csv)')
  parser.add_argument('-o', type=str, help='Output directory')
  args = parser.parse_args()
  if not args.i:
      print('ERROR: The catalog file is required')
      parser.exit(1)
  if not args.o:
      print('ERROR: The output folder is required')
      parser.exit(1)

  catalog_file = args.i
  db = pd.read_csv(catalog_file)
  download_full_texts(db, args.o)

