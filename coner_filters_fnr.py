# @author Daniel Vliegenthart

import os
from m1_preprocessing import seed_data_extraction, term_sentence_expansion, training_data_generation, ner_training
from m1_postprocessing import extract_new_entities,filtering
from config import ROOTPATH, data_date
import time
import nltk

model_names = ['dataset_50', 'method_50']

filters = ['pmi', 'ws', 'st', 'kbl', 'majority', 'coner', 'mv_coner']

iteration = 'coner_' + data_date
filter_lists = {}

# table = [[0 for j in range(0,len(filters))] for i in range(0,len(filters)) ]

def main():
  for model_name in model_names:
    table = [0 for i in range(0,len(filters)) ]
    print(model_name)

    filter_lists = { filter : read_filter(filter, model_name, iteration) for filter in filters }

    filter1 ='coner'

    for i2, filter2 in enumerate(filters):
      fnr = fn_filters(filter_lists, filter1,filter2)

      print(filter1, filter2, fnr)

      table[i2] = fnr

    print_overlap(table)

def print_overlap(table):
  print("\n\n")
  for index, value in enumerate(table):
    print("  &  " + str(value) + "\\%", end='')
  print('  \\\\')

def fn_filters(filter_lists, f1,f2):
  if f1 == f2: return 0
  f1_list = filter_lists[f1].copy()
  f2_list = filter_lists[f2].copy()
  
  fp = 0

  for entity in f1_list:
    if entity not in f2_list: fp +=1

  if fp == 0: return 0.0

  print(fp)

  return round(100*(float(fp)/len(f1_list)), 1)

def read_filter(filter, model_name, iteration):
  path = ROOTPATH + '/processing_files/' + model_name + '_filtered_entities_' + filter + "_" + str(iteration) + '.txt'
  with open(path, "r") as f:
    filtered_entities = [e.strip().lower() for e in f.readlines()]
  f.close()

  processed_entities = []
  for pp in filtered_entities:
    temp = pp.split(' ')
    if len(temp) > 1:
      bigram = list(nltk.bigrams(pp.split()))
      for bi in bigram:
        bi = bi[0].lower() + ' ' + bi[1].lower()
        processed_entities.append(bi)
    else:
      processed_entities.append(pp)

  return processed_entities


if __name__ == "__main__":
  main()
