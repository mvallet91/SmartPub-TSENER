# @author Daniel Vliegenthart

import os
from m1_preprocessing import seed_data_extraction, term_sentence_expansion, training_data_generation, ner_training
from m1_postprocessing import extract_new_entities,filtering
from config import ROOTPATH, data_date
import time

model_names = ['dataset_50', 'method_50']

# filters = ['pmi', 'majority']
filters = ['extracted']

# iteration = 'coner_' + data_date
iteration = 0

# table = [[0 for j in range(0,len(filters))] for i in range(0,len(filters)) ]

def main():
  for filter1 in filters:
    dataset_ents = read_filter(filter1, 'dataset_50', iteration)
    method_ents = read_filter(filter1, 'method_50', iteration)


    dataset_ents = [ent.lower().strip("\n ") for ent in dataset_ents]
    method_ents = [ent.lower().strip("\n ") for ent in method_ents]

    doubly = list(set(dataset_ents).intersection(set(method_ents)))
    total = list(set(dataset_ents).union(set(method_ents)))

    print(f'{filter1}: {len(doubly)}/{len(total)} {round(float(len(doubly))/len(total)*100,1)}%')


    # filter_lists = { filter : read_filter(filter, model_name, iteration) for filter in filters }


    # for i2, filter2 in enumerate(filters):
    #   fnr = fn_filters(filter_lists, filter1,filter2)

    #   print(filter1, filter2, fnr)

    #   table[i2] = fnr

    # print_overlap(table)

def print_overlap(table):
  print("\n\n")
  for index, value in enumerate(table):
    print("  &  " + str(value) + "\\%", end='')
  print('  \\\\')



def read_filter(filter, model_name, iteration):
  path = ROOTPATH + '/processing_files/' + model_name + '_extracted_entities_' + str(iteration) + '.txt'
  with open(path, "r") as f:
    filtered_entities = [e.strip().lower() for e in f.readlines()]
  f.close()
  return filtered_entities

if __name__ == "__main__":
  main()
