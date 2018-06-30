# @author Daniel Vliegenthart

import os
from m1_preprocessing import seed_data_extraction, term_sentence_expansion, training_data_generation, ner_training
from m1_postprocessing import extract_new_entities,filtering
from config import ROOTPATH, data_date
import time

model_names = ['dataset_50', 'method_50']

# filter_iteration = 'coner_' + data_date
filter_iteration = 0
expansion_iteration = 1

run_filters = False
run_expansion = True


def main():
  model_name = 'method_50'
  iteration = 1
  term_sentence_expansion.coner_term_expansion(model_name, iteration)

if __name__ == "__main__":
  main()