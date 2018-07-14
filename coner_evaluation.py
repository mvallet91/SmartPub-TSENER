# @author Daniel Vliegenthart

import os
from m1_preprocessing import seed_data_extraction, term_sentence_expansion, training_data_generation, ner_training
from m1_postprocessing import extract_new_entities,filtering
from config import ROOTPATH, data_date
import time
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet

model_names = ['dataset_50', 'method_50']

filter_iteration = 'coner_' + data_date
# filter_iteration = 0
expansion_iteration = 1

run_filters = True
run_expansion = False

def main():
  if run_filters: evaluate_filters()
  if run_expansion: evaluate_expansion()

def evaluate_filters():
  force = False

  start = time.time()

  # Generate data statistics for filtering
  print("\n\n#################################")
  print("##### FILTERINGS STATISTICS #####")
  print("#################################")
  
  for model_name in model_names:
    rel_scores, coner_entity_list = filtering.read_coner_overview(model_name, data_date)
    filter_results = []

    nr_entities = len(read_extracted_entities(model_name, filter_iteration))

    filter_results.append(['Pointwise Mutual Information', execute_filter(model_name, 'pmi', filter_iteration, force)])
    print(round((time.time() - start)/60, 2), 'minutes since start\n')

    filter_results.append(['Wordnet + Stopwords', execute_filter(model_name, 'ws', filter_iteration, force)])
    print(round((time.time() - start)/60, 2), 'minutes since start\n')

    filter_results.append(['Similar Terms', execute_filter(model_name, 'st', filter_iteration, force)])
    print(round((time.time() - start)/60, 2), 'minutes since start\n')

    filter_results.append(['Knowledge Base Look-up', execute_filter(model_name, 'kbl', filter_iteration, force)])
    print(round((time.time() - start)/60, 2), 'minutes since start\n')

    filter_results.append(['Ensemble Majority Vote', execute_filter(model_name, 'majority', filter_iteration, force)])
    print(round((time.time() - start)/60, 2), 'minutes since start\n')

    filter_results.append(['Coner Human Feedback', execute_filter(model_name, 'coner', filter_iteration, force)])
    print(round((time.time() - start)/60, 2), 'minutes since start\n')

    filter_results.append(['Coner Human Feedback + Ensemble Majority Vote', execute_filter(model_name, 'mv_coner', filter_iteration, force)])
    print(round((time.time() - start)/60, 2), 'minutes since start\n')
    
    print(f'{model_name}: Entities evaluated by Coner: {len(rel_scores.keys())}')
    print(f'{model_name}: Extracted entities evaluated: {nr_entities}')

    # Overview of ratings for facets and categories
    print(f'\n\n<MODEL NAME>: <FILTERING METHOD> filter kept <FILTERED ENTITIES>/<UNFILTERED ENTITIES> (<PERCENTAGE>) of unfiltered extracted entities by model\n-------------------------------------------------------------------------------------------------------')

    header = [f'<MODEL NAME>', '<FILTERING METHOD>', f'<FILTERED ENTITIES>/<UNFILTERED ENTITIES> (<PERCENTAGE>)']
    print("{: <20} {: <50} {: <40}".format(*header))

    table_data = []
    for results in filter_results:
      if results is None: continue
      table_data.append([model_name, results[0], f'{len(results[1])}/{nr_entities} ({round(float(len(results[1]*100))/nr_entities,1)}%)'])

    for row in table_data:
      print("{: <20} {: <50} {: <40}".format(*row))

def evaluate_expansion():
  force = False
  
  start = time.time()

  # Generate data statistics for expansion
  print("\n\n################################")
  print("##### EXPANSION STATISTICS #####")
  print("################################")
  
  for model_name in model_names:
    rel_scores = term_sentence_expansion.read_coner_overview(model_name, data_date)
    nr_added_entities = len(rel_scores.keys())
    expansion_results = []
    nr_seeds = len(read_seeds(model_name, expansion_iteration, 'majority'))

    expansion_results.append(['Term Expansion', execute_expansion(model_name, 'te', expansion_iteration, force)])
    print(round((time.time() - start)/60, 2), 'minutes since start\n')

    expansion_results.append(['Term Expansion + Coner Expansion', execute_expansion(model_name, 'tece', expansion_iteration, force)])
    print(round((time.time() - start)/60, 2), 'minutes since start\n')

    expansion_results.append(['Term Expansion + Coner Expansion (Separate Clustering)', execute_expansion(model_name, 'tecesc', expansion_iteration, force)])
    print(round((time.time() - start)/60, 2), 'minutes since start\n')

    print(f'{model_name}: Coner entities of type "selected" and rated as "relevant: {nr_added_entities}')

    # Overview of ratings for facets and categories
    print(f'\n\n<MODEL NAME>: <EXPANSION METHOD> expanded entities from <SEED ENTITIES> to <EXPANDED ENTITIES> (<PERCENTAGE>)\n-------------------------------------------------------------------------------------------------------')

    header = [f'<MODEL NAME>', '<EXPANSION METHOD>', f'<SEED ENTITIES> -> <EXPANDED ENTITIES> (<PERCENTAGE>)']
    print("{: <20} {: <60} {: <40}".format(*header))

    table_data = []
    for results in expansion_results:
      if results is None: continue
      table_data.append([model_name, results[0], f'{nr_seeds} -> {nr_seeds + len(results[1])} (+{len(results[1])}, {round(float((nr_seeds + len(results[1]))*100)/nr_seeds,1)}%)'])

    for row in table_data:
      print("{: <20} {: <60} {: <40}".format(*row))

def execute_filter(model_name, filter_name, iteration, force=False):
  context_words = { 'dataset_50': ['dataset', 'corpus', 'collection', 'repository', 'benchmark'], 'method_50': ['method', 'algorithm', 'approach', 'evaluate'] }
  original_seeds = read_initial_seeds(model_name)

  path = ROOTPATH + '/processing_files/' + model_name + '_filtered_entities_' + filter_name + '_' + str(iteration) + '.txt'
  if not force and os.path.isfile(path) and not filter_name in ['majority', 'coner', 'mv_coner']:
    # print("Getting filtered entities from file")
    with open(path, "r") as f:
      return [e.strip().lower() for e in f.readlines()]
  else:
    # print("Calculating filtered entities")
    if filter_name == 'pmi':
      return filtering.filter_pmi(model_name, iteration, context_words[model_name])
    if filter_name == 'ws':
      return filtering.filter_ws(model_name, iteration)
    if filter_name == 'st':
      return filtering.filter_st(model_name, iteration, original_seeds)
    if filter_name == 'kbl':
      return filtering.filter_kbl(model_name, iteration, original_seeds)
    if filter_name == 'majority':
      return filtering.majority_vote(model_name, iteration)
    if filter_name == 'coner':
      return filtering.filter_coner(model_name, iteration)
    if filter_name == 'mv_coner':
      return filtering.filter_mv_coner(model_name, iteration)

    return None

def execute_expansion(model_name, expansion_name, iteration, force=False):
  print(f'Executing expansion for model_name: {model_name}, expansion_name: {expansion_name}, iteration: {iteration}')
  path = ROOTPATH + '/processing_files/' + model_name + '_expanded_seeds_' + expansion_name + '_' + str(iteration) + '.txt'
  if not force and os.path.isfile(path):
    # print("Getting filtered entities from file")
    with open(path, "r") as f:
      return [e.strip().lower() for e in f.readlines()]
  else:
    # print("Calculating filtered entities")
    if expansion_name == 'te':
      return term_sentence_expansion.term_expansion(model_name, iteration)
    if expansion_name == 'tece':
      return term_sentence_expansion.coner_term_expansion(model_name, iteration)
    if expansion_name == 'tecesc':
      return term_sentence_expansion.coner_term_expansion_separate_clustering(model_name, iteration)
    
    return None

def read_extracted_entities(model_name, iteration):
  path = ROOTPATH + '/processing_files/' + model_name + '_extracted_entities_' + str(iteration) + '.txt'
  with open(path, "r") as f:
    extracted_entities = [e.strip().lower() for e in f.readlines()]
  f.close()

  processed_entities = []
  for pp in extracted_entities:
      temp = pp.split(' ')
      if len(temp) > 1:
          bigram = list(nltk.bigrams(pp.split()))
          for bi in bigram:
              bi = bi[0].lower() + ' ' + bi[1].lower()
              processed_entities.append(bi)
      else:
          processed_entities.append(pp)

  return processed_entities

def read_seeds(model_name, iteration, filter):
  path = ROOTPATH + '/processing_files/' + model_name + '_seeds_' + str(iteration) + f'_{filter}.txt'
  with open(path, "r") as f:
    seeds = [e.strip().lower() for e in f.readlines()]
  f.close()
  return seeds

def read_initial_seeds(model_name):
  path = ROOTPATH + '/processing_files/' + model_name + '_seeds_0.txt'
  with open(path, "r") as f:
    seeds = [e.strip().lower() for e in f.readlines()]
  f.close()
  return seeds

if __name__ == "__main__":
  main()
