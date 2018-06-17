# @author Daniel Vliegenthart

import os
from m1_postprocessing import extract_new_entities, filtering
from config import ROOTPATH

model_names = ['dataset_50', 'method_50']
data_date = '2018_05_28'


# iteration = 'coner_' + data_date
iteration = 0

def main():
    
  # Generate data statistics for 'ratings'
  print("\n\n#################################")
  print("##### FILTERINGS STATISTICS #####")
  print("#################################")
  
  for model_name in [model_names[0]]:
    rel_scores, coner_entity_list = filtering.read_coner_overview(model_name, data_date)
    filter_results = []

    nr_entities = len(read_extracted_entities(model_name, iteration))

    filter_results.append(['Pointwise Mutual Information', execute_filter(model_name, 'pmi', iteration)])

    filter_results.append(['Wordnet + Stopwords', execute_filter(model_name, 'ws', iteration)])

    # WAITING FOR ”/embedding_models/modelword2vecbigram.ve
    # filter_results.append(['Similar Terms', execute_filter(model_name, 'st', iteration)])

    filter_results.append(['Knowledge Base Look-up', execute_filter(model_name, 'kbl', iteration)])

    filter_results.append(['Ensemble Majority Vote', execute_filter(model_name, 'majority', iteration)])

    filter_results.append(['Coner Human Feedback', execute_filter(model_name, 'coner', iteration)])

    filter_results.append(['Coner Human Feedback + Ensemble Majority Vote', execute_filter(model_name, 'mv_coner', iteration)])

    print(f'{model_name}: Entities evaluated by Coner: {nr_entities}')

    # Overview of ratings for facets and categories
    print(f'\n\n<MODEL NAME>: <FILTERING METHOD> filter kept <FILTERED ENTITIES>/<UNFILTERED ENTITIES> (<PERCENTAGE>) of unfiltered extracted entities by model\n-------------------------------------------------------------------------------------------------------')

    header = [f'<MODEL NAME>', '<FILTERING METHOD>', f'<FILTERED ENTITIES>/<UNFILTERED ENTITIES> (<PERCENTAGE>)']
    print("{: <20} {: <50} {: <40}".format(*header))

    table_data = []
    for results in filter_results:
      if len(results[1]) > nr_entities: results[1] = results[1][0:nr_entities]
      table_data.append([model_name, results[0], f'{len(results[1])}/{nr_entities} ({round(float(len(results[1]*100))/nr_entities,1)}%)'])

    for row in table_data:
      print("{: <20} {: <50} {: <40}".format(*row))

def execute_filter(model_name, filter_name, iteration):
  context_words = { 'dataset_50': ['dataset', 'corpus', 'collection', 'repository', 'benchmark'] }
  original_seeds = { 'dataset_50': ['buzzfeed', 'pslnl', 'dailymed', 'robust04', 'scovo', 'ask.com', 'cacm', 'stanford large network dataset', 
    'mediaeval', 'lexvo', 'spambase', 'shop.com', 'orkut', 'jnlpba', 'cyworld', 'citebase', 'blog06', 'worldcat', 
    'booking.com', 'semeval', 'imagenet', 'nasdaq', 'brightkite', 'movierating', 'webkb', 'ionosphere', 'moviepilot', 
    'duc2001', 'datahub', 'cifar', 'tdt', 'refseq', 'stack overflow', 'wikiwars', 'blogpulse', 'ws-353', 'gerbil', 
    'wikia', 'reddit', 'ldoce', 'kitti dataset', 'specweb', 'fedweb', 'wt2g', 'as3ap', 'friendfeed', 'new york times', 
    'chemid', 'imageclef', 'newegg']}

  path = ROOTPATH + '/processing_files/' + model_name + '_filtered_entities_' + filter_name + '_' + str(iteration) + '.txt'
  if os.path.isfile(path):
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
      return filtering.filter_st(model_name, iteration, original_seeds[model_name])
    if filter_name == 'kbl':
      return filtering.filter_kbl(model_name, iteration, original_seeds[model_name])
    if filter_name == 'majority':
      return filtering.majority_vote(model_name, iteration)
    if filter_name == 'coner':
      return filtering.filter_coner(model_name, iteration)
    if filter_name == 'mv_coner':
      return filtering.filter_mv_coner(model_name, iteration)

    return None

def read_extracted_entities(model_name, iteration):
  path = ROOTPATH + '/processing_files/' + model_name + '_extracted_entities_' + str(iteration) + '.txt'
  with open(path, "r") as f:
    extracted_entities = [e.strip().lower() for e in f.readlines()]
  f.close()
  return extracted_entities

if __name__ == "__main__":
  main()



















