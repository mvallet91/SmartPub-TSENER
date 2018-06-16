# @author Daniel Vliegenthart

from m1_postprocessing import extract_new_entities, filtering


model_names = ['dataset_50', 'method_50']
data_date = '2018_05_28'
context_words = { 'dataset_50': ['dataset', 'corpus', 'collection', 'repository', 'benchmark'] }
original_seeds = { 'dataset_50': ['buzzfeed', 'pslnl', 'dailymed', 'robust04', 'scovo', 'ask.com', 'cacm', 'stanford large network dataset', 
 'mediaeval', 'lexvo', 'spambase', 'shop.com', 'orkut', 'jnlpba', 'cyworld', 'citebase', 'blog06', 'worldcat', 
 'booking.com', 'semeval', 'imagenet', 'nasdaq', 'brightkite', 'movierating', 'webkb', 'ionosphere', 'moviepilot', 
 'duc2001', 'datahub', 'cifar', 'tdt', 'refseq', 'stack overflow', 'wikiwars', 'blogpulse', 'ws-353', 'gerbil', 
 'wikia', 'reddit', 'ldoce', 'kitti dataset', 'specweb', 'fedweb', 'wt2g', 'as3ap', 'friendfeed', 'new york times', 
 'chemid', 'imageclef', 'newegg']}

iteration = 'coner_' + data_date

def main():
    
  # Generate data statistics for 'ratings'
  print("\n\n#################################")
  print("##### FILTERINGS STATISTICS #####")
  print("#################################")
  
  for model_name in [model_names[0]]:
    rel_scores, entity_list = filtering.read_coner_overviews(model_name, data_date)
    filter_results = []

    filter_results.append(['Pointwise Mutual Information', filtering.filter_pmi(model_name, iteration, context_words[model_name])])

    filter_results.append(['Wordnet + Stopwords', filtering.filter_ws(model_name, iteration)])

    # WAITING FOR ”/embedding_models/modelword2vecbigram.ve
    # filter_results.append(['Similar Terms', filtering.filter_st(model_name, iteration, original_seeds[model_name])])

    filter_results.append(['Knowledge Base Look-up', filtering.filter_kbl(model_name, iteration, original_seeds[model_name])])

    filter_results.append(['Ensemble Majority Vote', filtering.majority_vote(model_name, iteration)])

    filter_results.append(['Coner Human Feedback', filtering.coner_filtering(model_name, iteration)])

    filter_results.append(['Coner Human Feedback + Ensemble Majority Vote', filtering.mv_coner_filtering(model_name, iteration)])

    print(f'{model_name}: Entities evaluated by Coner: {len(entity_list)}')

    # Overview of ratings for facets and categories
    print(f'\n\n<MODEL NAME>: <FILTERING METHOD> filter kept <FILTERED ENTITIES>/<UNFILTERED ENTITIES> (<PERCENTAGE>) of unfiltered extracted entities by model\n-------------------------------------------------------------------------------------------------------')

    header = [f'<MODEL NAME>', '<FILTERING METHOD>', f'<FILTERED ENTITIES>/<UNFILTERED ENTITIES> (<PERCENTAGE>)']
    print("{: <20} {: <50} {: <40}".format(*header))

    table_data = []
    for results in filter_results:
      if len(results[1]) > len(entity_list): results[1] = results[1][0:len(entity_list)]
      table_data.append([model_name, results[0], f'{len(results[1])}/{len(entity_list)} ({round(float(len(results[1]*100))/len(entity_list),1)}%)'])

    for row in table_data:
      print("{: <20} {: <50} {: <40}".format(*row))

if __name__ == "__main__":
  main()

