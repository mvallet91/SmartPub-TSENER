from m1_preprocessing import seed_data_extraction, term_sentence_expansion, training_data_generation, ner_training
from m1_postprocessing import extract_new_entities, filtering
import config as cfg
import gensim
import elasticsearch
import time

doc2vec_model = gensim.models.Doc2Vec.load(cfg.ROOTPATH + '/embedding_models/doc2vec.model')
es = elasticsearch.Elasticsearch([{'host': 'localhost', 'port': 9200}])

# User input
seeds = ['buzzfeed', 'pslnl', 'dailymed', 'robust04', 'scovo', 'ask.com', 'cacm', 'stanford large network dataset', 'mediaeval', 'lexvo', 'spambase', 'shop.com', 'orkut', 'jnlpba', 'cyworld', 'citebase', 'blog06', 'worldcat', 'booking.com', 'semeval', 'imagenet', 'nasdaq', 'brightkite', 'movierating', 'webkb', 'ionosphere', 'moviepilot', 'duc2001', 'datahub', 'cifar', 'tdt', 'refseq', 'stack overflow', 'wikiwars', 'blogpulse', 'ws-353', 'gerbil', 'wikia', 'reddit', 'ldoce', 'kitti dataset', 'specweb', 'fedweb', 'wt2g', 'as3ap', 'friendfeed', 'new york times', 'chemid', 'imageclef', 'newegg']
context_words = ['dataset', 'corpus', 'collection', 'repository', 'benchmark']
sentence_expansion = True
training_cycles = 2
model_name = 'dataset_50'
filtering_pmi = True
filtering_st = True
filtering_ws = True
filtering_kbl = True
filtering_majority = True

start = time.time()
for cycle in range(training_cycles):
    seed_data_extraction.sentence_extraction(model_name, cycle, seeds)
    print(round((time.time() - start)/60, 2), 'minutes since start')
    term_sentence_expansion.term_expansion(model_name, cycle)
    print(round((time.time() - start)/60, 2), 'minutes since start')
    term_sentence_expansion.sentence_expansion(model_name, cycle, doc2vec_model)
    print(round((time.time() - start)/60, 2), 'minutes since start')
    training_data_generation.sentence_labelling(model_name, cycle, sentence_expansion)
    print(round((time.time() - start)/60, 2), 'minutes since start')
    ner_training.create_prop(model_name, cycle, sentence_expansion)
    print(round((time.time() - start)/60, 2), 'minutes since start')
    ner_training.train_model(model_name, cycle)
    print(round((time.time() - start)/60, 2), 'minutes since start')
    extract_new_entities.ne_extraction(model_name, cycle, sentence_expansion)
    if filtering_pmi:
        filtering.filter_pmi(model_name, cycle, context_words)
    if filtering_st:
        filtering.filter_st(model_name, cycle, seeds)
    if filtering_ws:
        filtering.filter_ws(model_name, cycle)
    if filtering_kbl:
        filtering.filter_kbl(model_name, cycle, seeds)
    filtering.majority_vote(model_name, cycle)
    print(round((time.time() - start)/60, 2), 'minutes since start')
    print('-'*50)
    print('')
