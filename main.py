from preprocessing import ner_training, expansion, seed_data_extraction
from postprocessing import training_data_generation, extract_new_entities, filtering
import config as cfg
import gensim
import elasticsearch

doc2vec_model = gensim.models.Doc2Vec.load(cfg.ROOTPATH + '/models/doc2vec.model')
es = elasticsearch.Elasticsearch([{'host': 'localhost', 'port': 9200}])

# User input
seeds = ['clueweb', 'imagenet', 'flickr', 'webkb', 'netflix', 'imdb']
context_words = ['dataset', 'corpus', 'collection', 'repository', 'benchmark']
sentence_expansion = True
training_cycles = 2
model_name = 'dataset'
filtering_pmi = True
filtering_st = True
filtering_ws = True
filtering_kbl = True
filtering_majority = True


for cycle in range(training_cycles):
    seed_data_extraction.sentence_extraction(model_name, cycle, seeds)
    expansion.term_expansion(model_name, cycle)
    expansion.sentence_expansion(model_name, cycle, doc2vec_model)
    training_data_generation.sentence_labelling(model_name, cycle, sentence_expansion)
    ner_training.create_prop(model_name, cycle, sentence_expansion)
    ner_training.train_model(model_name, cycle)
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