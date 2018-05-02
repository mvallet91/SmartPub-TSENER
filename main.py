from preprocessing import ner_training, expansion, seed_data_extraction
from postprocessing import trainingdata_generation, extract_new_entities, filtering
import config as cfg
import gensim
import elasticsearch

model_doc2vec = gensim.models.Doc2Vec.load(cfg.ROOTPATH + '/models/doc2vec.model')
es = elasticsearch.Elasticsearch([{'host': 'localhost', 'port': 9200}])

# User input
seeds = ['clueweb', 'imagenet', 'flickr', 'webkb', 'netflix', 'imdb']
sentence_expansion = True
training_cycles = 5
model_name = 'dataset'

# "Extract training data using seeds provided

seed_data_extraction.sentence_extraction(seeds)

if not sentence_expansion:
    for cycle in range(training_cycles):
        seed_data_extraction.sentence_extraction(seeds)
        expansion.term_expansion(model_name, cycle)
        trainingdata_generation.label_sentences_term_expansion_only(model_name, cycle)
        ner_training.create_austenprop(100, 'term_expansion', 0)
        ner_training.train(100, 'term_expansion', 0)
        extract_new_entities.ne_extraction(100, 'term_expansion', i, i + 1, 0, es)
        filtering.PMI(100, 'term_expansion', i, 0)

else:
    for cycle in range(training_cycles):
        seed_data_extraction.sentence_extraction(seeds)
        expansion.term_expansion(model_name, cycle)
        trainingdata_generation.label_sentences_term_sentence_expansion(model_name, cycle, model_doc2vec)
        ner_training.create_austenprop(100, 'term_expansion', 0)
        ner_training.train(100, 'term_expansion', 0)
        extract_new_entities.ne_extraction(100, 'term_expansion', i, i + 1, 0, es)
        filtering.PMI(100, 'term_expansion', i, 0)

