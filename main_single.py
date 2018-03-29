
"""
This scripts generates training data and trains NER using different number of seeds and approach.
The training data should be extracted using the training_data_extraction.py  using the seed terms (done with different number of seeds 10 times)
and be stored in the "evaluation_files folder".
"""

from preprocessing import ner_training, expansion, training_data_extraction
from postprocessing import trainingdata_generation, extract_new_entities, filtering
from config import ROOTPATH
from gensim.models import Doc2Vec
from elasticsearch import Elasticsearch

modeldoc2vec = Doc2Vec.load(ROOTPATH + '/models/doc2vec.model')
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

seeds = [5]#, 10, 25, 50, 100]

"""
Extract training data for different number of seeds
"""
for seed in seeds:
    training_data_extraction.extract(seed)

"""
Term expansion approach for the first iteration
"""
# perform term expansion on the text of the training data using different number of seeds (i.e. 5,10,25,50,100)
for number in range(10):
    expansion.term_expansion_proteins(5, 'term_expansion', str(0), str(number))

# training data generation
for number in range(10):
    trainingdata_generation.generate_trainingTE(5, 'term_expansion', str(0), str(number))

# training the NER model which will be saved in the crf_trained_files folder
ner_training.create_austenprop(5, 'term_expansion', str(0))

#
ner_training.train(5, 'term_expansion', str(0))
