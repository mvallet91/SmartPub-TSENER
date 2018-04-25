"""
This script generates training data and trains NER using different number of seeds and approach.
The training data should be extracted using the training_data_extraction.py  using the seed terms
(done with different number of seeds 10 times) and be stored in the "evaluation_files folder".
"""

from preprocessing import ner_training, expansion, training_data_extraction
from postprocessing import trainingdata_generation, extract_new_entities, filtering
from config import ROOTPATH
import gensim
import elasticsearch

model_doc2vec = gensim.models.Doc2Vec.load(ROOTPATH + '/models/doc2vec.model')
es = elasticsearch.Elasticsearch([{'host': 'localhost', 'port': 9200}])

seeds = [5, 25, 100]

"""
Extract training data for different number of seeds
"""
for seed in seeds:
    training_data_extraction.extract(seed)

"""
Term expansion approach for the first iteration
"""
# perform term expansion on the text of the training data using different number of seeds (i.e. 5,10,25,50,100)

# for number in range(10):
#     expansion.term_expansion_proteins(5, 'term_expansion', str(0), str(number))
#     expansion.term_expansion_proteins(10, 'term_expansion', str(0), str(number))
#     expansion.term_expansion_proteins(25, 'term_expansion', str(0), str(number))
#     expansion.term_expansion_proteins(50, 'term_expansion', str(0), str(number))
#     expansion.term_expansion_proteins(100, 'term_expansion', str(0), str(number))

# # training data generation
# for number in range(10):
#     trainingdata_generation.generate_trainingTE(5, 'term_expansion', str(0), str(number))
#     trainingdata_generation.generate_trainingTE(10, 'term_expansion', str(0), str(number))
#     trainingdata_generation.generate_trainingTE(25, 'term_expansion', str(0), str(number))
#     trainingdata_generation.generate_trainingTE(50, 'term_expansion', str(0), str(number))
#     trainingdata_generation.generate_trainingTE(100, 'term_expansion', str(0), str(number))

# # training the NER model which will be saved in the crf_trained_files folder
# ner_training.create_austenprop(5, 'term_expansion', str(0))
# ner_training.create_austenprop(10, 'term_expansion', str(0))
# ner_training.create_austenprop(25, 'term_expansion', str(0))
# ner_training.create_austenprop(50, 'term_expansion', str(0))
# ner_training.create_austenprop(100, 'term_expansion', str(0))

# #
# ner_training.train(5, 'term_expansion', str(0))
# ner_training.train(10, 'term_expansion', str(0))
# ner_training.train(25, 'term_expansion', str(0))
# ner_training.train(50, 'term_expansion', str(0))
# ner_training.train(100, 'term_expansion', str(0))

########################################################

# # An example for 5 iterations:
# for i in range(5):
#     extract_new_entities.ne_extraction(100, 'term_expansion', i, i + 1, 0, es)
#     filtering.PMI(100, 'term_expansion', i, 0)
#     trainingdata_generation.extract(100, 'term_expansion', i, 0)
#     expansion.term_expansion_protein(100, 'term_expansion', i, 0)
#     trainingdata_generation.generate_trainingTE(100, 'term_expansion', i, 0)
#     ner_training.create_austenprop(100, 'term_expansion', 0)
#     ner_training.train(100, 'term_expansion', 0)

########################################################


# """
# Sentence approach for the first iteration
# """
for number in range(0, 10):
    expansion.term_expansion_proteins(5, 'sentence_expansion_mix', str(0), str(number))
    expansion.term_expansion_proteins(25, 'sentence_expansion_mix', str(0), str(number))
    expansion.term_expansion_proteins(100, 'sentence_expansion_mix', str(0), str(0))

for number in range(0, 10):
    trainingdata_generation.generate_trainingSE(5, 'sentence_expansion_mix', str(0), str(number), model_doc2vec)
    trainingdata_generation.generate_trainingSE(25, 'sentence_expansion_mix', str(0), str(number), model_doc2vec)
    trainingdata_generation.generate_trainingSE(100, 'sentence_expansion_mix', str(0), str(0), model_doc2vec)

ner_training.create_austenprop(5, 'sentence_expansion_mix', str(0))
ner_training.create_austenprop(25, 'sentence_expansion_mix', str(0))
ner_training.create_austenprop(100, 'sentence_expansion_mix', str(0))

ner_training.train(5, 'sentence_expansion_mix', str(0))
ner_training.train(25, 'sentence_expansion_mix', str(0))
ner_training.train(100, 'sentence_expansion_mix', str(0))

########################################################
# ner_training.create_austenprop(5, 'sentence_expansion', str(1))
# ner_training.train(5, 'sentence_expansion', str(1))
# An example for 5 iterations:

###TESTING
# expansion.term_expansion_proteins(5, 'sentence_expansion_PROT', str(0), str(0))
# trainingdata_generation.generate_trainingSE(5, 'sentence_expansion_PROT', str(0), str(0), model_doc2vec)
# ner_training.create_austenprop(5, 'sentence_expansion_PROT', str(0))
# ner_training.train(5, 'sentence_expansion_PROT', str(0))

for i in range(0, 3):
    extract_new_entities.ne_extraction(5, 'sentence_expansion_bigrams', str(i), str(i+1), str(0), es)
    # filtering.PMI(100, 'sentence_expansion', i, 0)
    filtering.WordNet_StopWord(5, 'sentence_expansion_bigrams', str(i+1), str(0))
    trainingdata_generation.extract(5, 'sentence_expansion_bigrams', str(i+1), str(0))
    expansion.term_expansion_proteins(5, 'sentence_expansion_bigrams', str(i+1), str(0))
    trainingdata_generation.generate_trainingSE(5, 'sentence_expansion_bigrams', str(i+1), str(0), model_doc2vec)
    ner_training.create_austenprop(5, 'sentence_expansion_bigrams', str(i+1))
    ner_training.train(5, 'sentence_expansion_bigrams', str(i+1))
    print('Finished with iter', i)
    print('#' * 1000)

# for i in range(0,3):
#     extract_new_entities.ne_extraction(5, 'sentence_expansion_PROT', str(i), str(i + 1), str(0), es)
#     #filtering.PMI(100, 'sentence_expansion', i, 0)
#     filtering.WordNet_StopWord(5, 'sentence_expansion_PROT', str(i+1), str(0)) 
#     trainingdata_generation.extract(5, 'sentence_expansion_PROT', str(i+1), str(0))
#     expansion.term_expansion_proteins(5, 'sentence_expansion_PROT', str(i+1), str(0))
#     trainingdata_generation.generate_trainingSE(5, 'sentence_expansion_PROT', str(i+1), str(0), model_doc2vec)
#     ner_training.create_austenprop(5, 'sentence_expansion_PROT', str(i+1))
#     ner_training.train(5, 'sentence_expansion_PROT', str(i+1))

########################################################
