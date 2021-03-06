from m1_preprocessing import seed_data_extraction, term_sentence_expansion, training_data_generation, ner_training
from m1_postprocessing import extract_new_entities, filtering
import config as cfg
import gensim
import elasticsearch
import time

doc2vec_model = gensim.models.Doc2Vec.load(cfg.ROOTPATH + '/embedding_models/doc2vec.model')
word2vec_path = 'full_corpus_models/modelword2vecbigram.vec'
es = elasticsearch.Elasticsearch([{'host': 'localhost', 'port': 9200}])

# User input
seeds = ['mediaeval', 'melvyl', 'jnlpba', 'wikia', 'wechat', 'wt2g', 'foldoc', 'openstreetmap', 'nasdaq', 'datahub', 'github', 'linkedgeodata', 'amazon', 'chemid', 'wikipedia', 'citebase', 'bdbcomp', 'ace04', 'gene ontology', 'movielens', 'stack overflow', 'oaister', 'umls', 'booking.com', 'duc2001', 'movierating', 'mesur', 'scovo', 'semeval', 'newsvine', 'google', 'jester', 'robust04', 'wpbench', 'cifar', 'pubchem', 'sraa', 'blog06', 'euses', 'gdelt', 'sindice', 'fedweb', 'ask.com', 'ldoce', 'ocred', 'tripadvisor', 'locuslink', 'imagenet', 'walmart', 'technorati', 'shop.com', 'quora', 'orkut', 'reddit', 'allmusic', 'dbsnp', 'ionosphere', 'letor', 'blogpulse', 'spambase', 'fbis', 'webkb', 'craigslist', 'gerbil', 'tac kbp', 'citeseerx', 'replab', 'pslnl', 'douban', 'new york times', 'algoviz', 'friendfeed', 'labelme', 'newegg', 'brightkite', 'econstor', 'dmoz.org', 'billion triple challenge', 'wikiwars', 'lexvo', 'worldcat', 'ws-353', 'facc', 'as3ap', 'moviepilot', 'refseq', 'tpc-w', 'ratebeer', 'specweb', 'wikitravel', 'twitter', 'buzzfeed', 'imageclef', 'pinterest', 'dailymed', 'kddcup', 'semcor', 'xanga', 'netflix', 'cyworld']
context_words = ['dataset', 'corpus', 'collection', 'repository', 'benchmark']
model_name = 'dataset_tud_full'


# seeds = ['simulated annealing', 'hidden markov models', 'scoring engine', 'linear regression', 'genetic programming', 'kolmogorov-smirnov ks', 'qald', 'downhill simplex', 'regular expression', 'radial basis function network', 'recurrent neural network', 'restricted boltzmann machine', 'best-first search', 'pairwise personalized ranking', 'pairwise algorithm', 'convolutional neural network', 'selection algorithm', 'imrank', 'statistical relational', 'adarank', 'similarity search', 'game theory', 'breadth-first search', 'traditional materialized view', 'self-organizing map', 'mcmc', 'convolutional dnn', 'fourier analysis', 'tree sort', 'linkage analysis', 'pearson correlation', 'q-learning', 'dijkstra', 'cyclades', 'fuzzy clustering', 'bayesian nonparametric', 'latent semantic analysis', 'fast fourier', 'general interest model', 'clarke-tax', 'yield optimization', 'dmp method', 'query expansion', 'spectral clustering', 'transfer function', 'recursive function', 'rapid7', 'random forest', 'quicksort', 'imputation', 'hill climbing', 'likelihood function', 'dynamic programming', 'random indexing', 'skipgram', 'predictive modeling', 'deep learning', 'semantictyper', 'global collaborative ranking', 'bcdrw', 'space mapping', 'shannon entropy', 'ridge regularization', 'tagassist', 'lib*lif', 'lib+lif', 'model fitting', 'graph-based propagation', 'lstm', 'autoencoder', 'linear search', 'dbscan', 'stack search', 'folding-in', 'jump search', 'plsa', 'clir', 'random search', 'a* search', 'block sort', 'basic load control method', 'klsh', 'pattern matching', 'support vector machine', 'gpbased', 'merge sort', 'optics algorithm', 'query optimization', 'hierarchical agglomerative', 'k-means++', 'tdcm', 'semantic relevance', 'stochastic gradient descent', 'tsa algorithm', 'adaptive filter', 'genetic algorithm', 'greedy algorithm', 'kernel density estimation', 'simple random algorithm', 'ndcg-annealing']
# context_words = ['method', 'algorithm', 'approach', 'evaluate']
# model_name = 'method_tud_full'


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
    term_sentence_expansion.term_expansion(model_name, cycle, word2vec_path)
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
        filtering.filter_st(model_name, cycle, seeds, word2vec_path)
    if filtering_ws:
        filtering.filter_ws(model_name, cycle)
    if filtering_kbl:
        filtering.filter_kbl(model_name, cycle, seeds)
    filtering.majority_vote(model_name, cycle)
    print(round((time.time() - start)/60, 2), 'minutes since start')
    print('-'*50)
    print('')
