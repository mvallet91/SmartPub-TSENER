import codecs
import numpy
import sys
import nltk
import gensim

from numbers import Number
from elasticsearch import Elasticsearch
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import tokenize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config import ROOTPATH
from preprocessing.generic_entity_extraction import generic_named_entities

es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

class autovivify_list(dict):
    """
    Pickleable class to replicate the functionality of collections.defaultdict
    """

    def __missing__(self, key):
        value = self[key] = []
        return value

    def __add__(self, x):
        """Override addition for numeric types when self is empty"""
        if not self and isinstance(x, Number):
            return x
        raise ValueError

    def __sub__(self, x):
        """Also provide subtraction method"""
        if not self and isinstance(x, Number):
            return -1 * x
        raise ValueError


def build_word_vector_matrix(vector_file, proper_nouns, model_name):
    """
    Read a GloVe array from sys.argv[1] and return its vectors and labels as arrays
    """
    numpy_arrays = []
    labels_array = []
    with codecs.open(vector_file, 'r', 'utf-8') as f:
        for c, r in enumerate(f):
            sr = r.split()
            try:
                if sr[0] in proper_nouns and not wordnet.synsets(sr[0]) and sr[0].lower() not in stopwords.words(
                        'english') and model_name not in sr[0].lower():
                    labels_array.append(sr[0])
                    numpy_arrays.append(numpy.array([float(i) for i in sr[1:]]))
            except:
                continue
    return numpy.array(numpy_arrays), labels_array


def find_word_clusters(labels_array, cluster_labels):
    """
    Read the labels array and clusters label and return the set of words in each cluster
    """
    cluster_to_words = autovivify_list()
    for c, i in enumerate(cluster_labels):
        cluster_to_words[i].append(labels_array[c])
    return cluster_to_words


def term_expansion(model_name: str, training_cycle: int) -> None:
    """
    :param model_name:
    :type model_name:
    :param training_cycle:
    :type training_cycle:
    """
    print('Starting term expansion')
    unlabelled_sentences_file = (ROOTPATH + '/processing_files/' + model_name + '_sentences_' +
                                 str(training_cycle) + '.txt')
    all_entities = generic_named_entities(unlabelled_sentences_file)
    seed_entities = []

    # Extract seed entities
    path = ROOTPATH + '/processing_files/' + model_name + '_seeds_' + str(training_cycle) + '.txt'
    with open(path, 'r', encoding='utf-8') as file:
        for row in file.readlines():
            seed_entities.append(row.strip())
            all_entities.append(row.strip())
    seed_entities = [e.lower() for e in seed_entities]

    # Replace the space between the bigram words with underscore _ (for the word2vec embedding)
    processed_entities = []
    for pp in all_entities:
        temp = pp.split(' ')
        if len(temp) > 1:
            bigram = list(nltk.bigrams(pp.split()))
            for bi in bigram:
                bi = bi[0].lower() + '_' + bi[1].lower()
                processed_entities.append(bi)
        else:
            processed_entities.append(pp)
    processed_entities = [e.lower() for e in processed_entities]
    processed_entities = list(set(processed_entities))

    # Use the word2vec model
    df, labels_array = build_word_vector_matrix(ROOTPATH + '/models/modelword2vecbigram.vec',
                                                processed_entities, model_name)

    # We cluster all terms extracted from the sentences with respect to their embedding vectors using K-means.
    # Silhouette analysis is used to find the optimal number k of clusters. Finally, clusters that contain
    # at least one of the seed terms are considered to (only) contain entities the same type (e.g dataset).
    expanded_terms = []
    max_cluster = 0
    if len(df) >= 9:
        print('Started term clustering')
        for n_clusters in range(2, 10):
            df = StandardScaler().fit_transform(df)
            kmeans_model = KMeans(n_clusters=n_clusters, max_iter=300, n_init=100)
            kmeans_model.fit(df)
            cluster_labels = kmeans_model.labels_
            cluster_to_words = find_word_clusters(labels_array, cluster_labels)
            cluster_labels = kmeans_model.fit_predict(df)

            final_list = []
            for c in cluster_to_words:
                counter = dict()
                for word in cluster_to_words[c]:
                    counter[word] = 0
                for word in cluster_to_words[c]:
                    if word in seed_entities:
                        for ww in cluster_to_words[c]:
                            final_list.append(ww.replace('_', ' '))
            try:
                silhouette_avg = silhouette_score(df, cluster_labels)
                if silhouette_avg > max_cluster:
                    max_cluster = silhouette_avg
                    expanded_terms = final_list
            except:
                continue
    path = (ROOTPATH + '/processing_files/' + model_name + '_expanded_seeds_' + str(training_cycle) + '.txt')
    f = open(path, 'w', encoding='utf-8')
    for item in expanded_terms:
        f.write("%s\n" % item)
    print('Added', len(expanded_terms), 'expanded terms')


def extract_similar_sentences(es_id):
    """
    Function for finding similar sentences given the code of a sentence (everything is stored in elasticsearch)
    """
    query = {"query":
                 {"match":
                      {"_id":
                           {"query": es_id,
                            "operator": "and"
                           }
                      }
                 }
             }

    res = es.search(index="devtwosentnew", doc_type="devtwosentnorulesnew",
                    body=query, size=1)
    for doc in res['hits']['hits']:
        similar_sentence = doc['_source']['content.chapter.sentpositive']

    return similar_sentence


def sentence_expansion(model_name: str, training_cycle: int, doc2vec_model: gensim.models.doc2vec.Doc2Vec) -> None:
    """

    :param model_name:
    :param training_cycle:
    :param doc2vec_model:
    """
    print('Starting sentence expansion')

    path = ROOTPATH + '/processing_files/' + model_name + '_sentences_' + str(training_cycle) + '.txt'
    unlabelled_sentences_file = open(path, 'r', encoding='utf-8')
    text = unlabelled_sentences_file.read()
    text = text.replace('\\', '')
    text = text.replace('/', '')
    text = text.replace('"', '')
    text = text.replace('(', '')
    text = text.replace(')', '')
    text = text.replace('[', '')
    text = text.replace(']', '')
    text = text.replace(',', ' ,')
    text = text.replace('?', ' ?')
    text = text.replace('..', '.')

    sentences = (tokenize.sent_tokenize(text.strip()))
    sentences = list(set(sentences))
    print('Finding similar sentences')
    temp = []
    for i, line in enumerate(sentences):
        tokens = line.split()
        new_vector = doc2vec_model.infer_vector(tokens)
        sims = doc2vec_model.docvecs.most_similar([new_vector], topn=1)
        if sims:
            for ss in sims:
                if ss[1] > 0.50:
                    temp.append(extract_similar_sentences(str(ss[0])))
        if i % 1000 == 0:
            print('.', end='')

    sentences = list(set(sentences))
    temp = list(set(temp))
    print('Added', len(temp), 'expanded sentences to the', len(sentences), 'original')
    for tt in temp:
        sentences.append(tt)
    expanded_sentences = list(set(sentences))

    path = (ROOTPATH + '/processing_files/' + model_name + '_expanded_sentences_' + str(training_cycle) + '.txt')
    f = open(path, 'w', encoding='utf-8')
    for item in expanded_sentences:
        f.write('%s\n' % item)
    f.close()
