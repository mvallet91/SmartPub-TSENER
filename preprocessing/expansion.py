import codecs
import numpy
import sys
from numbers import Number

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config import ROOTPATH
from preprocessing.generic_entity_extraction import generic_named_entities


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


def term_expansion(model_name, training_cycle):
    """
    :param model_name:
    :type model_name:
    :param training_cycle:
    :type training_cycle:
    """
    # In the first cycle use the initial text extracted using the seeds
    if int(training_cycle) == 0:
        unlabelled_sentences_file = (ROOTPATH + '/processing_files/sentences_from_seeds.txt')

    # In the next iterations use the text extracted using the new set of seeds
    else:
        unlabelled_sentences_file = (ROOTPATH + '/processing_files/sentences_from_cycle' + str(training_cycle) + '.txt')

    all_entities = generic_named_entities(unlabelled_sentences_file)
    seed_entities = []

    # Extract seed entities
    path = ROOTPATH + '/processing_files/seeds.txt'
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

    max_cluster = 0
    if len(df) >= 9:
        print('started clustering')
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
                        print('.', end='')
                        sys.stdout.flush()
            try:
                silhouette_avg = silhouette_score(df, cluster_labels)
                if silhouette_avg > max_cluster:
                    max_cluster = silhouette_avg
                    f = open(ROOTPATH + '/processing_files/expanded_seeds_' + model_name + '_'
                             + str(training_cycle) + '.txt', 'w', encoding='utf-8')
                    for item in final_list:
                        f.write("%s\n" % item)
                    print('added', len(final_list), 'expanded terms')
            except:
                continue
