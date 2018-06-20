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


def generic_named_entities(file_path):
    """
    Obtains the generic entities from the sentences provided. This is because for the expansion strategies
    we only consider terms terms which are likely to be named entities by using NLTK entity detection, instead
    of all the words in the sentences.
    :param file_path:
    :return:
    """
    unlabelled_sentence_file = open(file_path, 'r', encoding='utf-8')
    text = unlabelled_sentence_file.read()
    print('Started to extract generic named entity from sentences...')
    sentences = nltk.sent_tokenize(text)
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
    chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)
    entity_names = []
    x = 0
    for tree in chunked_sentences:
        entity_names.extend(extract_entity_word(tree))
        x+=1
        if x % 1000 == 0:
            print('.', end='')
    print('Finished processing sentences with', len(entity_names), 'new possible entities')
    return entity_names


def extract_entity_word(t):
    """
    Recursively goes through the branches of the NLTK tagged sentences to extract the words tagged as entities
    :param t: NLTK tagged tree
    :return entity_names: a list of unique entity tokens
    """
    entity_names = []
    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_word(child))
    return set(entity_names)


def term_expansion(model_name: str, training_cycle: int, wordvector_path: str) -> None:
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
    df, labels_array = build_word_vector_matrix(wordvector_path, processed_entities, model_name)

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
    expanded_terms = list(set(expanded_terms))
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
    similar_sentence = ''
    res = es.search(index="devtwosentnew_tud", doc_type="devtwosentnorulesnew",
                    body=query, size=5)
    if len(res) > 1:
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
    print('Finding similar sentences to the', len(sentences), 'starting sentences')
    temp = []
    x = 0
    for i, line in enumerate(sentences):
        if x % 1000 == 0:
            print('.', end='')
        x += 1
        sys.stdout.flush()
        tokens = line.split()
        new_vector = doc2vec_model.infer_vector(tokens)
        sims = doc2vec_model.docvecs.most_similar([new_vector], topn=1)
        if sims:
            for ss in sims:
                if ss[1] > 0.50:
                    similar = extract_similar_sentences(str(ss[0]))
                    if len(similar) > 1:
                        temp.append(similar)

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
