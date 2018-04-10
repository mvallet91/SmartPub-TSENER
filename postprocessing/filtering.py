"""
This script will be used to filter the noisy extracted entities.
"""
from numbers import Number
from sklearn.preprocessing import StandardScaler
import codecs, numpy
from sklearn.metrics import silhouette_score
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import string
import gensim
from sklearn.cluster import KMeans
from config import ROOTPATH
import nltk
import requests
from xml.etree import ElementTree
from postprocessing import normalized_pub_distance
corpuspath = ROOTPATH + "/data/stopword_en.txt"
stopwordList=[]

with open(corpuspath, "r") as file:
        for row in file.readlines():
            stopwordList.append(row.strip())

class autovivify_list(dict):
    '''Pickleable class to replicate the functionality of collections.defaultdict'''

    def __missing__(self, key):
        value = self[key] = []
        return value

    def __add__(self, x):
        '''Override addition for numeric types when self is empty'''
        if not self and isinstance(x, Number):
            return x
        raise ValueError

    def __sub__(self, x):
        '''Also provide subtraction method'''
        if not self and isinstance(x, Number):
            return -1 * x
        raise ValueError


def build_word_vector_matrix(vector_file, propernouns):
    '''Read a GloVe array from sys.argv[1] and return its vectors and labels as arrays'''
    numpy_arrays = []
    labels_array = []
    with codecs.open(vector_file, 'r', 'utf-8') as f:
        for c, r in enumerate(f):
            sr = r.split()

            try:
                if sr[0] in propernouns and not wordnet.synsets(sr[0]) and sr[0].lower() not in stopwords.words(
                        'english'):
                    labels_array.append(sr[0])
                    # print(sr[0].strip())

                    numpy_arrays.append(numpy.array([float(i) for i in sr[1:]]))
            except:
                continue

    return numpy.array(numpy_arrays), labels_array


def find_word_clusters(labels_array, cluster_labels):
    '''Read the labels array and clusters label and return the set of words in each cluster'''
    cluster_to_words = autovivify_list()
    for c, i in enumerate(cluster_labels):
        cluster_to_words[i].append(labels_array[c])
    return cluster_to_words


"""
Majority vote filtering
"""


def majorityVote(result):
    finalresult = []
    print(len(result))
    result = list(set(result))
    print(len(result))
    for rr in result:

        count = 0
        # check if in DBpedia
        url = 'http://lookup.dbpedia.org/api/search/KeywordSearch?QueryClass=&QueryString=' + str(rr)
        try:

            resp = requests.request('GET', url)
            root = ElementTree.fromstring(resp.content)
            check_if_exist = []
            for child in root.iter('*'):
                check_if_exist.append(child)
            if len(check_if_exist) == 1:
                count = count + 1
        except:
            pass

        # check if in wordnet or stopword
        if not wordnet.synsets(rr) and rr.lower() not in stopwords.words('english'):
            count = count + 1
        temp = []

        # check PMI
        temp.append(rr)
        temp = normalized_pub_distance.NPD(temp)
        if temp:
            count = count + 1

        if count > 1:
            finalresult.append(rr)
    return finalresult


"""
Embedding clustering filtering
"""


def ec_clustering(numberOfSeeds, name, numberOfIteration, iteration):
    print('started embeding ranking....', numberOfSeeds, name, iteration)
    propernouns = []

    # read the extracted entities from the file
    path = ROOTHPATH + '/post_processing_files/' + name + '_Iteration' + numberOfIteration + str(
        numberOfSeeds) + '_' + str(iteration) + '.txt'
    with open(path, "r") as file:
        for row in file.readlines():
            propernouns.append(row.strip())
    dsnames = []
    dsnamestemp = []

    # read the seed terms
    corpuspath = ROOTHPATH + '/evaluation_files/X_Seeds_' + str(
        numberOfSeeds) + '_' + str(iteration) + '.txt'

    with open(corpuspath, "r") as file:
        for row in file.readlines():
            dsnames.append(row.strip())
            propernouns.append(row.strip())

    # read the new seed terms (if exist)
    for i in range(1, int(numberOfIteration) + 1):
        try:
            with open(ROOTHPATH + '/evaluation_files/' + name + '_Iteration' + str(i) + '_POS_' + str(
                    numberOfSeeds) + '_' + str(iteration) + '.txt', 'r') as file:
                for row in file.readlines():
                    dsnames.append(row.strip())
                    propernouns.append(row.strip())
        except:
            continue

    newpropernouns = []
    for pp in propernouns:
        temp = pp.split(' ')
        if len(temp) > 1:
            bigrams = list(nltk.bigrams(pp.split()))
            for bi in bigrams:
                aa = bi[0].translate(str.maketrans('', '', string.punctuation))
                bb = bi[1].translate(str.maketrans('', '', string.punctuation))
                bi = aa.lower() + '_' + bb.lower()

                newpropernouns.append(bi)
        else:
            newpropernouns.append(pp)

    dsnames = [x.lower() for x in dsnames]
    dsnames = [s.translate(str.maketrans('', '', string.punctuation)) for s in dsnames]

    sentences_split = [s.lower() for s in newpropernouns]

    df, labels_array = build_word_vector_matrix(
        ROOTHPATH + "/models/modelword2vecbigram.txt", sentences_split)

    sse = {}
    maxcluster = 0
    if len(df) >= 9:
        for n_clusters in range(2, 10):
            finallist = []

            df = StandardScaler().fit_transform(df)
            kmeans_model = KMeans(n_clusters=n_clusters, max_iter=300, n_init=100)
            kmeans_model.fit(df)
            cluster_labels = kmeans_model.labels_

            cluster_to_words = find_word_clusters(labels_array, cluster_labels)
            cluster_labelss = kmeans_model.fit_predict(df)
            sse[n_clusters] = kmeans_model.inertia_

            for c in cluster_to_words:
                counter = dict()
                for word in cluster_to_words[c]:
                    counter[word] = 0
                for word in cluster_to_words[c]:
                    if word in dsnames:

                        for ww in cluster_to_words[c]:
                            finallist.append(ww.replace('_', ' '))

            try:

                silhouette_avg = silhouette_score(df, cluster_labelss)
                print("For n_clusters =", n_clusters,
                      "The average silhouette_score is :", silhouette_avg)
                if silhouette_avg > maxcluster:
                    maxcluster = silhouette_avg
                    thefile = open(
                        ROOTHPATH + "/evaluation_files/" + name + "_Iteration" + numberOfIteration + "_POS_" + str(
                            numberOfSeeds) + "_" + str(iteration) + ".txt", 'w')
                    finallist = list(set(finallist))

                    for item in finallist:
                        if item.lower() not in dsnamestemp and item.lower() not in dsnames:
                            thefile.write("%s\n" % item)

            except:
                print("ERROR:::Silhoute score invalid")
                continue
    else:
        for n_clusters in range(2, len(df)):
            finallist = []

            df = StandardScaler().fit_transform(df)
            kmeans_model = KMeans(n_clusters=n_clusters, max_iter=300, n_init=100)
            kmeans_model.fit(df)
            cluster_labels = kmeans_model.labels_

            cluster_to_words = find_word_clusters(labels_array, cluster_labels)
            cluster_labelss = kmeans_model.fit_predict(df)
            sse[n_clusters] = kmeans_model.inertia_

            for c in cluster_to_words:
                print(cluster_to_words[c])

            for c in cluster_to_words:
                counter = dict()
                for word in cluster_to_words[c]:
                    counter[word] = 0
                for word in cluster_to_words[c]:
                    if word in dsnames:

                        for ww in cluster_to_words[c]:
                            if ww not in dsnames:
                                finallist.append(ww.replace('_', ' '))
            try:

                silhouette_avg = silhouette_score(df, cluster_labelss)
                print("For n_clusters =", n_clusters,
                      "The average silhouette_score is :", silhouette_avg)
                if silhouette_avg > maxcluster:
                    maxcluster = silhouette_avg
                    thefile = open(
                        ROOTHPATH + "/evaluation_files/" + name + "_Iteration" + numberOfIteration + "_POS_" + str(
                            numberOfSeeds) + "_" + str(iteration) + ".txt", 'w')
                    finallist = list(set(finallist))

                    for item in finallist:
                        if item.lower() not in dsnamestemp and item.lower() not in dsnames:
                            thefile.write("%s\n" % item)

            except:
                print("ERROR:::Silhoute score invalid")
                continue


"""
Knowledgebase look-up + EC filtering
"""


def Kb_ecall(numberOfSeeds, name, numberOfIteration, iteration):
    # for iteration in range(0,10):
    propernouns = []
    print('filteriiingg....' + str(numberOfSeeds) + '_' + str(name) + '_' + str(iteration))

    path = ROOTHPATH + '/post_processing_files/' + name + '_Iteration' + str(numberOfIteration) + str(
        numberOfSeeds) + '_' + str(iteration) + '.txt'
    print(path)
    with open(path, "r") as file:
        for row in file.readlines():
            propernouns.append(row.strip())

    dsnames = []

    corpuspath = ROOTHPATH + '/evaluation_files/X_Seeds_' + str(
        numberOfSeeds) + '_' + str(iteration) + '.txt'

    with open(corpuspath, "r") as file:
        for row in file.readlines():
            dsnames.append(row.strip())
            propernouns.append(row.strip())
    for i in range(1, int(numberOfIteration)):
        try:
            with open(ROOTHPATH + '/evaluation_files/' + name + '_Iteration' + str(i) + '_POS_' + str(
                    numberOfSeeds) + '_' + str(iteration) + '.txt', 'r') as file:
                for row in file.readlines():
                    dsnames.append(row.strip())
                    propernouns.append(row.strip())
        except:
            continue
    newpropernouns = []
    bigrams = []
    for pp in propernouns:
        temp = pp.split(' ')
        if len(temp) > 1:
            bigram = list(nltk.bigrams(pp.split()))

            for bi in bigram:
                bi = bi[0].lower() + '_' + bi[1].lower()
                # print(bi)
                newpropernouns.append(bi)
        else:
            newpropernouns.append(pp)

    dsnames = [x.lower() for x in dsnames]
    dsnamestemp = [s.translate(str.maketrans('', '', string.punctuation)) for s in dsnames]
    finalds = []
    for ds in dsnamestemp:
        dss = ds.split(' ')
        if len(dss) > 1:
            ds = ds.replace(' ', '_')
        finalds.append(ds)

    sentences_split = [s.lower() for s in newpropernouns]

    sentences_split = [s.replace('"', '') for s in sentences_split]

    df, labels_array = build_word_vector_matrix(
        ROOTHPATH + "/models/modelword2vecbigram.txt", sentences_split)

    sse = {}
    maxcluster = 0
    if len(df) >= 9:
        for n_clusters in range(2, 10):
            finallist = []

            df = StandardScaler().fit_transform(df)
            kmeans_model = KMeans(n_clusters=n_clusters, max_iter=300, n_init=100)
            kmeans_model.fit(df)
            cluster_labels = kmeans_model.labels_
            cluster_to_words = find_word_clusters(labels_array, cluster_labels)
            cluster_labelss = kmeans_model.fit_predict(df)
            sse[n_clusters] = kmeans_model.inertia_

            # print("\n")

            for c in cluster_to_words:
                counter = dict()
                dscounter = 0
                for word in cluster_to_words[c]:
                    counter[word] = 0
                for word in cluster_to_words[c]:
                    if word in finalds:

                        for ww in cluster_to_words[c]:
                            finallist.append(ww.replace('_', ' '))

            # print(finallist)
            try:

                silhouette_avg = silhouette_score(df, cluster_labelss)

                if silhouette_avg > maxcluster:
                    maxcluster = silhouette_avg
                    thefile = open(
                        ROOTHPATH + "/evaluation_files/" + name + "_Iteration" + str(
                            numberOfIteration) + "_POS_" + str(
                            numberOfSeeds) + "_" + str(iteration) + ".txt", 'w')
                    finallist = list(set(finallist))

                    for item in finallist:
                        if item.lower() not in dsnamestemp and item.lower() not in dsnames:
                            thefile.write("%s\n" % item)
                    for item in bigrams:
                        if item.lower() not in dsnamestemp and item.lower() not in dsnames:
                            thefile.write("%s\n" % item)
                    thefile.close()
            except:
                print("ERROR:::Silhoute score invalid")
                continue


    else:
        for n_clusters in range(2, len(df)):
            finallist = []

            df = StandardScaler().fit_transform(df)
            kmeans_model = KMeans(n_clusters=n_clusters, max_iter=300, n_init=100)
            kmeans_model.fit(df)
            cluster_labels = kmeans_model.labels_
            cluster_to_words = find_word_clusters(labels_array, cluster_labels)
            cluster_labelss = kmeans_model.fit_predict(df)
            sse[n_clusters] = kmeans_model.inertia_

            # print("\n")

            for c in cluster_to_words:
                counter = dict()
                dscounter = 0
                for word in cluster_to_words[c]:
                    counter[word] = 0
                for word in cluster_to_words[c]:
                    if word in finalds:

                        for ww in cluster_to_words[c]:
                            url = 'http://lookup.dbpedia.org/api/search/KeywordSearch?QueryClass=place&QueryString=' + str(
                                ww)
                            resp = requests.request('GET', url)
                            root = ElementTree.fromstring(resp.content)
                            check_if_exist = []
                            for child in root.iter('*'):
                                check_if_exist.append(child)
                            if len(check_if_exist) == 1:
                                finallist.append(ww.replace('_', ' '))

            try:

                silhouette_avg = silhouette_score(df, cluster_labelss)

                if silhouette_avg > maxcluster:
                    maxcluster = silhouette_avg
                    thefile = open(
                        ROOTHPATH + "/evaluation_files/" + name + "_Iteration" + numberOfIteration + "_POS_" + str(
                            numberOfSeeds) + "_" + str(iteration) + ".txt", 'w')
                    finallist = list(set(finallist))

                    for item in finallist:
                        if item.lower() not in dsnamestemp and item.lower() not in dsnames:
                            thefile.write("%s\n" % item)
                    for item in bigrams:
                        if item.lower() not in dsnamestemp and item.lower() not in dsnames:
                            thefile.write("%s\n" % item)
                    thefile.close()
            except:
                print("ERROR:::Silhoute score invalid")
                continue


"""
Knowledge base look-up filtering
"""


def Kb(numberOfSeeds, name, numberOfIteration, iteration):
    # for iteration in range(0,10):
    propernouns = []
    print('filteriiingg....' + str(numberOfSeeds) + '_' + str(name) + '_' + str(iteration))

    path = ROOTHPATH + '/post_processing_files/' + name + '_Iteration' + str(numberOfIteration) + str(
        numberOfSeeds) + '_' + str(iteration) + '.txt'
    print(path)
    with open(path, "r") as file:
        for row in file.readlines():
            propernouns.append(row.strip())

    dsnames = []
    corpuspath = ROOTHPATH + '/evaluation_files/X_Seeds_' + str(
        numberOfSeeds) + '_' + str(iteration) + '.txt'

    with open(corpuspath, "r") as file:
        for row in file.readlines():
            dsnames.append(row.strip())
            propernouns.append(row.strip())
    for i in range(1, int(numberOfIteration)):
        try:
            with open(ROOTHPATH + '/evaluation_files/' + name + '_Iteration' + str(i) + '_POS_' + str(
                    numberOfSeeds) + '_' + str(iteration) + '.txt', 'r') as file:
                for row in file.readlines():
                    dsnames.append(row.strip())
                    propernouns.append(row.strip())
        except:
            continue
    newpropernouns = []

    for pp in propernouns:
        temp = pp.split(' ')
        if len(temp) > 1:
            bigram = list(nltk.bigrams(pp.split()))

            for bi in bigram:
                bi = bi[0].lower() + '_' + bi[1].lower()

                newpropernouns.append(bi)
        else:
            newpropernouns.append(pp)
    finallist = []
    for nn in newpropernouns:
        url = 'http://lookup.dbpedia.org/api/search/KeywordSearch?QueryClass=place&QueryString=' + str(
            nn)
        try:
            resp = requests.request('GET', url)
            root = ElementTree.fromstring(resp.content)
            check_if_exist = []
            for child in root.iter('*'):
                check_if_exist.append(child)
            if len(check_if_exist) == 1:
                finallist.append(nn.replace('_', ' '))
        except:
            finallist.append(nn.replace('_', ' '))
    thefile = open(
        ROOTHPATH + "/evaluation_files/" + name + "_Iteration" + numberOfIteration + "_POS_" + str(
            numberOfSeeds) + "_" + str(iteration) + ".txt", 'w')
    finallist = list(set(finallist))

    for item in finallist:
        if item.lower() not in dsnames:
            thefile.write("%s\n" % item)
    thefile.close()


"""
PMI filtering
"""


def PMI(numberOfSeeds, name, numberOfIteration, iteration):
    # for iteration in range(0,10):
    propernouns = []
    print('filteriiingg....' + str(numberOfSeeds) + '_' + str(name) + '_' + str(iteration))

    path = ROOTPATH + '/post_processing_files/' + name + '_Iteration' + str(numberOfIteration) + str(
        numberOfSeeds) + '_' + str(iteration) + '.txt'
    print(path)
    with open(path, "r") as file:
        for row in file.readlines():
            propernouns.append(row.strip())
    dsnames = []
    corpuspath = ROOTPATH + '/evaluation_files_prot/X_Seeds_' + str(
        numberOfSeeds) + '_' + str(iteration) + '.txt'

    with open(corpuspath, "r") as file:
        for row in file.readlines():
            dsnames.append(row.strip())

    newpropernouns = []
    for pp in propernouns:
        temp = pp.split(' ')
        if len(temp) > 1:
            bigram = list(nltk.bigrams(pp.split()))

            for bi in bigram:
                bi = bi[0].lower() + '_' + bi[1].lower()
                newpropernouns.append(bi)
        else:
            newpropernouns.append(pp)
    finallist = normalized_pub_distance.NPD(newpropernouns)

    thefile = open(
        ROOTPATH + "/evaluation_files_prot/" + name + "_Iteration" + numberOfIteration + "_POS_" + str(
            numberOfSeeds) + "_" + str(iteration) + ".txt", 'w')
    finallist = list(set(finallist))

    for item in finallist:
        thefile.write("%s\n" % item)
    thefile.close()


"""
Majority vote filtering
"""


def MV(numberOfSeeds, name, numberOfIteration, iteration):
    # for iteration in range(0,10):
    propernouns = []
    print('filteriiingg....' + str(numberOfSeeds) + '_' + str(name) + '_' + str(iteration))

    """
    Use the extracted entities from the publications
    """
    path = ROOTHPATH + '/post_processing_files/' + name + '_Iteration' + str(numberOfIteration) + str(
        numberOfSeeds) + '_' + str(iteration) + '.txt'
    print(path)
    with open(path, "r") as file:
        for row in file.readlines():
            propernouns.append(row.strip())
    dsnames = []
    corpuspath = ROOTHPATH + '/evaluation_files/X_Seeds_' + str(
        numberOfSeeds) + '_' + str(iteration) + '.txt'

    with open(corpuspath, "r") as file:
        for row in file.readlines():
            dsnames.append(row.strip())

    newpropernouns = []
    for pp in propernouns:
        temp = pp.split(' ')
        if len(temp) > 1:
            bigram = list(nltk.bigrams(pp.split()))

            for bi in bigram:
                bi = bi[0].lower() + '_' + bi[1].lower()
                newpropernouns.append(bi)
        else:
            newpropernouns.append(pp)
    finallist = majorityVote(newpropernouns, numberOfSeeds, iteration)

    thefile = open(
        ROOTHPATH + "/evaluation_filesMet/" + name + "_Iteration" + numberOfIteration + "_POS_" + str(
            numberOfSeeds) + "_" + str(iteration) + ".txt", 'w')
    finallist = list(set(finallist))

    for item in finallist:
        thefile.write("%s\n" % item)
    thefile.close()
def WordNet_StopWord(numberOfSeeds, name, numberOfIteration, iteration):
    # for iteration in range(0,10):
    propernouns = []
    print('filteriiingg....' + str(numberOfSeeds) + '_' + str(name) + '_' + str(iteration))

    path = ROOTPATH + '/post_processing_files/' + name + '_Iteration' + str(numberOfIteration) + str(
        numberOfSeeds) + '_' + str(iteration) + '.txt'
    print(path)
    with open(path, "r") as file:
        for row in file.readlines():
            propernouns.append(row.strip())
    dsnames = []
    corpuspath = ROOTPATH + '/evaluation_files_prot/X_Seeds_' + str(
        numberOfSeeds) + '_' + str(iteration) + '.txt'

    with open(corpuspath, "r") as file:
        for row in file.readlines():
            dsnames.append(row.strip())
    filterbywordnet = []
    filtered_words = [word for word in set(propernouns) if word not in stopwords.words('english')]
    filtered_words = [word for word in set(filtered_words) if word.lower() not in stopwordList]

    # filterbywordnet = [word for word in filtered_words if not wordnet.synsets(word)]
    print(filtered_words)
    for word in set(filtered_words):

        inwordNet = 1

        if not wordnet.synsets(word):
            filterbywordnet.append(word)
    thefile = open(
        ROOTPATH + "/evaluation_files_prot/" + name + "_Iteration" + numberOfIteration + "_POS_" + str(
            numberOfSeeds) + "_" + str(iteration) + ".txt", 'w')
    finallist = list(set(filterbywordnet))

    for item in finallist:
        thefile.write("%s\n" % item)
    thefile.close()