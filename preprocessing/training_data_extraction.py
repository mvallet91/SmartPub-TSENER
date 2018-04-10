"""
Here we will extract a number of sentences for the training data using different number of dataset name seeds (e.g  2,5,10,...)
"""

import random
from elasticsearch import Elasticsearch
import nltk
import re
from sklearn.model_selection import train_test_split
from nltk import tokenize
import config as cfg
import sys

def extract(numberOfSeeds):
    
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])    
    print('Started training data extraction')
    
    # First we get the dataset names which have been used in the testing set (TestB) to exclude them from the training sentences
    X_testB = []
    with open(cfg.ROOTPATH + '/data/protein-names-test.txt', 'r') as file:
        for row in file.readlines():
            X_testB.append(row.strip())
    # lowercase the names
    X_testB = [ds.lower() for ds in X_testB]
    X_testB = list(set(X_testB))
    
    # List of seed names
    dsnames = []
    corpuspath = cfg.ROOTPATH + "/data/protein-names-train.txt"
    with open(corpuspath, "r") as file:
        for row in file.readlines():
            dsnames.append(row.strip())
    dsnames = [x.lower() for x in dsnames]
    dsnames = list(set(dsnames))

    # 10 times randomly pick i number of seeds
    for i in range(10):

        # shuffle the list
        X_train = random.sample(dsnames, numberOfSeeds)
        print('Selected seed terms', X_train)
        paragraph = []

        # using the seeds, extract the sentences from the publications using the query in elastic search
        for dataset in X_train:
            datasetname = re.sub(r'\([^)]*\)', '', dataset)

            # Matching
            query = {"query":
                {"match": {
                    "content.chapter.sentpositive": {
                        "query": datasetname,
                        "operator": "and"
                    }
                }
                }
            }

            res = es.search(index="twosent", doc_type="twosentnorules",
                            body=query, size=1000)
            print(datasetname)
            print("Got %d hits in ES" % res['hits']['total'])
            
            # clean up the sentences and if they dont contain the names of the testB then add them as the training data
            for doc in res['hits']['hits']:

                sentence = doc["_source"]["content.chapter.sentpositive"]

                words = nltk.word_tokenize(doc["_source"]["content.chapter.sentpositive"])
                lengths = [len(x) for x in words]
                average = sum(lengths) / len(lengths)
                
                if average < 3:
                    continue

                sentence = sentence.replace("@ BULLET", "")
                sentence = sentence.replace("@BULLET", "")
                sentence = sentence.replace(", ", " , ")
                sentence = sentence.replace('(', '')
                sentence = sentence.replace(')', '')
                sentence = sentence.replace('[', '')
                sentence = sentence.replace(']', '')
                sentence = sentence.replace(',', ' ,')
                sentence = sentence.replace('?', ' ?')
                sentence = sentence.replace('..', '.')

#                 if any(ext in sentence.lower() for ext in X_testB):

                if any(ext in words for ext in X_testB):
                    print('sentence removed')
                    continue

                else:
                    paragraph.append(sentence)

        paragraph = list(set(paragraph))
        print(len(paragraph), 'sentences added')
        print('')
        sys.stdout.flush()
        
        f1 = open(cfg.ROOTPATH + '/evaluation_files_prot/X_train_' + str(numberOfSeeds) + '_' + str(i) + '.txt', 'w')
        for item in paragraph:
            f1.write(item)
        f1.close()
        
        f1 = open(cfg.ROOTPATH + '/evaluation_files_prot/X_Seeds_' + str(numberOfSeeds) + '_' + str(i) + '.txt', 'w')
        for item in X_train:
            f1.write(item + '\n')
        f1.close()