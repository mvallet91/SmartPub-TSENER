"""
Here we will extract a number of sentences for the training data using different number of dataset name seeds
(e.g  2, 5, 10, ...)
"""

import random
import elasticsearch
import nltk
import re
import config as cfg
import sys


def extract(number_of_seeds):
    es = elasticsearch.Elasticsearch([{'host': 'localhost', 'port': 9200}])
    print('Started training data extraction')

    # First we get the dataset names which have been used in the testing set (TestB) to exclude them from the
    # training sentences
    test_entities = []
    with open(cfg.ROOTPATH + '/data/protein-names-test.txt', 'r') as file:
        for row in file.readlines():
            test_entities.append(row.strip())
    # lowercase the names
    test_entities = [e.lower() for e in test_entities]
    test_entities = list(set(test_entities))

    # List of seed names
    seed_entities = []
    path = cfg.ROOTPATH + "/data/protein-names-train.txt"
    with open(path, "r") as file:
        for row in file.readlines():
            seed_entities.append(row.strip())
    seed_entities = [e.lower() for e in seed_entities]
    seed_entities = list(set(seed_entities))

    # 10 times randomly pick i number of seeds
    for i in range(10):

        # shuffle the list
        training_entities = random.sample(seed_entities, number_of_seeds)
        print('Selected seed terms', training_entities)
        paragraph = []

        # using the seeds, extract the sentences from the publications using the query in elastic search
        for entity in training_entities:
            entity_name = re.sub(r'\([^)]*\)', '', entity)

            # Matching
            query = {"query":
                        {"match":
                            {"content.chapter.sentpositive":
                                {"query": entity_name,
                                 "operator": "and"
                                 }
                             }
                         }
                     }

            res = es.search(index="twosent", doc_type="twosentnorules",
                            body=query, size=1000)
            print(entity_name)
            print("Got %d hits in ES" % res['hits']['total'])

            # clean up the sentences and if they dont contain the names of the test set then add them as
            # the training data
            for doc in res['hits']['hits']:

                sentence = doc["_source"]["content.chapter.sentpositive"]
                words = nltk.word_tokenize(doc["_source"]["content.chapter.sentpositive"])
                lengths = [len(x) for x in words]
                average = sum(lengths) / len(lengths)
                # Remove noise sentences
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

                if any(ext in words for ext in test_entities):
                    print('sentence removed')
                    continue

                else:
                    paragraph.append(sentence)

        paragraph = list(set(paragraph))
        print(len(paragraph), 'sentences added')
        print('')
        sys.stdout.flush()

        f1 = open(cfg.ROOTPATH + '/evaluation_files_prot/X_train_' + str(number_of_seeds) + '_' + str(i) + '.txt', 'w')
        for item in paragraph:
            f1.write(item)
        f1.close()

        f1 = open(cfg.ROOTPATH + '/evaluation_files_prot/X_Seeds_' + str(number_of_seeds) + '_' + str(i) + '.txt', 'w')
        for item in training_entities:
            f1.write(item + '\n')
        f1.close()
