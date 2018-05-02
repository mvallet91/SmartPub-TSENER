import elasticsearch
import nltk
import re
from config import ROOTPATH
import sys


def sentence_extraction(list_of_seeds: str) -> None:
    """
    Extracts from the corpus all sentences that include one of the given list of seeds (list_of_seeds).
    In addition, it excludes sentences that have any of the entities from a test set, when provided.
    :param list_of_seeds: text list of seed entities
    :type list_of_seeds: str
    :returns: Creates and saves files with sentences in path
    :rtype: None
    """
    es = elasticsearch.Elasticsearch([{'host': 'localhost', 'port': 9200}])
    print('Started training data extraction')

    testing = False
    test_entities = []
    if testing:
        # We get the entity names which have been used in the testing set to exclude them from the
        # training sentences
        test_entities = []
        path = ROOTPATH + '/data/demo-test.txt'
        with open(path, 'r') as file:
            for row in file.readlines():
                test_entities.append(row.strip())
        test_entities = [e.lower() for e in test_entities]
        test_entities = list(set(test_entities))

    # List of seed names
    seed_entities = []
    path = ROOTPATH + '/data/demo-train.txt'
    if list_of_seeds:
        seed_entities = list_of_seeds
    else:
        with open(path, 'r') as file:
            for row in file.readlines():
                seed_entities.append(row.strip())
    seed_entities = [e.lower() for e in seed_entities]
    seed_entities = list(set(seed_entities))

    print('Selected seed terms', seed_entities)
    paragraph = []

    # Using the seeds, extract the sentences from the publications text in Elasticsearch index
    for entity in seed_entities:
        entity_name = re.sub(r'\([^)]*\)', '', entity)

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

        # clean up the sentences and if they don't contain the names of the test set then add them as
        # the training data
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

            if any(e in words for e in test_entities):
                print('sentence removed')
                continue
            else:
                paragraph.append(sentence)

    paragraph = list(set(paragraph))
    print(len(paragraph), 'sentences added for training in iteration')
    print('')
    sys.stdout.flush()

    # Store sentences and seeds in files
    f1 = open(ROOTPATH + '/processing_files/sentences_from_seeds.txt', 'w', encoding='utf-8')
    for item in paragraph:
        f1.write('%s\n' % item)
    f1.close()

    f1 = open(ROOTPATH + '/processing_files/seeds.txt', 'w')   # We could use mongodb instead of txt file...
    for item in seed_entities:
        f1.write('%s\n' % item)
    f1.close()

    print('Process finished with', len(seed_entities), 'seeds')
