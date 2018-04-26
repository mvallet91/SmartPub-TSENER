import random
import elasticsearch
import nltk
import re
import config as cfg
import sys


def sentence_extraction(number_of_seeds: int) -> None:
    """
    Extracts from the corpus all sentences that include one of the given number of seeds (number_of_seeds).
    This iterative process is executed 10 times selecting number_of_seeds new entities, generating one file
    with the sentences for each iteration. In addition, it excludes sentences that have any of the entities
    from a test set, when provided.
    :param number_of_seeds: desired number of seed entities to extract from set
    :type number_of_seeds: int
    :returns: Creates and saves files with sentences in path
    :rtype: None
    """
    es = elasticsearch.Elasticsearch([{'host': 'localhost', 'port': 9200}])
    print('Started training data extraction')

    # First we get the entity names which have been used in the testing set to exclude them from the
    # training sentences
    test_entities = []
    path = cfg.ROOTPATH + '/data/names-test.txt'
    with open(path, 'r') as file:
        for row in file.readlines():
            test_entities.append(row.strip())
    test_entities = [e.lower() for e in test_entities]
    test_entities = list(set(test_entities))

    # List of seed names
    seed_entities = []
    path = cfg.ROOTPATH + '/data/names-train.txt'
    with open(path, 'r') as file:
        for row in file.readlines():
            seed_entities.append(row.strip())
    seed_entities = [e.lower() for e in seed_entities]
    seed_entities = list(set(seed_entities))

    # 10 times randomly pick n number of seeds
    for i in range(10):
        training_entities = random.sample(seed_entities, number_of_seeds)
        print('Selected seed terms', training_entities)
        paragraph = []

        # using the seeds, extract the sentences from the publications' text (with Elasticsearch)
        for entity in training_entities:
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
        print(len(paragraph), 'sentences added for training in iteration', i)
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

    print('Process finished with', number_of_seeds, 'seeds')
