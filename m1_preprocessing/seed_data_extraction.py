import elasticsearch
import nltk
import re
from config import ROOTPATH, data_date, nr_sentence, coner_sentence_weight
from m1_preprocessing import term_sentence_expansion
import sys

def sentence_extraction(model_name: str, training_cycle: int, list_of_seeds: list, filter: str = 'majority', coner_expansion: bool = False) -> None:
    """
    Extracts from the corpus all sentences that include at least one of the given seeds (in list_of_seeds).
    In addition, it excludes sentences that have any of the entities from a test set, when provided.
    :param model_name:
    :type model_name:
    :param training_cycle:
    :type training_cycle:
    :param list_of_seeds: text list of seed entities
    :type list_of_seeds: str
    :returns: Creates and saves files for seeds and sentences
    :rtype: None
    """
    es = elasticsearch.Elasticsearch([{'host': 'localhost', 'port': 9200}])
    print(f'Started initial training data extraction for model "{model_name}" and entities resulting from filter "{filter}"')

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
    seed_entities = list_of_seeds
    
    if training_cycle == 0:
        seed_entities = list_of_seeds
    else:
        path = ROOTPATH + '/processing_files/' + model_name + f'_filtered_entities_{filter}_' + str(training_cycle - 1) + '.txt'
        with open(path, 'r') as file:
            for row in file.readlines():
                seed_entities.append(row.strip())
        file.close()

    seed_entities = [e.lower() for e in seed_entities]
    seed_entities = list(set(seed_entities))

    print('Number seed terms:', len(seed_entities))

    # Custom Coner expansion code to append relevant coner entities to seed entities
    if coner_expansion:
        rel_scores = term_sentence_expansion.read_coner_overview(model_name, data_date)
        coner_entities = list(rel_scores.keys())

        seed_entities = list(set(seed_entities + coner_entities))
        print(f'Number Coner relevant entities: {len(coner_entities)}')
        print(f'Number seed entities + Coner selected relevant entities (distinct entities): {len(seed_entities)}')

    print('Extracting sentences for', len(seed_entities), 'seed terms')
    paragraph = []

    # Using the seeds, extract the sentences from the publications text in Elasticsearch index
    for entity in seed_entities:
        sample_size = nr_sentence
        
        # Weighted increase in sentece fetching for entities with Coner human feedback,
        # because they are judged as more likely to have positive occurences of facet entities
        # (human feedback closer to golden standard than heuristic ensemble filtering)
        if coner_expansion and entity in coner_entities: sample_size = sample_size * coner_sentence_weight

        entity_name = re.sub(r'\([^)]*\)', '', entity)
        # print('.', end='')
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
                        body=query, size=sample_size)

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
                continue
            else:
                paragraph.append(sentence)

    paragraph = list(set(paragraph))

    # Store sentences and seeds
    path = ROOTPATH + '/processing_files/' + model_name + '_sentences_' + str(training_cycle) + f'_{filter}.txt'
    f = open(path, 'w', encoding='utf-8')
    for item in paragraph:
        f.write('%s\n' % item)
    f.close()

    path = ROOTPATH + '/processing_files/' + model_name + '_seeds_' + str(training_cycle) + f'_{filter}.txt'
    f = open(path, 'w', encoding='utf-8')   # We could use mongodb instead
    for item in seed_entities:
        f.write('%s\n' % item)
    f.close()

    print('Process finished with', len(seed_entities), 'seeds and',
          len(paragraph), 'sentences added for training in cycle number', str(training_cycle), f'("{filter}" filter method and model "{model_name}")\n')
    sys.stdout.flush()

    return paragraph, seed_entities