from elasticsearch import Elasticsearch
from config import ROOTPATH
from nltk.tokenize import word_tokenize
import re
import csv
from nltk import tokenize
import sys
import os

es = Elasticsearch([{'host': 'localhost', 'port': 9200}])


def extract_similar_sentences(count):
    """
    Function for finding similar sentences given the code of a sentence (everything is stored in elasticsearch)
    """
    query = {"query":
                 {"match":
                      {"_id":
                           {"query": count,
                            "operator": "and"
                           }
                      }
                 }
             }

    res = es.search(index="devtwosentnew", doc_type="devtwosentnorulesnew",
                    body=query, size=1)
    for doc in res['hits']['hits']:
        sentence = doc['_source']['content.chapter.sentpositive']
    return sentence


def extract(numberOfSeeds, name, numberOfIteration, iteration):
    """
    function for extracting new sentences for either method or dataset entities and excludes the testing sentences
    """
    print('started extract....', numberOfSeeds, name, iteration)

    # These are all sentences containing any term from the test set, and they need to to be excluded from any
    #  training

    fileUnlabelled = open(ROOTPATH + '/data/protein-names-test.txt')

    text = fileUnlabelled.read()
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
    testsentences = tokenize.sent_tokenize(text)
    dsnames = []
    X_testB = []

    # Exclude all the entities appeared in the test file from the training file
    with open(ROOTPATH + '/data/protein-names-test.txt', 'r') as file:
        for row in file.readlines():
            X_testB.append(row.strip())

    X_testB = [ds.lower() for ds in X_testB]

    """
        Use the  extracted entities in the Expansion steps to find sentences containing those seeds
    """
    for i in range(1, int(numberOfIteration) + 1):
        with open(ROOTPATH + '/evaluation_files_prot/' + name + '_Iteration' + str(i) + '_POS_' + str(
                numberOfSeeds) + '_' + str(iteration) + '.txt', 'r') as file:
            for row in file.readlines():
                dsnames.append(row.strip())

    """
        Use the  initial seeds
    """
    with open(ROOTPATH + '/evaluation_files_prot/X_Seeds_' + str(numberOfSeeds) + '_' + str(iteration) + '.txt',
              'r') as file:
        for row in file.readlines():
            dsnames.append(row.strip())

    dsnames = [ds.lower() for ds in dsnames]
    dsnames = list(set(dsnames))
    temp = []
    # print(X_testB)

    """
        Exclude the terms of the test set
    """
    for word in dsnames:
        if word not in X_testB:
            temp.append(word)

    """
       Instantiate the elasticsearch
    """
    es = Elasticsearch(
        [{'host': 'localhost', 'port': 9200}]
    )

    X_train = temp

    paragraph = []
    print(len(X_train))

    """
        Extract new training data using the new seeds from the elasticsearch
    """
    for dataset in X_train:
        datasetname = re.sub(r'\([^)]*\)', '', dataset)

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
                        body=query, size=100)

        for doc in res['hits']['hits']:

            sentence = doc["_source"]["content.chapter.sentpositive"]

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

            """
                Exclude the sentences containing an entity of the testset
            """
            if any(ext in sentence.lower() for ext in X_testB):

                continue
                #
                #
            else:
                #                 sentences = tokenize.sent_tokenize(sentence)
                #                 if sentences[0] not in paragraph and sentences[0] not in testsentences:
                #                     if sentences[1] not in paragraph and sentences[1] not in testsentences:
                paragraph.append(sentence)

    paragraph = list(set(paragraph))
    paragraph = ' '.join(paragraph)

    sentences = re.sub(r"(\.)([A-Z])", r"\1 \2", paragraph)

    text_file = open(
        ROOTPATH + '/evaluation_files_prot/' + name + 'text_Iteration' + numberOfIteration + str(
            numberOfSeeds) + '_' + str(
            iteration) + '.txt', 'w')
    text_file.write(sentences)
    text_file.close()


def label_sentences_term_expansion_only(model_name, training_cycle):
    """
    Function for annotating the training data using the extracted terms only (TE)
    """
    seed_entities = []

    # Use the initial seed terms
    path = ROOTPATH + '/processing_files/seeds.txt'
    with open(path, "r") as file:
        for row in file.readlines():
            seed_entities.append(row.strip())

    # Use the entities extracted from the Term Expansion approach
    path = (ROOTPATH + '/processing_files/expanded_seeds_' + model_name + '_' + str(training_cycle) + '.txt')
    if os.path.exists(path):
        with open(path, 'r') as file:
            for row in file.readlines():
                seed_entities.append(row.strip())
    else:
        print('No terms expanded for model', model_name, 'cycle', str(training_cycle))

    # This part only works for iterations >1. It adds the newly extracted and filtered entities to the set of seed terms
    # for i in range(1, int(training_cycle) + 1):
    #     try:
    #         with open(ROOTPATH + '/evaluation_files_prot/' + name + '_Iteration' + str(i) + '_POS_' + str(
    #                 numberOfSeeds) + '_' + str(iteration) + '.txt', 'r') as file:
    #             for row in file.readlines():
    #                 seed_entities.append(row.strip())
    #     except:
    #         continue

    seed_entities = [x.lower() for x in seed_entities]
    seed_entities = list(set(seed_entities))

    # If this is the first iteration, use the initial text extracted using the initial seeds.
    # Else, use the new extracted training data which contain the new filtered entities

    if int(training_cycle) == 0:
        path = ROOTPATH + '/processing_files/sentences_from_seeds.txt'
        unlabelled_sentences_file = open(path, 'r', encoding='utf-8')
    else:
        path = ROOTPATH + '/processing_files/sentences_from_cycle' + str(training_cycle) + '.txt'
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
    lines = (tokenize.sent_tokenize(text.strip()))

    labelled_sentences = []
    lines = list(set(lines))

    # In this loop, for each sentence, we check if it contains the terms of the seed_entities (seed terms),
    # if yes annotate that word with the model_name. We basically create  a dictionary with all the words
    # and their corresponding labels

    print('Labelling')
    for line in lines:
        index = [i for i, x in enumerate(seed_entities) if seed_entities[i] in line.lower()]
        word_dict = {}
        words = word_tokenize(line)
        tag = '/' + model_name.upper()
        for word in words:
            word_dict[word] = ''

        if index:
            for i in index:
                split_entity = seed_entities[i].split()
                flag = False
                for idx, word in enumerate(words):
                    if flag:
                        flag = False
                        if word[0].isupper():
                            word_dict[word] = word + tag

                    elif seed_entities[i] in word.lower() and len(word) > 2 and len(seed_entities[i]) > 3:
                        if len(seed_entities[i]) < 5:
                            if word.lower().startswith(seed_entities[i]):
                                word_dict[word] = word + tag
                        else:
                            word_dict[word] = word + tag

                    elif len(split_entity) > 1:
                        try:
                            if len(split_entity) == 2:
                                if word.lower() in split_entity[0] and words[idx + 1].lower() in split_entity[1]:
                                    if len(word) > 2:
                                        word_dict[word] = word + tag
                                        word_dict[words[idx + 1]] = words[idx + 1] + tag

                            elif len(split_entity) == 3:
                                if word.lower() in split_entity[0] and words[idx + 1].lower() in split_entity[1] and \
                                        words[idx + 2].lower() in split_entity[2]:
                                    if len(word) > 2:
                                        word_dict[word] = word + tag
                                        word_dict[words[idx + 1]] = words[idx + 1] + tag
                                        word_dict[words[idx + 2]] = words[idx + 2] + tag
                                elif word.lower() in split_entity[0] and words[idx + 1].lower() in \
                                        split_entity[1]:
                                    if len(word) > 2:
                                        word_dict[word] = word + tag
                                        word_dict[words[idx + 1]] = words[idx + 1] + tag
                        except:
                            continue

            sentence = ''

            # Now that we have a dictionary with all the words and their corresponding labels and generate the
            # training data in the tab separated format

            for i, word in enumerate(words):
                if word_dict[word] == '':
                    sentence = sentence + ' ' + word

                else:
                    if tag in word_dict[word]:
                        sentence = sentence + ' ' + word_dict[word]

            labelled_sentences.append(sentence)

        else:
            labelled_sentences.append(line)

    print(len(lines), 'lines labelled')
    inputs = []
    for ll in labelled_sentences:
        words = word_tokenize(ll)
        for word in words:
            if tag in word:
                label = tag
                word = word.split('/')
                word = word[0]
            else:
                label = 'O'
            inputs.append([word, label])

    with open(ROOTPATH + '/processing_files/TE_tagged_sentence_cycle_' + str(training_cycle) + '.txt',
              'w', encoding='utf-8') as f:
        for item in inputs:
            row = str(item[0]) + '\t' + str(item[1]) + "\n"
            f.write(row)

    file = open(ROOTPATH + '/processing_files/TE_tagged_sentence_cycle_' + str(training_cycle) + '_splitted.txt',
                'w', encoding='utf-8')

    with open(ROOTPATH + '/processing_files/TE_tagged_sentence_cycle_' + str(training_cycle) + '.txt',
              'r', encoding='utf-8') as tsvin:
        tsvin = csv.reader(tsvin, delimiter='\t')

        for row in tsvin:
            if '###' in row[0]:
                continue
            elif row[0] == '.':
                rows = str(row[0]) + '\t' + str(row[1]) + "\n"
                file.write(rows)
                file.write("\n")
            else:
                rows = str(row[0]) + '\t' + str(row[1]) + "\n"
                file.write(rows)
    file.close()


def label_sentences_term_sentence_expansion(model_name, training_cycle, vector_model):
    """
    Function for generating the training data using the extracted terms and sentences (SE)
    """

    seed_entities = []

    # Use the initial seed terms
    path = ROOTPATH + '/processing_files/seeds.txt'
    with open(path, "r") as file:
        for row in file.readlines():
            seed_entities.append(row.strip())

    # Use the entities extracted from the Term Expansion approach
    path = (ROOTPATH + '/processing_files/expanded_seeds_' + model_name + '_' + str(training_cycle) + '.txt')
    if os.path.exists(path):
        with open(path, 'r') as file:
            for row in file.readlines():
                seed_entities.append(row.strip())
    else:
        print('No terms expanded for model', model_name, 'cycle', str(training_cycle))

    # This part only works for iterations >1. It adds the newly extracted and filtered entities to the set of seed terms
    # for i in range(1, int(training_cycle) + 1):
    #     try:
    #         with open(ROOTPATH + '/evaluation_files_prot/' + name + '_Iteration' + str(i) + '_POS_' + str(
    #                 numberOfSeeds) + '_' + str(iteration) + '.txt', 'r') as file:
    #             for row in file.readlines():
    #                 seed_entities.append(row.strip())
    #     except:
    #         continue

    seed_entities = [x.lower() for x in seed_entities]
    seed_entities = list(set(seed_entities))

    # If this is the first iteration, use the initial text extracted using the initial seeds.
    # Else, use the new extracted training data which contain the new filtered entities

    if int(training_cycle) == 0:
        path = ROOTPATH + '/processing_files/sentences_from_seeds.txt'
        unlabelled_sentences_file = open(path, 'r', encoding='utf-8')
    else:
        path = ROOTPATH + '/processing_files/sentences_from_cycle' + str(training_cycle) + '.txt'
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

    lines = (tokenize.sent_tokenize(text.strip()))
    lines = list(set(lines))

    # For each sentence in the training data, Find the most similar sentences and add it to the training data.
    # With this step we expand our training set

    print('Expanding sentences')
    temp = []
    for i, line in enumerate(lines):
        tokens = line.split()
        new_vector = vector_model.infer_vector(tokens)
        sims = vector_model.docvecs.most_similar([new_vector], topn=1)
        if sims:
            for ss in sims:
                if ss[1] > 0.50:
                    temp.append(extract_similar_sentences(str(ss[0])))

    lines = list(set(lines))
    temp = list(set(temp))
    print('Added', len(temp), 'expanded sentences to the', len(lines), 'original')

    for tt in temp:
        lines.append(tt)

    lines = list(set(lines))
    sys.stdout.flush()

    # In this loop, for each sentence, we check if it contains the terms of the seed_entities (seed terms),
    # if yes annotate that word with the model_name. We basically create  a dictionary with all the words
    # and their corresponding labels
    labelled_sentences = []

    print('Labelling')
    for line in lines:
        index = [i for i, x in enumerate(seed_entities) if seed_entities[i] in line.lower()]
        word_dict = {}
        words = word_tokenize(line)
        tag = '/' + model_name.upper()
        for word in words:
            word_dict[word] = ''

        if index:
            for i in index:
                split_entity = seed_entities[i].split()
                flag = False
                for idx, word in enumerate(words):
                    if flag:
                        flag = False
                        if word[0].isupper():
                            word_dict[word] = word + tag

                    elif seed_entities[i] in word.lower() and len(word) > 2 and len(seed_entities[i]) > 3:
                        if len(seed_entities[i]) < 5:
                            if word.lower().startswith(seed_entities[i]):
                                word_dict[word] = word + tag
                        else:
                            word_dict[word] = word + tag

                    elif len(split_entity) > 1:
                        try:
                            if len(split_entity) == 2:
                                if word.lower() in split_entity[0] and words[idx + 1].lower() in split_entity[1]:
                                    if len(word) > 2:
                                        word_dict[word] = word + tag
                                        word_dict[words[idx + 1]] = words[idx + 1] + tag

                            elif len(split_entity) == 3:
                                if word.lower() in split_entity[0] and words[idx + 1].lower() in split_entity[1] and \
                                        words[idx + 2].lower() in split_entity[2]:
                                    if len(word) > 2:
                                        word_dict[word] = word + tag
                                        word_dict[words[idx + 1]] = words[idx + 1] + tag
                                        word_dict[words[idx + 2]] = words[idx + 2] + tag
                                elif word.lower() in split_entity[0] and words[idx + 1].lower() in \
                                        split_entity[1]:
                                    if len(word) > 2:
                                        word_dict[word] = word + tag
                                        word_dict[words[idx + 1]] = words[idx + 1] + tag
                        except:
                            continue

            sentence = ''

            # Now that we have a dictionary with all the words and their corresponding labels and generate the
            # training data in the tab separated format

            for i, word in enumerate(words):
                if word_dict[word] == '':
                    sentence = sentence + ' ' + word

                else:
                    if tag in word_dict[word]:
                        sentence = sentence + ' ' + word_dict[word]

            labelled_sentences.append(sentence)

        else:
            labelled_sentences.append(line)

    print(len(lines), 'lines labelled')
    inputs = []
    for ll in labelled_sentences:
        words = word_tokenize(ll)
        for word in words:
            if tag in word:
                label = tag
                word = word.split('/')
                word = word[0]
            else:
                label = 'O'
            inputs.append([word, label])

    with open(ROOTPATH + '/processing_files/TSE_tagged_sentence_cycle_' + str(training_cycle) + '.txt',
              'w', encoding='utf-8') as f:
        for item in inputs:
            row = str(item[0]) + '\t' + str(item[1]) + "\n"
            f.write(row)

    file = open(ROOTPATH + '/processing_files/TSE_tagged_sentence_cycle_' + str(training_cycle) + '_splitted.txt',
                'w', encoding='utf-8')

    with open(ROOTPATH + '/processing_files/TSE_tagged_sentence_cycle_' + str(training_cycle) + '.txt',
              'r', encoding='utf-8') as tsvin:
        tsvin = csv.reader(tsvin, delimiter='\t')

        for row in tsvin:
            if '###' in row[0]:
                continue
            elif row[0] == '.':
                rows = str(row[0]) + '\t' + str(row[1]) + "\n"
                file.write(rows)
                file.write("\n")
            else:
                rows = str(row[0]) + '\t' + str(row[1]) + "\n"
                file.write(rows)
    file.close()
