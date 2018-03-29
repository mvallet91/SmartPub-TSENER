"""
This script will be used to generate the training data from the extracted entities or extracting new training data.
"""
from elasticsearch import Elasticsearch
from config import ROOTPATH
from nltk.tokenize import word_tokenize
import re
import csv
from nltk import tokenize
import sys

es = Elasticsearch(
    [{'host': 'localhost', 'port': 9200}]
)

"""
Function for finding similar sentences given the code of a sentence (everything is stored in elasticsearch)
"""


def extract_similarsentences(count):
    query = {"query":
        {"match": {
            "_id": {
                "query": count,
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


"""
function for extracting new sentences for either method or dataset entities and excludes the testing sentences
"""


def extract(numberOfSeeds, name, numberOfIteration, iteration):
    print('started extract....', numberOfSeeds, name, iteration)

    """
    These are all sentences containing any term from the test set, and they need to to be excluded from any
     training
    """
    fileUnlabelled = open('/data/testB_datasettext.txt')

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

    """
        Exclude all the entities appeared in the test file from the training file
    """
    with open('/data/dataset-names-testb.txt', 'r') as file:
        # with open('/data/method-names-testb', 'r') as file:
        for row in file.readlines():
            X_testB.append(row.strip())

    X_testB = [ds.lower() for ds in X_testB]

    """
        Use the  extracted entities in the Expansion steps to find sentences containing those seeds
    """
    for i in range(1, int(numberOfIteration) + 1):
        with open(ROOTHPATH + '/evaluation_files/' + name + '_Iteration' + str(i) + '_POS_' + str(
                numberOfSeeds) + '_' + str(iteration) + '.txt', 'r') as file:
            for row in file.readlines():
                dsnames.append(row.strip())

    """
        Use the  initial seeds
    """
    with open(ROOTHPATH + '/evaluation_files/X_Seeds_' + str(numberOfSeeds) + '_' + str(iteration) + '.txt',
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
                sentences = tokenize.sent_tokenize(sentence)
                if sentences[0] not in paragraph and sentences[0] not in testsentences:
                    if sentences[1] not in paragraph and sentences[1] not in testsentences:
                        paragraph.append(sentence)

    paragraph = list(set(paragraph))
    paragraph = ' '.join(paragraph)

    sentences = re.sub(r"(\.)([A-Z])", r"\1 \2", paragraph)

    text_file = open(
        ROOTHPATH + '/evaluation_files/' + name + 'text_Iteration' + numberOfIteration + str(numberOfSeeds) + '_' + str(
            iteration) + '.txt', 'w')
    text_file.write(sentences)
    text_file.close()


"""
Function for annotating the training data using the extracted terms only (TE)
"""


def generate_trainingTE(numberOfSeeds, name, numberOfIteration, iteration):
    dsnames = []
    """
        Use the intial seed terms
     """
    corpuspath = ROOTPATH + '/evaluation_files_prot/X_Seeds_' + str(numberOfSeeds) + '_' + str(iteration) + '.txt'
    with open(corpuspath, "r") as file:
        for row in file.readlines():
            dsnames.append(row.strip())

    """
        Use the entities extracted from the Term Expansion approach
     """
    try:
        with open(ROOTPATH + '/evaluation_files_prot/' + name + 'Pre_Iteration' + str(numberOfIteration) + '_POS_' + str(
                numberOfSeeds) + '_' + str(iteration) + '.txt', 'r') as file:
            for row in file.readlines():
                dsnames.append(row.strip())
    except:
        print('No terms expanded for', 'Pre_Iteration' + str(numberOfIteration) + '_POS_' + str(
                numberOfSeeds) + '_' + str(iteration))

    """
    This part only works for iterations >1. It adds the newly extracted and filtered entities to the set of seed terms
    """
    for i in range(1, int(numberOfIteration) + 1):
        try:
            with open(ROOTPATH + '/evaluation_files_prot/' + name + '_Iteration' + str(i) + '_POS_' + str(
                    numberOfSeeds) + '_' + str(iteration) + '.txt', 'r') as file:
                for row in file.readlines():
                    dsnames.append(row.strip())
        except:
            continue

    dsnames = [x.lower() for x in dsnames]
    dsnames = list(set(dsnames))


    """
        If in the first iteration use the initial text extracted using the  intial seeds else use the new extracted training data which contain the new filtered entities
    """
    print(numberOfIteration)
    print('/evaluation_files_prot/X_train_' + str(numberOfSeeds) + '_' + str(iteration) + '.txt')
    if int(numberOfIteration) == 0:
        fileUnlabelled = open(
            ROOTPATH + '/evaluation_files_prot/X_train_' + str(numberOfSeeds) + '_' + str(iteration) + '.txt', 'r')


    else:
        fileUnlabelled = open(
            ROOTHPATH + '/evaluation_files_prot/' + name + 'text_Iteration' + numberOfIteration + str(
                numberOfSeeds) + '_' + str(
                iteration) + '.txt')

    text = fileUnlabelled.read()
    print(text)
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
    labelledtext = list()

    print(len(lines))

    lines = list(set(lines))
    """
        In this loop, for each sentence, we check if it contains the terms of the dsnames (seed terms),
        if yes annotate that word with 'PROT'. We basically create  a dictionary with all the words and their corresponding labels
    """
    for line in lines:

        index = [i for i, x in enumerate(dsnames) if dsnames[i] in line.lower()]
        worddict = dict()
        words = word_tokenize(line)

        for word in words:
            worddict[word] = ''

        if index:

            for i in index:

                dsnamesplitted = dsnames[i].split()

                flag = False
                for idx, word in enumerate(words):

                    if flag == True:

                        flag = False
                        if word[0].isupper():
                            worddict[word] = word + '/PROT'

                    elif dsnames[i] in word.lower() and len(word) > 2 and len(dsnames[i]) > 3:
                        if len(dsnames[i]) < 5:
                            if word.lower().startswith(dsnames[i]):
                                worddict[word] = word + '/PROT'
                        else:
                            worddict[word] = word + '/PROT'


                    elif len(dsnamesplitted) > 1:
                        try:
                            if len(dsnamesplitted) == 2:

                                if word.lower() in dsnamesplitted[0] and words[idx + 1].lower() in dsnamesplitted[1]:
                                    if len(word) > 2:
                                        worddict[word] = word + '/PROT'
                                        worddict[words[idx + 1]] = words[idx + 1] + '/PROT'

                            elif len(dsnamesplitted) == 3:

                                if word.lower() in dsnamesplitted[0] and words[idx + 1].lower() in dsnamesplitted[1] and \
                                                words[idx + 2].lower() in dsnamesplitted[2]:
                                    if len(word) > 2:
                                        worddict[word] = word + '/PROT'
                                        worddict[words[idx + 1]] = words[idx + 1] + '/PROT'
                                        worddict[words[idx + 2]] = words[idx + 2] + '/PROT'
                                elif word.lower() in dsnamesplitted[0] and words[idx + 1].lower() in \
                                        dsnamesplitted[1]:
                                    if len(word) > 2:
                                        worddict[word] = word + '/PROT'
                                        worddict[words[idx + 1]] = words[idx + 1] + '/PROT'

                        except:
                            continue

            sentence = ''
            """
                Now that we have a dictionary with all the words and their corresponding labels and generate the
                training data in the tabseperated format
            """
            for i, word in enumerate(words):
                if worddict[word] == '':
                    sentence = sentence + ' ' + word

                else:
                    if '/PROT' in worddict[word]:
                        sentence = sentence + ' ' + worddict[word]

            labelledtext.append(sentence)

        else:
            labelledtext.append(line)
    inputs = []
    for ll in labelledtext:
        words = word_tokenize(ll)

        for word in words:
            if '/PROT' in word:
                label = 'PROT'
                word = word.split('/')
                word = word[0]

            else:
                label = 'O'
            inputs.append([word, label])
    print(inputs)
    sys.stdout.flush()
    with open(ROOTPATH + '/evaluation_files_prot/' + name + '_text_iteration' + numberOfIteration + str(
            numberOfSeeds) + '_' + str(iteration) + '.txt', 'w') as file1:
        for item in inputs:
            row = str(item[0]) + '\t' + str(item[1]) + "\n"
            file1.write(row)

    file = open(ROOTPATH + '/evaluation_files_prot/' + name + '_text_iteration' + numberOfIteration + '_splitted' + str(
        numberOfSeeds) + '_' + str(iteration) + '.txt', 'w')
    with open(ROOTPATH + '/evaluation_files_prot/' + name + '_text_iteration' + numberOfIteration + str(
            numberOfSeeds) + '_' + str(iteration) + '.txt', 'r') as tsvin:
        tsvin = csv.reader(tsvin, delimiter='\t')

        for row in tsvin:
            # print(row)
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


"""
Function for generating the training data using the extracted terms and sentences (SE)
"""


def generate_trainingSE(numberOfSeeds, name, numberOfIteration, iteration, model):
    dsnames = []
    dstemp = []

    """
        Use the intial seed terms
    """
    corpuspath = ROOTHPATH + '/evaluation_files/X_Seeds_' + str(numberOfSeeds) + '_' + str(
        iteration) + '.txt'
    with open(corpuspath, "r") as file:
        for row in file.readlines():
            dsnames.append(row.strip())
            dstemp.append(row.strip())

    """
            Use the entities extracted from the Term Expansion approach
    """
    try:
        with open(ROOTHPATH + '/evaluation_files/' + name + 'Pre_Iteration' + str(
                numberOfIteration) + '_POS_' + str(
            numberOfSeeds) + '_' + str(iteration) + '.txt', 'r') as file:
            for row in file.readlines():
                dsnames.append(row.strip())
    except:
        pass

    dsnames = [x.lower() for x in dsnames]
    dsnames = list(set(dsnames))

    """
        If in the first iteration use the initial text extracted using the  intial seeds else use the new extracted training data which contain the new filtered entities
    """
    if int(numberOfIteration) == 0:
        fileUnlabelled = open(
            ROOTHPATH + '/evaluation_files/X_train_' + str(numberOfSeeds) + '_' + str(
                iteration) + '.txt', 'r')
    else:
        fileUnlabelled = open(
            ROOTHPATH + '/evaluation_files/' + name + 'text_Iteration' + numberOfIteration + str(
                numberOfSeeds) + '_' + str(
                iteration) + '.txt')

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

    lines = (tokenize.sent_tokenize(text.strip()))
    temp = []

    """
        For each sentence in the training data, Find the most similar sentences and add it to the training data. With this step
        we expand our training set
    """
    for i, line in enumerate(lines):
        tokens = line.split()

        new_vector = model.infer_vector(tokens)
        sims = model.docvecs.most_similar([new_vector], topn=1)
        try:
            for ss in sims:

                if ss[1] > 0.50:
                    temp.append(extract_similarsentences(str(ss[0])))
        except:
            continue

    lines = list(set(lines))
    temp = list(set(temp))
    for tt in temp:
        lines.append(tt)
    labelledtext = list()

    print(len(lines))

    """
        In this loop, for each sentence, we check if it contains the terms of the dsnames (seed terms),
        if yes annotate that word with 'DATA'. We basically create  a dictionary with all the words and their corresponding labels
    """
    lines = list(set(lines))
    for line in lines:

        index = [i for i, x in enumerate(dsnames) if dsnames[i] in line.lower()]
        worddict = dict()
        words = word_tokenize(line)

        for word in words:
            worddict[word] = ''

        if index:

            for i in index:

                dsnamesplitted = dsnames[i].split()

                flag = False
                for idx, word in enumerate(words):

                    if flag == True:

                        flag = False
                        if word[0].isupper():
                            worddict[word] = word + '/DATA'

                    elif dsnames[i] in word.lower() and len(word) > 2 and len(dsnames[i]) > 3:
                        if len(dsnames[i]) < 5:
                            if word.lower().startswith(dsnames[i]):
                                worddict[word] = word + '/DATA'
                        else:
                            worddict[word] = word + '/DATA'

                    elif len(dsnamesplitted) > 1:
                        try:
                            if len(dsnamesplitted) == 2:

                                if word.lower() in dsnamesplitted[0] and words[idx + 1].lower() in \
                                        dsnamesplitted[1]:
                                    worddict[word] = word + '/DATA'
                                    worddict[words[idx + 1]] = words[idx + 1] + '/DATA'

                            elif len(dsnamesplitted) == 3:

                                if word.lower() in dsnamesplitted[0] and words[idx + 1].lower() in \
                                        dsnamesplitted[1] and \
                                                words[idx + 2].lower() in dsnamesplitted[2]:
                                    worddict[word] = word + '/DATA'
                                    worddict[words[idx + 1]] = words[idx + 1] + '/DATA'
                                    worddict[words[idx + 2]] = words[idx + 2] + '/DATA'
                                elif word.lower() in dsnamesplitted[0] and words[idx + 1].lower() in \
                                        dsnamesplitted[1]:
                                    worddict[word] = word + '/DATA'
                                    worddict[words[idx + 1]] = words[idx + 1] + '/DATA'

                        except:
                            continue

            sentence = ''
            for i, word in enumerate(words):
                if worddict[word] == '':
                    sentence = sentence + ' ' + word

                else:
                    if '/DATA' in worddict[word]:
                        sentence = sentence + ' ' + worddict[word]

            labelledtext.append(sentence)
        else:
            labelledtext.append(line)
    inputs = []
    for ll in labelledtext:
        words = word_tokenize(ll)

        for word in words:
            if '/DATA' in word:

                word = word.split('/')
                word = word[0]

                label = 'DATA'


            else:
                label = 'O'
            inputs.append([word, label])
    with open(ROOTHPATH + '/evaluation_files/' + name + '_text_iteration' + numberOfIteration + str(
            numberOfSeeds) + '_' + str(iteration) + '.txt', 'w') as file:
        for item in inputs:
            row = str(item[0]) + '\t' + str(item[1]) + "\n"
            file.write(row)
    file = open(
        ROOTHPATH + '/evaluation_files/' + name + '_text_iteration' + numberOfIteration + '_splitted' + str(
            numberOfSeeds) + '_' + str(iteration) + '.txt', 'w')
    with open(ROOTHPATH + '/evaluation_files/' + name + '_text_iteration' + numberOfIteration + str(
            numberOfSeeds) + '_' + str(iteration) + '.txt', 'r') as tsvin:
        tsvin = csv.reader(tsvin, delimiter='\t')

        for row in tsvin:
            # print(row)
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