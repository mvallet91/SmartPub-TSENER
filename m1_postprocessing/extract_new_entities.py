import re
import string
import sys

from config import ROOTPATH, STANFORD_NER_PATH
from nltk.tag.stanford import StanfordNERTagger
from nltk.corpus import stopwords
from elasticsearch import Elasticsearch

es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

filter_by_wordnet = []


def ne_extraction(model_name, training_cycle, sentence_expansion):
    print('started extraction for the', model_name, 'model, in cycle number', training_cycle)

    if sentence_expansion:
        path_to_model = ROOTPATH + '/crf_trained_files/' + model_name + '_TSE_model_' + str(training_cycle) + '.ser.gz'
    else:
        path_to_model = ROOTPATH + '/crf_trained_files/' + model_name + '_TE_model_' + str(training_cycle) + '.ser.gz'

    # use the trained Stanford NER model to extract entities from the publications

    ner_tagger = StanfordNERTagger(path_to_model, STANFORD_NER_PATH)
    result = []
    # filter_conference = ["WWW", "ICSE", "VLDB", "JCDL", "TREC", "SIGIR",
    filter_conference = ["ICWSM", "ECDL", "ESWC", "TPDL"]
    # "PLoS Biology", "Breast Cancer Research", "BMC Evolutionary Biology", "BMC Genomics",
    #  "BMC Biotechnology", "BMC Neuroscience", "Genome Biology", "PLoS Genetics",
    #  "Breast Cancer Research : BCR", "Genome Biology and Evolution",
    # "Breast Cancer Research and Treatment"]

    for conference in filter_conference:
        query = {
            "query": {
                "match": {
                    "publication": {
                        "query": conference,
                        "operator": "and"
                                    }
                    }
                }
            }

        res = es.search(index="ir", doc_type="publications",
                        body=query, size=10000)

        print(len(res['hits']['hits']))
        sys.stdout.flush()

        for doc in res['hits']['hits']:
            sentence = doc["_source"]["text"]
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
            sentence = re.sub(r"(\.)([A-Z])", r"\1 \2", sentence)

            tagged = ner_tagger.tag(sentence.split())

            for jj, (a, b) in enumerate(tagged):
                tag = model_name.upper()
                if b == tag:
                    a = a.translate(str.maketrans('', '', string.punctuation))
                    try:
                        if res[jj + 1][1] == tag:
                            temp = res[jj + 1][0].translate(str.maketrans('', '', string.punctuation))
                            bigram = a + ' ' + temp
                            result.append(bigram)
                    except:
                        result.append(a)
                        continue
                    result.append(a)
            print('.', end='')
            sys.stdout.flush()

    result = list(set(result))
    result = [w.replace('"', '') for w in result]
    filtered_words = [word for word in set(result) if word not in stopwords.words('english')]
    print('Total of', len(filtered_words), 'filtered entities added')
    sys.stdout.flush()
    f1 = open(ROOTPATH + '/processing_files/' + model_name + '_extracted_entities_' + str(training_cycle) + '.txt', 'w',
              encoding='utf-8')
    for item in filtered_words:
        f1.write(item + '\n')
    f1.close()
