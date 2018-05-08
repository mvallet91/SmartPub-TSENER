import re
import string
import sys
import nltk
import os

from config import ROOTPATH, STANFORD_NER_PATH
from nltk.tag.stanford import StanfordNERTagger
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from elasticsearch import Elasticsearch

es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

filter_by_wordnet = []


def long_tail_labelling(model_name, input_text):

    result = []
    print('started extraction for the', model_name, 'model')
    path_to_model = ROOTPATH + '/crf_trained_files/' + model_name + '_TSE_model_0.ser.gz'

    # use the trained Stanford NER model to extract entities from the publications
    ner_tagger = StanfordNERTagger(path_to_model, STANFORD_NER_PATH)
    sentences = nltk.sent_tokenize(input_text)

    for sentence in sentences:
        sentence = re.sub(r'\[[^\(]*?\]', r'', sentence)
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
                    if sentences[jj + 1][1] == tag:
                        temp = sentences[jj + 1][0].translate(str.maketrans('', '', string.punctuation))
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
    filtered_words = [word for word in filtered_words if not wordnet.synsets(word)]
    print('Total of', len(filtered_words), 'filtered entities added')
    sys.stdout.flush()
    for sentence in sentences:
        print(sentence)
    print('')
    print('Entities labelled', filtered_words)
    return filtered_words
