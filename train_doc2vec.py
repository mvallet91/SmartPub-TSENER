import logging
import os.path
import sys

import gensim
LabeledSentence = gensim.models.doc2vec.LabeledSentence
from os import listdir
from nltk import tokenize, sent_tokenize
from os.path import isfile, join
from gensim.models import Doc2Vec
import multiprocessing

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    inp, outp1 = sys.argv[1:3]
    
    file = open(inp,'r')
    text = file.read()
    sentences = tokenize.sent_tokenize(text)
    count = 0
    docLabels = []

    for i in range(0, len(sentences)):
        docLabels.append(count)
        count = count + 1

    print(len(sentences))
    print(len(docLabels))

    class LabeledLineSentence(object):
        def __init__(self, doc_list, labels_list):
            self.labels_list = labels_list
            self.doc_list = doc_list
        def __iter__(self):
            for idx, doc in enumerate(self.doc_list):
                yield LabeledSentence(words = doc.split(), tags = [self.labels_list[idx]])

    it = LabeledLineSentence(sentences, docLabels)

    model = Doc2Vec(vector_size = 100, window = 10, min_count = 5, workers = multiprocessing.cpu_count(),
                    epochs = 10, alpha = 0.025, min_alpha = 0.025) # use fixed learning rate
    
    model.build_vocab(it)

    model.train(it, total_examples=model.corpus_count, epochs=model.epochs)

    model.save(outp1)
