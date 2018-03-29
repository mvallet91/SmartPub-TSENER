import logging
import os.path
import sys
import multiprocessing
import nltk
import string

from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec,Phrases
from gensim.models.word2vec import LineSentence

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 4:
        #print(globals()['__doc__'] % locals())
        sys.exit(1)
         
    sentence_stream = []
    inp, outp1, outp2 = sys.argv[1:4]
    filesent = open(inp)
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    lines = (sent_detector.tokenize(filesent.read().strip()))
    bigram_transformer = Phrases(min_count = 1, threshold = 2)
    for line in lines:
        sentence = [word
                    for word in nltk.word_tokenize(line.lower())
                    if word not in string.punctuation]
        sentence_stream.append(sentence)
        bigram_transformer.add_vocab([sentence])

    #sentence_stream = [doc.split(" ") for doc in lines]

    #bigram_transformer = Phrases(sentence_stream, min_count=1, threshold=2)
    model = Word2Vec(bigram_transformer[sentence_stream], size = 100, window = 2, min_count = 2,
                     workers = multiprocessing.cpu_count(), sg = 1)

    # model = Word2Vec(LineSentence(inp), size=400, window=5, min_count=2,
    #                 workers=multiprocessing.cpu_count())

    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary = False)