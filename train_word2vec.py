import logging
import os.path
import sys
import multiprocessing
import nltk
import string
from gensim.models import Word2Vec, Phrases

# run: python3 word2vec.py data/dataWord2vec.txt data/modelword2vecbigram.model data/modelword2vecbigram.vec

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 4:
        sys.exit(1)

    sentence_stream = []
    inp, output_1, output_2 = sys.argv[1:4]
    file_sent = open(inp)
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    lines = (sent_detector.tokenize(file_sent.read().strip()))
    bigram_transformer = Phrases(min_count=1, threshold=2)
    for line in lines:
        sentence = [word
                    for word in nltk.word_tokenize(line.lower())
                    if word not in string.punctuation]
        sentence_stream.append(sentence)
        bigram_transformer.add_vocab([sentence])

    model = Word2Vec(bigram_transformer[sentence_stream], size=100, window=2, min_count=2,
                     workers=multiprocessing.cpu_count(), sg=1)

    model.save(output_1)
    model.wv.save_word2vec_format(output_2, binary=False)
