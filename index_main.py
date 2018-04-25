import pymongo
import elasticsearch
from elasticsearch import helpers
import nltk
import config as cfg
import logging

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
client = pymongo.MongoClient('localhost:' + str(cfg.mongoDB_Port))
publications_collection = client.pub.publications
es = elasticsearch.Elasticsearch([{'host': 'localhost', 'port': 9200}],
                                 timeout=30, max_retries=10, retry_on_timeout=True)
es.cluster.health(wait_for_status='yellow', request_timeout=1)


def extract_metadata(documents):
    list_of_docs = []
    extracted = {
        "_id": "",
        "title": "",
        "publication": "",
        "year": "",
        "content": "",
        "abstract": "",
        "authors": [],
        "paragraphs": [],
        "references": []

    }
    for i, r in enumerate(documents):
        # try:
        # list_of_sections = list()
        extracted['_id'] = r['_id']
        extracted['title'] = r['title']
        try:
            extracted['publication'] = r['booktitle']
        except:
            pass
        try:
            extracted['publication'] = r['journal']
        except:
            pass
        try:
            extracted['year'] = r['year']
        except:
            pass
        try:
            extracted['content'] = r['content']['fulltext']
        except:
            extracted['content'] = ""
        try:
            extracted['abstract'] = r['content']['abstract']
        except:
            extracted['abstract'] = ""
        try:
            extracted['authors'] = r['authors']
        except:
            extracted['authors'] = []
        try:
            extracted['references'] = r['content']['references']
        except:
            extracted['references'] = []

        paragraphs = []
        try:
            for chapter in r['content']['chapters']:
                if chapter == {}:
                    continue

                if len(chapter) == 1:
                    for paragraph in chapter[0]['paragraphs']:
                        if paragraph == {}:
                            continue
                        paragraphs.append(paragraph)

                else:
                    for paragraph in chapter['paragraphs']:
                        if paragraph == {}:
                            continue
                        paragraphs.append(paragraph)

            extracted['paragraphs'] = paragraphs

        except:
            logging.exception('No chapters in ' + r['_id'], exc_info=True)
            continue

        list_of_docs.append(extracted)

        extracted = {
            "_id": "",
            "title": "",
            "publication": "",
            "year": "",
            "content": "",
            "abstract": "",
            "authors": [],
            "paragraphs": [],
            "references": []

        }
    return list_of_docs


def index_metadata(publication_list):

    for publication in publication_list:
        actions = []
        for article in publication:
            print(article['_id'])
            print(article['publication'])

            authors = []
            if len(article['authors']) > 0:
                if type(article['authors'][0]) == list:
                    try:
                        for name in article['authors']:
                            authors.append(', '.join([name[1], name[0]]))
                        authors = list(set(authors))
                    except:
                        pass
                else:
                    authors = article['authors']

            actions.append({
                "_index": "ir_full",  # surfall   # ir_full
                "_type": "publications",  # pubs      # publications
                "_id": article['_id'],
                "_source": {
                    "title": article["title"],
                    "journal": article['publication'],
                    "year": str(article['year']),
                    "content": article["content"],
                    "abstract": article["abstract"],
                    "authors": authors,
                    "references": article["references"]
                }
            })
        if len(actions) == 0:
            continue
        res = helpers.bulk(es, actions)
        print(res)


def index_sentences(publication_list):

    for publication in publication_list:
        for article in publication:
            actions = []
            dataset_sent = []

            for paragraph in article['paragraphs']:
                if paragraph == {}:
                    continue

                lines = (sent_detector.tokenize(paragraph.strip()))
                with open('data/full_text_corpus.txt', 'a') as f:
                    for line in lines:
                        f.write(line)

                if len(lines) < 3:
                    continue

                for i in range(len(lines)):
                    words = nltk.word_tokenize(lines[i])
                    word_lengths = [len(x) for x in words]
                    average_word_length = sum(word_lengths) / len(word_lengths)
                    if average_word_length < 4:
                        continue

                    two_sentences = ''
                    try:
                        two_sentences = lines[i] + ' ' + lines[i - 1]
                    except:
                        two_sentences = lines[i] + ' ' + lines[i + 1]

                    dataset_sent.append(two_sentences)

            for num, added_lines in enumerate(dataset_sent):
                actions.append({
                    "_index": "twosent",
                    "_type": "twosentnorules",
                    "_id": article['_id'] + str(num),
                    "_source": {
                        "title": article['title'],
                        "content.chapter.sentpositive": added_lines,
                        "paper_id": article['_id']
                    }})

            if len(actions) == 0:
                continue
            res = helpers.bulk(es, actions)
            print(res)


filter_publications = ["WWW", "ICSE", "VLDB", "PVLDB", "JCDL", "TREC", "SIGIR", "ICWSM", "ECDL", "ESWC",
                       "IEEE J. Robotics and Automation", "IEEE Trans. Robotics", "ICRA", "ICARCV", "HRI",
                       "ICSR", "PVLDB", "TPDL", "ICDM", "Journal of Machine Learning Research", "Machine Learning",
                       "PLoS Biology", "Breast Cancer Research", "BMC Evolutionary Biology", "BMC Genomics",
                       "BMC Neuroscience", "Genome Biology", "PLoS Genetics", "Breast Cancer Research : BCR",
                       "Genome Biology and Evolution", "Breast Cancer Research and Treatment", "BMC Biotechnology"]

extracted_publications = []
for publication in filter_publications:
    query = {'$or': [{'$and': [{'booktitle': publication}, {'content.fulltext': {'$exists': True}}]},
                     {'$and': [{'journal': publication}, {'content.fulltext': {'$exists': True}}]}]}
    results = publications_collection.find(query)
    extracted_publications.append(extract_metadata(results))

index_metadata(extracted_publications)
index_sentences(extracted_publications)