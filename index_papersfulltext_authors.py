from pymongo import MongoClient
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import nltk
import config as cfg

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

client = MongoClient('localhost:' + str(cfg.mongoDB_Port))
db = client.pub
pub = client.pub.publications
es = Elasticsearch([{'host': 'localhost', 'port': 9200}],
                   timeout=30, max_retries=10, retry_on_timeout=True)

es.cluster.health(wait_for_status='yellow', request_timeout=1)


def return_chapters(mongo_string_search, db):
    results = db.publications.find(mongo_string_search)
    chapters = list()
    chapter_nums = list()
    list_of_docs = list()
    merged_chapters = list()

    my_dict = {
        "_id": "",
        "title": "",
        "publication": "",
        "year": "",
        "content": "",
        "abstract": "",
        "authors": [],
        "references": []

    }
    for i, r in enumerate(results):
        # try:
        # list_of_sections = list()
        my_dict['_id'] = r['_id']
        my_dict['title'] = r['title']
        try:
            my_dict['publication'] = r['booktitle']
        except:
            pass
        try:
            my_dict['publication'] = r['journal']
        except:
            pass
        try:
            my_dict['year'] = r['year']
        except:
            pass
        try:
            my_dict['content'] = r['content']['fulltext']
        except:
            my_dict['content'] = ""
        try:
            my_dict['abstract'] = r['content']['abstract']
        except:
            my_dict['abstract'] = ""
        try:
            my_dict['authors'] = r['authors']
        except:
            my_dict['authors'] = []
        try:
            my_dict['references'] = r['content']['references']
        except:
            my_dict['references'] = []

        list_of_docs.append(my_dict)

        my_dict = {
            "_id": "",
            "title": "",
            "publication": "",
            "year": "",
            "content": "",
            "abstract": "",
            "authors": [],
            "references": []

        }

    return list_of_docs


filter_publications = ["WWW", "ICSE", "VLDB", "JCDL", "TREC", "SIGIR", "ICWSM", "ECDL", "ESWC", "TPDL"]

filter_publications = ["WWW", "ICSE", "VLDB", "PVLDB", "JCDL", "TREC", "SIGIR", "ICWSM", "ECDL", "ESWC",
                       "IEEE J. Robotics and Automation", "IEEE Trans. Robotics", "ICRA", "ICARCV", "HRI",
                       "ICSR", "PVLDB", "TPDL", "ICDM", "Journal of Machine Learning Research", "Machine Learning"]

#                          "PLoS Biology", "Breast Cancer Research", "BMC Evolutionary Biology", "BMC Genomics", "BMC Biotechnology",
#                         "BMC Neuroscience", "Genome Biology", "PLoS Genetics", "Breast Cancer Research : BCR", 
#                        "Genome Biology and Evolution", "Breast Cancer Research and Treatment"]

list_of_pubs = []

for publication in filter_publications:
    mongo_string_search = {'$or': [{'$and': [{'booktitle': publication}, {'content.fulltext': {'$exists': True}}]},
                                   {'$and': [{'journal': publication}, {'content.fulltext': {'$exists': True}}]}]}

    list_of_pubs.append(return_chapters(mongo_string_search, db))

for pubs in list_of_pubs:

    actions = []

    for cur in pubs:

        print(cur['_id'])
        print(cur['publication'])

        authors = []
        if len(cur['authors']) > 0:
            if type(cur['authors'][0]) == list:
                try:
                    for name in cur['authors']:
                        authors.append(', '.join([name[1], name[0]]))
                    authors = list(set(authors))
                except:
                    pass
            else:
                authors = cur['authors']

        actions.append({
            "_index": "ir_full",  # surfall ir_full
            "_type": "publications",  # pubs  publications
            "_id": cur['_id'],
            "_source": {
                "title": cur["title"],
                "journal": cur['publication'],
                "year": str(cur['year']),
                "content": cur["content"],
                "abstract": cur["abstract"],
                "authors": authors,
                "references": cur["references"]
            }
        })
    if len(actions) == 0:
        continue

    res = helpers.bulk(es, actions)
    print(res)
