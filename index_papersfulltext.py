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
        "content": "",
        "publication": "",
        "year":""
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

        list_of_docs.append(my_dict)

        my_dict = {
            "_id": "",
            "title": "",
            "content": "",
            "publication": "",
            "year": ""
        }

    return list_of_docs
    
filter_publications = ["WWW", "ICSE", "VLDB", "JCDL", "TREC",  "SIGIR", "ICWSM", "ECDL", #"ESWC", "TPDL", 
                         "PLoS Biology", "Breast Cancer Research", "BMC Evolutionary Biology", "BMC Genomics", "BMC Biotechnology",
                        "BMC Neuroscience", "Genome Biology", "PLoS Genetics", "Breast Cancer Research : BCR", 
                       "Genome Biology and Evolution", "Breast Cancer Research and Treatment"]

list_of_pubs = []

for publication in filter_publications:
    mongo_string_search = {'$or': [{'$and': [{'booktitle': publication}, {'content.fulltext': {'$exists': True}}]} ,
                                   {'$and': [{'journal': publication},   {'content.fulltext': {'$exists': True}}]} ]}
    
    list_of_pubs.append(return_chapters(mongo_string_search, db))

for pubs in list_of_pubs:
    actions = []
    
    for cur in pubs:

        print(cur['_id'])
        print(cur['publication'])

        actions.append({
                    "_index": "ir",
                    "_type": "publications",
                    "_id" : cur['_id'],
                    "_publication": cur['publication'],
                    "_year": cur['year'],
                    "_source" : {
                        "text" : cur["content"],
                        "title": cur["title"]
                    }
                })
    if len(actions) == 0:
        continue

    res = helpers.bulk(es, actions)
    print(res)