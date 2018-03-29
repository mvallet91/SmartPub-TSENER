from pymongo import MongoClient
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import math
import nltk
import string
import config as cfg
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
import sys
import re

###############################

client = MongoClient('localhost:' + str(cfg.mongoDB_Port))
db = client.pub
pub = client.pub.publications
es = Elasticsearch(
    [{'host': 'localhost', 'port': 9200}], timeout=30, max_retries=10, retry_on_timeout=True
)
es.cluster.health(wait_for_status='yellow', request_timeout=1)

def returnnames(mongo_string_search, db):
    results = db.publications.find(mongo_string_search)
    list_of_docs = list()
    my_dict = {
        "_id": "",

    }
    for i, r in enumerate(results):

        my_dict['_id'] = r['_id']
        
        list_of_docs.append((my_dict))

        my_dict = {
            "_id": "",
        }

    return list_of_docs
    
filter_publications = ["WWW", "ICSE", "VLDB", "JCDL", "TREC",  "SIGIR", "ICWSM", "ECDL", "ESWC", "TPDL", 
                          "PLoS Biology", "Breast Cancer Research", "BMC Evolutionary Biology", "BMC Genomics", "BMC Biotechnology",
                        "BMC Neuroscience", "Genome Biology", "PLoS Genetics", "Breast Cancer Research : BCR", 
                       "Genome Biology and Evolution", "Breast Cancer Research and Treatment"]

list_of_pubs = []

for publication in filter_publications:
    mongo_string_search = {'$or': [{'$and': [{'booktitle': publication}, {'content.fulltext': {'$exists': True}}]},
                                   {'$and': [{'journal': publication},   {'content.fulltext': {'$exists': True}}]}
                                  ]
                          }
    list_of_pubs.append(returnnames(mongo_string_search, db))

papersText = []
sentText = []

translator = str.maketrans('', '', string.punctuation)

for pubs in list_of_pubs:
    for cur in pubs:
        query = {"query":
            {"match": {
                "_id": {
                    "query": cur['_id'],
                    "operator": "and"
                }
            }
            }
        }

        res = es.search(index = "ir", body = query, size = 200)

        for doc in res['hits']['hits']:
            fulltext = doc["_source"]["text"]
            fulltext = re.sub("[\[].*?[\]]", "", fulltext)
            word_text = fulltext.translate(translator)
            papersText.append(word_text.lower())
            sentText.append(fulltext)
            print('.', end="")
            sys.stdout.flush()
            
    print('Done', '-'*100)
    
papersText = " ".join(papersText)
sentText = ". ".join(sentText)

f = open("data/dataWord2vec.txt", "w")
f.write(papersText)
f.close()

f = open("data/dataDoc2vec.txt", "w")
f.write(sentText)
f.close()