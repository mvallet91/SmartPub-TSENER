from pymongo import MongoClient
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import math
import requests
import nltk
import _pickle as cPickle
import config as cfg
import logging

###############################
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
client = MongoClient('localhost:' + str(cfg.mongoDB_Port))
pub = client.pub.publications
db = client.pub
es = Elasticsearch([{'host': 'localhost', 'port': 9200}],
                   timeout=30, max_retries=10, retry_on_timeout=True)
es.cluster.health(wait_for_status='yellow', request_timeout=1)

def return_paragraphs(mongo_string_search, db):
    # mongo_string_search = {"dblpkey": "{}".format(dblkey)}
    results = db.publications.find(mongo_string_search)
    chapters = list()
    chapter_nums = list()
    list_of_docs = list()
    # list_of_abstracts = list()
    merged_chapters = list()
    my_dict = {
        "_id": "",
        "paragraphs": list(),
        "title": ""
    }
    for i, r in enumerate(results):
        my_dict['_id'] = r['_id']
        my_dict['title'] = r['title']
        paragraphs = []
        
        try:

            for chapter in r['content']['chapters']:
                if (chapter == {}):
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

                            
            my_dict['paragraphs'] = paragraphs

            paragraphs = []
            
        except:
            logging.exception('No chapters in ' + r['_id'], exc_info=True)
            continue

        list_of_docs.append(my_dict)
        my_dict = {
            "_id": "",
            "paragraphs": list(),
            "title": ""
        }


    return list_of_docs


filter_publications = ["WWW", "ICSE", "VLDB", "JCDL", "TREC",  "SIGIR", "ICWSM", "ECDL", "ESWC", "TPDL", 
                         "PLoS Biology", "Breast Cancer Research", "BMC Evolutionary Biology", "BMC Genomics", "BMC Biotechnology",
                        "BMC Neuroscience", "Genome Biology", "PLoS Genetics", "Breast Cancer Research : BCR", 
                       "Genome Biology and Evolution", "Breast Cancer Research and Treatment"]

list_of_pubs = []

for publication in filter_publications:
    mongo_string_search = {'$or': [{'$and': [{'booktitle': publication}, {'content.fulltext': {'$exists': True}}]} ,
                                   {'$and': [{'journal': publication},   {'content.fulltext': {'$exists': True}}]} ]}
    
    list_of_pubs.append(return_paragraphs(mongo_string_search, db))

print("Total journals:", len(list_of_pubs))

for pubs in list_of_pubs:
    for paper in pubs:
        print(paper['_id'])
        actions = []
        cleaned = []
        datasetsent = []
        othersent = []
        
        for paragraph in paper['paragraphs']:
            if paragraph == {}:
                continue
            lines = (sent_detector.tokenize(paragraph.strip()))
            
            if len(lines) < 3:
                continue

            for i in range(len(lines)):
                words = nltk.word_tokenize(lines[i])
                lengths = [len(x) for x in words]
                average = sum(lengths) / len(lengths)
                if average < 4:
                    continue
                    
                twosentences = ''
                try:
                    twosentences = lines[i] + ' ' + lines[i-1]

                except:
                    twosentences = lines[i] + ' ' + lines[i+1]
                    
                datasetsent.append(twosentences)

        for num, parag in enumerate(datasetsent):
            actions.append({
                "_index": "twosent",
                "_type": "twosentnorules",
                "_id": paper['_id'] + str(num),

                "_source" : {
                    "title" : paper['title'],
                    "content.chapter.sentpositive" : parag,
                    "paper_id":paper['_id']
                }})
            
        if len(actions) == 0:
            continue

        res = helpers.bulk(es, actions)
        print(res)