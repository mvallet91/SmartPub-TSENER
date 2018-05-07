from nltk.tag.stanford import StanfordNERTagger
from elasticsearch import Elasticsearch
import itertools
import nltk
from pymongo import MongoClient
from nltk.corpus import wordnet
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from nltk.corpus import stopwords
import re
import csv
from gensim.models.wrappers import FastText
model = FastText.load_fasttext_format('/data/modelFT')
from app.modules import filter_entities
import random
from nltk.tag.stanford import StanfordPOSTagger
import config as cfg
from config import ROOTPATH

client = MongoClient('localhost:' + cfg.mongoDB_Port)
db = client.pub

dsnames = []


def get_continuous_chunks(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    #print pos_tag(word_tokenize(text))

    prev = None
    continuous_chunk = []
    current_chunk = []

    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue
    return continuous_chunk


es = Elasticsearch(
    [{'host': 'localhost', 'port': 9200}]
)

es.cluster.health(wait_for_status='yellow')

pub = client.pub.dataset_names.find()

path_to_jar = cfg.STANFORD_NER_PATH

#if you want to extract all the dataset entities you need to change all the 'MET' to 'DATA' or 'PROT' and change the path_to_model and also in the store_datasetname_in_mongo function change the label field to 'dataset' or 'protein'

def get_NEs(res):

    for i, (a, b) in enumerate(res):
        if b == 'PROT':
            temp = a
            if i > 1:
                j = i - 1
                if res[j][1] == 'PROT':
                    continue
            j = i + 1
            try:
                if res[j][1] == 'PROT':
                    temp = b
                    temp = res[j][0] + ' ' + b
                    z = j + 1
                    if res[j][1] == 'PROT':
                        temp = a + ' ' + res[j][0] + ' ' + res[z][0]

            except:
                continue

            # result.append(a)
            result.append(temp)

    print(result)
    filterbywordnet = []
    filtered_words = [word for word in set(result) if word not in stopwords.words('english')]

    # filter_by_wordnet = [word for word in filtered_words if not wordnet.synsets(word)]
    print(filtered_words)
    for word in set(filtered_words):

        inwordNet = 1

        if not wordnet.synsets(word):
            filterbywordnet.append(word)
            inwordNet = 0
        filteredword, PMIdata, PMImethod, dssimilarity, mtsimilarity, ds_sim_50, ds_sim_60, ds_sim_70, ds_sim_80, ds_sim_90, mt_sim_50, mt_sim_60, mt_sim_70, mt_sim_80, mt_sim_90 = filter_entities.filter_it(word, model)


        store_datasetname_in_mongo(db,doc["_id"],doc["_source"]["title"], doc["_source"]["journal"],doc["_source"]["year"],  word, inwordNet, filteredword, PMIdata, PMImethod, dssimilarity, mtsimilarity, ds_sim_50, ds_sim_60, ds_sim_70, ds_sim_80, ds_sim_90, mt_sim_50, mt_sim_60, mt_sim_70, mt_sim_80, mt_sim_90)

def check_if_id_exist_in_db(db, id, word):
    check_string = {'$and':[{'paper_id':id},{'word':word}]}
    if db.datasetNER.find_one(check_string) is not None:
        print("We already checked this paper")
        return True
    else:
        return False
    
def store_datasetname_in_mongo(db, id, title, journal, year, word, inwordNet, filtered_words, PMIdata, PMImethod, dssimilarity, mtsimilarity, ds_sim_50, ds_sim_60, ds_sim_70, ds_sim_80, ds_sim_90, mt_sim_50, mt_sim_60, mt_sim_70, mt_sim_80, mt_sim_90):
    my_ner = {
        "paper_id":id,
        "title":title,
        "journal":journal,
        "year":year,
        "word":word,
        "label":'protein',
        "inwordNet":inwordNet,
        "filtered_words":filtered_words,
        "PMIdata":PMIdata,
        "PMImethod":PMImethod,
        "dssimilarity":dssimilarity,
        "mtsimilarity":mtsimilarity,
        "ds_sim_50":ds_sim_50,
        "ds_sim_60":ds_sim_60,
        "ds_sim_70":ds_sim_70,
        "ds_sim_80":ds_sim_80,
        "ds_sim_90":ds_sim_90,
        "mt_sim_50":mt_sim_50,
        "mt_sim_60":mt_sim_60,
        "mt_sim_70":mt_sim_70,
        "mt_sim_80":mt_sim_80,
        "mt_sim_90":mt_sim_90

    }
    if check_if_id_exist_in_db(db, id, word):
        return False
    else:
        db.entities.insert_one(my_ner)
        return True


publications = ["WWW", "ICSE", "VLDB", "JCDL", "TREC",  "SIGIR", "ICWSM", "ECDL", #"ESWC", "TPDL", 
                "PLoS Biology", "Breast Cancer Research", "BMC Evolutionary Biology", "BMC Genomics", "BMC Biotechnology",
                "BMC Neuroscience", "Genome Biology", "PLoS Genetics", "Breast Cancer Research : BCR", 
                "Genome Biology and Evolution", "Breast Cancer Research and Treatment"]

for publication in publications:

    query = {"query":
        {"match": {
            "journal": {
                "query": publication,
                "operator": "and"
            }
        }
        }
    }

    res = es.search(index="ir", doc_type="publications",
                    body=query,  size=10000)
    
    print(len(res['hits']['hits']))
    
    # random_int=random.sample(range(0, 616), 150)
    # for i in random_int:
    #      print(res['hits']['hits'][i]['_id'])

    for doc in res['hits']['hits']:
        # sentence = doc["_source"]["text"].replace(',', ' ')

        query = doc["_source"]["content"]
        print(doc["_source"]["title"])
        
        path_to_model = 'crf_trained_files/term_expansion_text_iteration0_splitted10_0.ser.gz'
       
        nertagger = StanfordNERTagger(path_to_model, path_to_jar)

        res = nertagger.tag(query.split())

        result = []
        result2 = []

        get_NEs(res)