import gensim
import sys
import os
from nltk.tag.stanford import StanfordNERTagger
from elasticsearch import Elasticsearch
from pymongo import MongoClient
from nltk.corpus import wordnet
from nltk.corpus import stopwords
# from gensim.models.wrappers import FastText

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from m1_postprocessing import process_extracted_entities
import config as cfg

# embedding_model = FastText.load_fasttext_format('/data/modelFT')
embedding_model = gensim.models.Word2Vec.load('embedding_models/modelword2vecbigram.model')

es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
es.cluster.health(wait_for_status='yellow')

client = MongoClient('localhost:' + cfg.mongoDB_Port)
db_mongo = client.TU_Delft_Library
path_to_jar = cfg.STANFORD_NER_PATH
entity_names = []


# def get_continuous_chunks(text):
#     chunked = ne_chunk(pos_tag(word_tokenize(text)))
#     prev = None
#     continuous_chunk = []
#     current_chunk = []
#     for i in chunked:
#         if type(i) == Tree:
#             current_chunk.append(" ".join([token for token, pos in i.leaves()]))
#         elif current_chunk:
#             named_entity = " ".join(current_chunk)
#             if named_entity not in continuous_chunk:
#                 continuous_chunk.append(named_entity)
#                 current_chunk = []
#         else:
#             continue
#     return continuous_chunk


def store_entity_in_mongo(db, _id, title, journal, year, word, in_wordnet, filtered_words, pmi_data, pmi_method,
                          ds_similarity, mt_similarity, ds_sim_50, ds_sim_60, ds_sim_70, ds_sim_80, ds_sim_90,
                          mt_sim_50, mt_sim_60, mt_sim_70, mt_sim_80, mt_sim_90):
    my_ner = {
        "paper_id": _id,
        "title": title,
        "journal": journal,
        "year": year,
        "word": word,
        "label": 'protein',
        "in_wordnet": in_wordnet,
        "filtered_words": filtered_words,
        "pmi_data": pmi_data,
        "pmi_method": pmi_method,
        "ds_similarity": ds_similarity,
        "mt_similarity": mt_similarity,
        "ds_sim_50": ds_sim_50,
        "ds_sim_60": ds_sim_60,
        "ds_sim_70": ds_sim_70,
        "ds_sim_80": ds_sim_80,
        "ds_sim_90": ds_sim_90,
        "mt_sim_50": mt_sim_50,
        "mt_sim_60": mt_sim_60,
        "mt_sim_70": mt_sim_70,
        "mt_sim_80": mt_sim_80,
        "mt_sim_90": mt_sim_90

    }

    db.entities.update_one({'_id': my_ner['_id']}, {'$set': {my_ner}})


def get_entities(words, current_model):
    results = []
    facet_tag = current_model.upper()
    for i, (a, b) in enumerate(words):
        if b == facet_tag:
            temp = a
            if i > 1:
                j = i - 1
                if words[j][1] == facet_tag:
                    continue
            j = i + 1
            try:
                if words[j][1] == facet_tag:
                    temp = b
                    temp = words[j][0] + ' ' + b
                    z = j + 1
                    if words[j][1] == facet_tag:
                        temp = a + ' ' + words[j][0] + ' ' + words[z][0]
            except KeyError:
                continue
            results.append(temp)

    wordnet_filtered = []
    filtered_words = [word for word in set(results) if word not in stopwords.words('english')]

    for word in set(filtered_words):
        in_wordnet = 1
        if not wordnet.synsets(word):
            wordnet_filtered.append(word)
            in_wordnet = 0

        filtered_word, pmi_data, pmi_method, ds_similarity, mt_similarity, ds_sim_50, \
        ds_sim_60, ds_sim_70, ds_sim_80, ds_sim_90, mt_sim_50, mt_sim_60, mt_sim_70, \
        mt_sim_80, mt_sim_90 = process_extracted_entities.filter_it(word, embedding_model)

        store_entity_in_mongo(db_mongo, doc["_id"], doc["_source"]["title"], doc["_source"]["journal"],
                              doc["_source"]["year"], word, in_wordnet, filtered_word, pmi_data, pmi_method,
                              ds_similarity, mt_similarity, ds_sim_50, ds_sim_60, ds_sim_70, ds_sim_80, ds_sim_90,
                              mt_sim_50, mt_sim_60, mt_sim_70, mt_sim_80, mt_sim_90)


publications = ['tudelft']
# "WWW", "ICSE", "VLDB", "JCDL", "TREC", "SIGIR", "ICWSM", "ECDL",  # "ESWC", "TPDL",
# "PLoS Biology", "Breast Cancer Research", "BMC Evolutionary Biology", "BMC Genomics",
# "BMC Biotechnology",
# "BMC Neuroscience", "Genome Biology", "PLoS Genetics", "Breast Cancer Research : BCR",
# "Genome Biology and Evolution", "Breast Cancer Research and Treatment"]

model_name = 'dataset_tud'

for publication in publications:
    query = {"query":
                {"match":
                    {"journal":
                        {"query": publication,
                         "operator": "and"
                         }
                     }
                 }
             }

    res = es.search(index="ir", doc_type="publications",
                    body=query, size=10000)

    print(len(res['hits']['hits']))

    for doc in res['hits']['hits']:
        text = doc["_source"]["content"]
        print(doc["_source"]["title"])
        path_to_model = 'crf_trained_files/' + model_name + '_TSE_model_0.ser.gz'
        ner_tagger = StanfordNERTagger(path_to_model, path_to_jar)
        labelled_words = ner_tagger.tag(text.split())
        result = []
        result2 = []
        get_entities(labelled_words, model_name)
