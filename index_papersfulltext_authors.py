import pymongo
import elasticsearch
from elasticsearch import helpers
import nltk
import config as cfg

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
client = pymongo.MongoClient('localhost:' + str(cfg.mongoDB_Port))
publications_collection = client.pub.publications
es = elasticsearch.Elasticsearch([{'host': 'localhost', 'port': 9200}],
                                 timeout=30, max_retries=10, retry_on_timeout=True)
es.cluster.health(wait_for_status='yellow', request_timeout=1)


def extract_from_mongodb(search_string, collection):
    results = collection.find(search_string)
    list_of_docs = []

    extracted = {
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

        list_of_docs.append(extracted)

        extracted = {
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
    extracted_publications.append(extract_from_mongodb(query, publications_collection))

for publication in extracted_publications:
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
