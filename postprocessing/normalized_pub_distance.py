from elasticsearch import Elasticsearch
import re
import math

regex = re.compile(".*?\((.*?)\)")


def calculate_npd(extracted_entities, context):
    """

    :param extracted_entities:
    :param context:
    :return filtered_entities:
    """
    filtered_entities = []
    context_words = context

    # context words for dataset
    # context_words = ['dataset', 'corpus', 'collection', 'repository', 'benchmark', 'website']

    # context words for method
    # context_words = ['method', 'model', 'algorithm', 'approach','technique']

    # context words for proteins
    # context_words = ['protein', 'receptor']

    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
    extracted_entities = [x.lower() for x in extracted_entities]
    extracted_entities = list(set(extracted_entities))
    for cn in context_words:
        for entity in extracted_entities:
            if any(x in entity.lower() for x in context_words):
                filtered_entities.append(entity)
            NN = 2897901
            query = {"query":
                {"match": {
                    "content.chapter.sentpositive": {
                        "query": entity,
                        "operator": "and"
                    }
                }
                }
            }
            res = es.search(index="twosent", doc_type="twosentnorules", body=query)
            total_a = res['hits']['total']
            query = {"query":
                {"match": {
                    "content.chapter.sentpositive": {
                        "query": cn,
                        "operator": "and"
                    }
                }
                }
            }
            res = es.search(index="twosent", doc_type="twosentnorules", body=query)
            total_b = res['hits']['total']
            query_text = entity + ' ' + cn
            query = {"query":
                {"match": {
                    "content.chapter.sentpositive": {
                        "query": query_text,
                        "operator": "and"
                    }
                }
                }
            }
            res = es.search(index="twosent", doc_type="twosentnorules", body=query)
            total_ab = res['hits']['total']
            if total_a and total_b and total_ab:
                total_ab = total_ab / NN
                total_a = total_a / NN
                total_b = total_b / NN
                pmi = total_ab / (total_a * total_b)
                pmi = math.log(pmi, 2)
                if pmi >= 2:
                    filtered_entities.append(entity)
    return filtered_entities
