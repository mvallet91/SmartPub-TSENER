entity_occurrences = {}
word2vec_avg = { 'pmi_data': 0, 'pmi_method': 0, 'ds_similarity': 0, 'mt_similarity': 0, }
surfall_avg = 0

for entity in entities:
    _query_occurrences = {
        'query': {
            'match_phrase': {
                'lower': entity
            }
        }
    }

    _query_occurrences = es.search(index = "surfall_entities", doc_type = "entities", body = _query_occurrences, size = 1)
    entity_occurrences[entity] = bool(_query_occurrences['hits']['total'])
    
#     for hit in _query_occurrences['hits']['hits']:
#         print(entity, '- most similar to -', hit['_source']['lower'])

    in_wordnet = 1
    if not wordnet.synsets(entity):
        in_wordnet = 0

    filtered_word, pmi_data, pmi_method, ds_similarity, mt_similarity, ds_sim_50, \
    ds_sim_60, ds_sim_70, ds_sim_80, ds_sim_90, mt_sim_50, mt_sim_60, mt_sim_70, \
    mt_sim_80, mt_sim_90 = process_extracted_entities.filter_it(entity, embedding_model)

    print(filtered_word, int(bool(pmi_data)), int(bool(pmi_method)), 
          'in word2vec:', int(bool(ds_similarity)), int(bool(mt_similarity)), 
          'in surfall_index', int(bool(_query_occurrences['hits']['total'])))
    
    word2vec_avg['pmi_data'] += int(bool(pmi_data))
    word2vec_avg['pmi_method'] += int(bool(pmi_method))
    word2vec_avg['ds_similarity'] += int(bool(ds_similarity))
    word2vec_avg['mt_similarity'] += int(bool(mt_similarity))
    surfall_avg += int(bool(_query_occurrences['hits']['total']))
    
nr_entities = len(entities)
print('\n\nPercentage entities in pmi_data:', word2vec_avg["pmi_data"], "/", nr_entities, "(", round(float(word2vec_avg["pmi_data"]*100)/nr_entities, 1), "%)")
print('Percentage entities in pmi_method:', word2vec_avg["pmi_method"], "/", nr_entities, "(", round(float(word2vec_avg["pmi_method"]*100)/nr_entities, 1), "%)")
print('Percentage entities in ds_similarity index:', word2vec_avg["ds_similarity"], "/", nr_entities, "(", round(float(word2vec_avg["ds_similarity"]*100)/nr_entities, 1), "%)")
print('Percentage entities in mt_similarity index:', word2vec_avg["mt_similarity"], "/", nr_entities, "(", round(float(word2vec_avg["mt_similarity"]*100)/nr_entities, 1), "%)")
print('Percentage entities in surfall index:', surfall_avg, "/", nr_entities, "(", round(float(surfall_avg*100)/nr_entities, 1), "%)")

