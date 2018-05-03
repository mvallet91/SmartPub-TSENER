import nltk


def generic_named_entities(file_path):
    """
    Obtains the generic entities from the sentences provided. This is because for the expansion strategies
    we only consider terms terms which are likely to be named entities by using NLTK entity detection, instead
    of all the words in the sentences.
    :param file_path:
    :return:
    """
    unlabelled_sentence_file = open(file_path, 'r', encoding='utf-8')
    text = unlabelled_sentence_file.read()
    print('Started to extract generic named entity from sentences...')
    sentences = nltk.sent_tokenize(text)
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
    chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)
    entity_names = []
    x = 0
    for tree in chunked_sentences:
        entity_names.extend(extract_entity_word(tree))
        x+=1
        if x % 1000 == 0:
            print('.', end='')
    print('Finished processing sentences with', len(entity_names), 'new possible entities')
    return entity_names


def extract_entity_word(t):
    """
    Recursively goes through the branches of the NLTK tagged sentences to extract the words tagged as entities
    :param t: NLTK tagged tree
    :return entity_names: a list of unique entity tokens
    """
    entity_names = []
    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_word(child))
    return set(entity_names)
