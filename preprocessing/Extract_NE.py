import nltk


def generic_named_entities(file_path):
    """

    :param file_path:
    :return:
    """
    unlabelled_sentence_file = open(file_path, 'r')
    text = unlabelled_sentence_file.read()
    print('started to extract general NE from text....')
    sentences = nltk.sent_tokenize(text)
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
    chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)
    entity_names = []
    for tree in chunked_sentences:
        entity_names.extend(extract_entity_names(tree))
    return entity_names


def extract_entity_names(t):
    """

    :param t:
    :return:
    """
    entity_names = []
    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))
    return set(entity_names)
