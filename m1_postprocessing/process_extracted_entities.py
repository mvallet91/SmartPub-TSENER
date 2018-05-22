from nltk.corpus import wordnet
from nltk.corpus import stopwords
from m1_postprocessing.filtering import normalized_pub_distance

ds_names = []
mt_names = []
# prnames = []

dataset_path = '/data/dataset_names.txt'
with open(dataset_path, "r") as file:
    for row in file.readlines():
        ds_names.append(row.strip())

method_path = '/data/method_names.txt'
with open(method_path, "r") as file:
    for row in file.readlines():
        mt_names.append(row.strip())


#
# proteinpath = '/data/protein_names.txt'
# with open(proteinpath, "r") as file:
#         for row in file.readlines():
#                     prnames.append(row.strip())


def is_int_or_float(s):
    """
    return 1 for int, 2 for float, -1 for not a number
    """
    try:
        float(s)
        return 1 if s.count('.') == 0 else 2
    except ValueError:
        return -1


def filter_it(ner_word, model):
    ds_sim_90 = 0
    ds_sim_80 = 0
    ds_sim_70 = 0
    ds_sim_60 = 0
    ds_sim_50 = 0

    mt_sim_90 = 0
    mt_sim_80 = 0
    mt_sim_70 = 0
    mt_sim_60 = 0
    mt_sim_50 = 0

    # pr_sim_90 = 0
    # pr_sim_80 = 0
    # pr_sim_70 = 0
    # pr_sim_60 = 0
    # pr_sim_50 = 0

    ner_word = ner_word.split()

    if len(ner_word) > 1:
        filter_by_wordnet = []
        filtered_words = [word for word in set(ner_word) if word not in stopwords.words('english')]

        for word in filtered_words:
            isint = is_int_or_float(word)
            if isint != -1:
                filtered_words.remove(word)

        for word in set(filtered_words):
            in_wordnet = 1
            inds = 0

            if not wordnet.synsets(word):
                in_wordnet = 0
                filter_by_wordnet.append(word)

        filtered_words = ' '.join(filtered_words)
        filtered_words = filtered_words.replace('(', '')
        filtered_words = filtered_words.replace(')', '')
        filtered_words = filtered_words.replace('[', '')
        filtered_words = filtered_words.replace(']', '')
        filtered_words = filtered_words.replace('{', '')
        filtered_words = filtered_words.replace('}', '')
        filtered_words = filtered_words.replace(',', '')
        lower_filtered_words = filtered_words.lower()
        filter_by_wordnet = ' '.join(filter_by_wordnet)
        pmi_data = normalized_pub_distance(filtered_words, 'dataset')
        pmi_method = normalized_pub_distance(filtered_words, 'method')

        ds_similarity = []
        mt_similarity = []

        for ds in ds_names:
            try:
                similarity = model.wv.similarity(ds, lower_filtered_words)
                ds_similarity.append(similarity)
                if similarity > 0.89:
                    ds_sim_90 = 1
                elif similarity > 0.79:
                    ds_sim_80 = 1
                elif similarity > 0.69:
                    ds_sim_70 = 1
                elif similarity > 0.59:
                    ds_sim_60 = 1
                elif similarity > 0.49:
                    ds_sim_50 = 1

            except:
                pass

        for mt in mt_names:
            try:
                similarity = model.wv.similarity(mt, lower_filtered_words)
                mt_similarity.append(similarity)
                if similarity > 0.89:
                    mt_sim_90 = 1
                elif similarity > 0.79:
                    mt_sim_80 = 1
                elif similarity > 0.69:
                    mt_sim_70 = 1
                elif similarity > 0.59:
                    mt_sim_60 = 1
                elif similarity > 0.49:
                    mt_sim_50 = 1

            except:
                pass

        try:
            mt_similarity = float(sum(mt_similarity)) / len(mt_similarity)
        except:
            mt_similarity = 0

        try:
            ds_similarity = float(sum(ds_similarity)) / len(ds_similarity)
        except:
            ds_similarity = 0

    else:
        isint = is_int_or_float(ner_word)
        if isint == -1:
            filtered_words = ner_word.replace('(', '')
            filtered_words = filtered_words.replace(')', '')
            filtered_words = filtered_words.replace('[', '')
            filtered_words = filtered_words.replace(']', '')
            filtered_words = filtered_words.replace('{', '')
            filtered_words = filtered_words.replace('}', '')
            pmi_data = normalized_pub_distance(filtered_words, 'dataset')
            pmi_method = normalized_pub_distance(filtered_words, 'method')
            ds_similarity = []
            mt_similarity = []

            for ds in ds_names:
                try:
                    similarity = model.wv.similarity(ds, filtered_words.lower())
                    ds_similarity.append(similarity)
                    if similarity > 0.89:
                        ds_sim_90 = 1
                    elif similarity > 0.79:
                        ds_sim_80 = 1
                    elif similarity > 0.69:
                        ds_sim_70 = 1
                    elif similarity > 0.59:
                        ds_sim_60 = 1
                    elif similarity > 0.49:
                        ds_sim_50 = 1

                except:
                    pass

            for mt in mt_names:
                try:
                    similarity = model.wv.similarity(mt, filtered_words.lower())
                    mt_similarity.append(similarity)
                    if similarity > 0.89:
                        mt_sim_90 = 1
                    elif similarity > 0.79:
                        mt_sim_80 = 1
                    elif similarity > 0.69:
                        mt_sim_70 = 1
                    elif similarity > 0.59:
                        mt_sim_60 = 1
                    elif similarity > 0.49:
                        mt_sim_50 = 1

                except:
                    pass

            try:
                mt_similarity = float(sum(mt_similarity)) / len(mt_similarity)
            except:
                mt_similarity = 0

            try:
                ds_similarity = float(sum(ds_similarity)) / len(ds_similarity)
            except:
                ds_similarity = 0

    return (filtered_words, pmi_data, pmi_method, ds_similarity, mt_similarity,
            ds_sim_50, ds_sim_60, ds_sim_70, ds_sim_80, ds_sim_90,
            mt_sim_50, mt_sim_60, mt_sim_70, mt_sim_80, mt_sim_90)
