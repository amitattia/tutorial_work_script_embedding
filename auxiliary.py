import collections
import numpy as np

DEF_VOCABULARY_SIZE = 10 ** 10


def dict_append(l, word, d):
    if word in d:
        l.append(d[word])
    else:
        l.append(0)


def build_dict(words, vocabulary_size=DEF_VOCABULARY_SIZE):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary


def build_set(scripts, dictionary):
    p0_list = []
    a0_list = []
    p1_list = []
    a1_list = []
    labels = []
    for script in scripts:
        for i in range(len(script)):
            for j in range(i + 1, len(script)):
                dict_append(p0_list, script[i][0], dictionary)
                # dict_append(a1_list, script[i][1][0], dictionary)
                dict_append(p1_list, script[j][0], dictionary)
                # dict_append(a2_list, script[j][1][0], dictionary)
                a1 = []
                for w in script[i][1]:
                    dict_append(a1, w, dictionary)
                a0_list.append(a1)
                a2 = []
                for w in script[j][1]:
                    dict_append(a2, w, dictionary)
                a1_list.append(a2)
                labels.append(1)
    res = [np.array(p0_list), np.array(p1_list), [np.array(a) for a in a0_list], [np.array(a) for a in a1_list],
           np.array(labels)]
    res[4] = res[4].reshape(res[4].shape[0], 1)
    return res
