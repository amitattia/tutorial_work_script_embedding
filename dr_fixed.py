import xml.etree.ElementTree as ET
import collections
import head_of_argument

DOT_XML = '.xml'
SCRIPTS_FILE = 'segmented'
EVENT_ORDERING = 'event-ordering'
DEF_VOCABULARY_SIZE = 10**10


def print_root_info(root, tab=''):
    print(root.tag, root.attrib)
    nt = tab + '\t'
    for child in root:
        print(nt, end='')
        print_root_info(child, nt)


def read_xml(address):
    tree = ET.parse(address)
    root = tree.getroot()
    # print_root_info(root)
    return root


def read_scripts(path):
    scripts_node = read_xml(path + SCRIPTS_FILE + DOT_XML)
    scripts = []
    for script_node in scripts_node:
        events = []
        for event_node in script_node:
            args = []
            for arg_node in event_node:
                args.append(head_of_argument.get_head(arg_node.attrib['text']))
                # args.append(arg_node.attrib['text'].split()[-1])
            if len(args) == 0:
                args = ['imp_protagonist']
            events.append((event_node.attrib['text'], args))
        scripts.append(events)
    return scripts


def build_dict(words, vocabulary_size=DEF_VOCABULARY_SIZE):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary


def read_data(path, vocabulary_size=DEF_VOCABULARY_SIZE):
    scripts = read_scripts(path)
    words = []
    for script in scripts:
        for event in script:
            words.append(event[0])
            # words.extend(event[1][:1])
            words.append(event[1][0])
    dictionary, r_dictionary = build_dict(words, vocabulary_size)
    # print('#words/#events: %.3f ' % (len(dictionary) / sum(len(script) for script in scripts)))
    return scripts, dictionary, r_dictionary


def read_pairs(path):
    pairs_node = read_xml(path + EVENT_ORDERING + DOT_XML)
    pairs = []
    labels = []
    for pair_node in pairs_node:
        pair = []
        for event_node in pair_node:
            args = []
            for arg_node in event_node:
                args.append(head_of_argument.get_head(arg_node.attrib['text']))
                # args.append(arg_node.attrib['text'].split()[-1])
            if len(args) == 0:
                args = ['imp_protagonist']
            pair.append([event_node.attrib['text'], args])
        pairs.append(pair)
        labels.append(1 if pair_node.attrib['type'] == 'FOLLOWUP' else 0)
    return pairs, labels


def read_test_data(path):
    pairs, labels = read_pairs(path)
    return pairs, labels


def main():
    # scripts, dictionary, r_dictionary = read_data(r'data/dev/doorbell/')
    # print(scripts)
    s, d, r = read_data(r'data/dev/doorbell/')
    print(s[0:2])


if __name__ == "__main__":
    main()
