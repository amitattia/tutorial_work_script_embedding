import nltk

AFTER_ARG = r'"><'
BEFORE_ARG = r'<ptcp text="'

def before_tag(s, tag):
    index = s.find(tag)
    return s[index+len(tag):]
    
def after_tag(s, tag):
    index = s.find(tag)
    return s[:index]

def get_head(s):
    spl = s.split()
    if len(spl) == 0:
        return 'imp_protagonist'
    if len(spl) == 1:
        return spl[-1]
    pos = nltk.pos_tag(nltk.word_tokenize(s))
    if 'NN' in pos[-1][-1]:
        return spl[-1]
    for p in pos:
        if 'NN' in p[-1]:
            return p[0]
    return 'UNK_HEAD'

def main():
    path = 'dev/omelette/segmented.xml'
    f = open(path, 'r')

    args = []
    for line in f.readlines():
        if 'ptcp' in line:
            args.append(after_tag(before_tag(line, BEFORE_ARG), AFTER_ARG))

    MAX_INDEX = len(args)
    # print(' '.join(s for s in args[:MAX_INDEX]))
    # print()
    for s in args[:MAX_INDEX]:
        if len(s.split()) > 1:
            print(s,'head is:',get_head(s))
            # print(s)
            # pos = nltk.pos_tag(nltk.word_tokenize(s))
            # if(('NN' not in pos[-1][-1])):
            #     print(pos)

if __name__ == "__main__":
    main()
