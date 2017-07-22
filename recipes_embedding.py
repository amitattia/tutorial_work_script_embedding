import analyze_events
import auxiliary
import embedding_model

TRAIN_SET_F = 'recipes/train_set'
EVENT_TAG = 'EVENT'
SCRIPT_END = '========='
PREDICATE_ENTRY = 'pred_entry'
ARGUMENTS_ENTRY = 'args_entry'


def event2feat(event):
    # e = {PREDICATE_ENTRY: event.pred(), ARGUMENTS_ENTRY: [arg[0][arg[1]] for arg in event._args]}
    # return e
    return [event.pred(), [arg[0][arg[1]] for arg in event._args]]


def read_event(line):
    event = analyze_events.TextEvent(line)
    return event2feat(event)


def read_recipe_file(filename, max_num_of_events=-1):
    with open(filename, 'r') as f:
        num_of_events = 0
        scripts = []
        script = []
        for line in f:
            if line.startswith(SCRIPT_END):
                if len(script) > 0:
                    scripts.append(script)
                    script = []
                    num_of_events += 1
                    if num_of_events == max_num_of_events:
                        break
            else:
                if line.startswith(EVENT_TAG):
                    script.append(read_event(line))
                else:
                    continue
    return scripts


def scripts2words(scripts):
    words = []
    for script in scripts:
        for event in script:
            words.append(event[0])
            for arg in event[1]:
                words.append(arg)
    return words


def main():
    scripts = read_recipe_file(TRAIN_SET_F, 100)
    train_scripts = scripts[:int(0.9*len(scripts))]
    test_scripts = scripts[int(0.9*len(scripts)):]
    words = scripts2words(train_scripts)
    dictionary, r_dictionary = auxiliary.build_dict(words)
    train_set = auxiliary.build_set(train_scripts, dictionary)
    test_set = auxiliary.build_set(test_scripts, dictionary)
    # test_set = train_set
    accuracy = embedding_model.test_scenario(len(dictionary), train_set, test_set)
    print('accuracy is: %.2f' % accuracy)


if __name__ == "__main__":
    main()
