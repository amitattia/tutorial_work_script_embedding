from itertools import product

import numpy as np
import tensorflow as tf
import dr_fixed
import os
import recipes_embedding
import auxiliary

SLASH = r'/'

DEF_EMBEDDINGS_SIZE = 20
NUM_EPOCHS = 2000
DEF_VOCABULARY_SIZE = 1000
TR_size = 20
LEARNING_RATE = 0.005
REG_FACTOR = 0.1
USE_REG = False
DP = 0.2
BATCH_SIZE = 1000
EPOCH_PRINT = 10000

NUM_OF_SCRIPTS = (2**10)


def dict_append(l, word, d):
    if word in d:
        l.append(d[word])
    else:
        l.append(0)


def dict_append_test(l, word, d):
    if word in d:
        l.append(d[word])
        dict_append.h += 1
    else:
        l.append(0)
        dict_append.m += 1
    dict_append.it += 1
    if dict_append.it % 2**11 == 0 and False:
        print([dict_append.h,dict_append.m])
dict_append.h = 0
dict_append.m = 0
dict_append.it = 0


# dict_append.cnt += 1
#         if dict_append.cnt % 1 == 0:
#             print('cnt %d' % dict_append.cnt)
# dict_append.cnt = 0


def build_train_data(scripts, dictionary, max_args):
    p1_list = []
    a1_list = []
    p2_list = []
    a2_list = []
    labels = []
    for script in scripts:
        for i in range(len(script)):
            for j in range(i + 1, len(script)):
                dict_append(p1_list, script[i][0], dictionary)
                # dict_append(a1_list, script[i][1][0], dictionary)
                dict_append(p2_list, script[j][0], dictionary)
                # dict_append(a2_list, script[j][1][0], dictionary)
                a1 = []
                for w in script[i][1]:
                    dict_append(a1, w, dictionary)
                if len(a1) < max_args:
                    a1.extend([len(dictionary)] * (max_args - len(a1)))
                a1_list.append(a1)
                a2 = []
                for w in script[j][1]:
                    dict_append(a2, w, dictionary)
                if len(a2) < max_args:
                    a2.extend([len(dictionary)] * (max_args - len(a2)))
                a2_list.append(a2)
                # p1_list.append(dictionary[script[i][0]])
                # a1_list.append(dictionary[script[i][1][0]])
                # p2_list.append(dictionary[script[j][0]])
                # a2_list.append(dictionary[script[j][1][0]])
                labels.append(1)
    res = [np.array(p1_list), np.array(a1_list), np.array(p2_list), np.array(a2_list), np.array(labels)]
    # for i in range(len(res)):
    #     res[i] = res[i].reshape(res[i].shape[0], 1)
    res[4] = res[4].reshape(res[4].shape[0], 1)
    return res


def build_test_data(pairs, labels, dictionary, max_args):
    p1_list = []
    a1_list = []
    p2_list = []
    a2_list = []
    for pair in pairs:
        dict_append_test(p1_list, pair[0][0], dictionary)
        # dict_append(a1_list, pair[0][1][0], dictionary)
        dict_append_test(p2_list, pair[1][0], dictionary)
        # dict_append(a2_list, pair[1][1][0], dictionary)

        a1 = []
        for w in pair[0][1]:
            dict_append_test(a1, w, dictionary)
        if len(a1) < max_args:
            a1.extend([len(dictionary)] * (max_args - len(a1)))
        a1_list.append(a1)
        a2 = []
        for w in pair[1][1]:
            dict_append_test(a2, w, dictionary)
        if len(a2) < max_args:
            a2.extend([len(dictionary)] * (max_args - len(a2)))
        a2_list.append(a2)

    res = [np.array(p1_list), np.array(a1_list), np.array(p2_list), np.array(a2_list), np.array(labels)]
    res[4] = res[4].reshape(res[4].shape[0], 1)
    return res

'''
This is old implementation of the event_ordering_model, the newer is the eom below the commented implementation.
'''

# def event_ordering_model(path, embedding_size=DEF_EMBEDDINGS_SIZE, num_epochs=NUM_EPOCHS,
#                          vocabulary_size=DEF_VOCABULARY_SIZE, batch_size=BATCH_SIZE, dp=DP):
#     scripts, dictionary, r_dictionary = dr_fixed.read_data(path, vocabulary_size)
#     max_args = max(max(len(s[i][1]) for i in range(len(s))) for s in scripts)
#     train_data = build_train_data(scripts, dictionary, max_args)
#     pairs, labels = dr_fixed.read_test_data(path)
#     test_data = build_test_data(pairs, labels, dictionary, max_args)
#
#     # Define session
#     sess = tf.InteractiveSession()
#
#     # Define placeholders
#     p1 = tf.placeholder(tf.int32, shape=[None])
#     mult_a1 = tf.placeholder(tf.int32, shape=[None, max_args])
#     p2 = tf.placeholder(tf.int32, shape=[None])
#     mult_a2 = tf.placeholder(tf.int32, shape=[None, max_args])
#     y_ = tf.placeholder(tf.float32, shape=[None, 1])
#
#     # Define embedding
#     embeddings = tf.Variable(
#         tf.random_uniform([len(dictionary) + 1, embedding_size], -1.0, 1.0))
#     embed_p1 = tf.nn.embedding_lookup(embeddings, p1)
#     mult_embed_a1 = tf.nn.embedding_lookup(embeddings, mult_a1)
#     embed_p2 = tf.nn.embedding_lookup(embeddings, p2)
#     mult_embed_a2 = tf.nn.embedding_lookup(embeddings, mult_a2)
#
#     embed_a1 = tf.reduce_sum(mult_embed_a1, 1)
#     embed_a2 = tf.reduce_sum(mult_embed_a2, 1)
#
#     # Define graph
#     T = tf.Variable(tf.random_uniform([embedding_size, TR_size], -1.0, 1.0))
#     R = tf.Variable(tf.random_uniform([embedding_size, TR_size], -1.0, 1.0))
#     A = tf.Variable(tf.random_uniform([TR_size, 1], -1.0, 1.0))
#
#     keep_prob = tf.placeholder(tf.float32)
#     h_A_drop = tf.nn.dropout(A, keep_prob)
#
#     hidden_e1 = tf.nn.sigmoid(tf.matmul(embed_p1, R) + tf.matmul(embed_a1, T))
#     hidden_e2 = tf.nn.sigmoid(tf.matmul(embed_p2, R) + tf.matmul(embed_a2, T))
#     y = tf.nn.sigmoid(tf.matmul(hidden_e2, h_A_drop)) - tf.nn.sigmoid(tf.matmul(hidden_e1, h_A_drop))
#
#
#     # Define loss
#     # reg_factor = REG_FACTOR
#     # l2_loss = (tf.nn.l2_loss(embeddings)/(10*len(dictionary)) + tf.nn.l2_loss(T)/100 + tf.nn.l2_loss(R)/100 + tf.nn.l2_loss(A)/10) / 4
#     # if USE_REG:
#     #     reg = reg_factor * l2_loss
#     # else:
#     #     reg = 0
#     loss = tf.contrib.losses.hinge_loss((y + 1) / 2, y_)
#     avg_loss = tf.reduce_mean(loss)
#
#     # Training
#     c_lr = LEARNING_RATE
#     # train_step = tf.train.GradientDescentOptimizer(c_lr).minimize(loss)
#     train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
#     sess.run(tf.global_variables_initializer())
#     last_loss = 10 ** 10
#     for step in range(num_epochs):
#         if step%EPOCH_PRINT==EPOCH_PRINT-1:
#             print('epoch', step)
#         num_batch = train_data[0].shape[0] // batch_size
#         num_batch = 1
#         cl = 0
#         for _ in range(num_batch):
#             idx = np.random.randint(train_data[0].shape[0], size=batch_size)
#             # if step == num_steps / 2:
#             #     train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
#
#             _, loss_val = sess.run([train_step, avg_loss], feed_dict={p1: train_data[0][idx], mult_a1: train_data[1][idx], p2: train_data[2][idx], mult_a2: train_data[3][idx],
#                                                                       y_: train_data[4][idx], keep_prob: dp})
#             cl += loss_val
#         # if last_loss < cl:
#         #     c_lr /= 1.2
#         #     train_step = tf.train.GradientDescentOptimizer(c_lr).minimize(loss)
#         # last_loss = cl
#
#             # train_step.run(
#             #     feed_dict={p1: train_data[0], mult_a1: train_data[1], p2: train_data[2], mult_a2: train_data[3],
#             #                y_: train_data[4]})
#         # test_data = train_data
#
#     prediction = tf.cast((tf.sign(y) + 1) // 2, tf.int64)
#     correct_prediction = tf.equal(test_data[4], prediction)
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     acc_eval = accuracy.eval(
#         feed_dict={p1: test_data[0], mult_a1: test_data[1], p2: test_data[2], mult_a2: test_data[3], keep_prob: 1})
#     return acc_eval
#
#     # print('accuracy is %.3f' % acc_eval)


def eom(scripts, embedding_size=DEF_EMBEDDINGS_SIZE, num_epochs=NUM_EPOCHS,
                         vocabulary_size=DEF_VOCABULARY_SIZE, batch_size=BATCH_SIZE, dp=DP):
    words = recipes_embedding.scripts2words(scripts)
    dictionary, r_dictionary = auxiliary.build_dict(words)
    max_args = max(max(len(s[i][1]) for i in range(len(s))) for s in scripts)
    train_size = int(0.9*NUM_OF_SCRIPTS)
    train_data = build_train_data(scripts[:train_size], dictionary, max_args)
    test_data = build_train_data(scripts[train_size:], dictionary, max_args)
    # test_data = train_data
    # pairs, labels = dr_fixed.read_test_data(path)
    # test_data = build__data(pairs, labels, dictionary, max_args)

    # Define session
    sess = tf.InteractiveSession()

    # Define placeholders
    p1 = tf.placeholder(tf.int32, shape=[None])
    mult_a1 = tf.placeholder(tf.int32, shape=[None, max_args])
    p2 = tf.placeholder(tf.int32, shape=[None])
    mult_a2 = tf.placeholder(tf.int32, shape=[None, max_args])
    y_ = tf.placeholder(tf.float32, shape=[None, 1])

    # Define embedding
    embeddings = tf.Variable(
        tf.random_uniform([len(dictionary) + 1, embedding_size], -1.0, 1.0))
    embed_p1 = tf.nn.embedding_lookup(embeddings, p1)
    mult_embed_a1 = tf.nn.embedding_lookup(embeddings, mult_a1)
    embed_p2 = tf.nn.embedding_lookup(embeddings, p2)
    mult_embed_a2 = tf.nn.embedding_lookup(embeddings, mult_a2)

    embed_a1 = tf.reduce_sum(mult_embed_a1, 1)
    embed_a2 = tf.reduce_sum(mult_embed_a2, 1)

    # Define graph
    T = tf.Variable(tf.random_uniform([embedding_size, TR_size], -1.0, 1.0))
    R = tf.Variable(tf.random_uniform([embedding_size, TR_size], -1.0, 1.0))
    A = tf.Variable(tf.random_uniform([TR_size, 1], -1.0, 1.0))

    keep_prob = tf.placeholder(tf.float32)
    h_A_drop = tf.nn.dropout(A, keep_prob)

    hidden_e1 = tf.nn.sigmoid(tf.matmul(embed_p1, R) + tf.matmul(embed_a1, T))
    hidden_e2 = tf.nn.sigmoid(tf.matmul(embed_p2, R) + tf.matmul(embed_a2, T))
    y = tf.nn.sigmoid(tf.matmul(hidden_e2, h_A_drop)) - tf.nn.sigmoid(tf.matmul(hidden_e1, h_A_drop))

    # Define loss
    # reg_factor = REG_FACTOR
    # l2_loss = (tf.nn.l2_loss(embeddings)/(10*len(dictionary)) + tf.nn.l2_loss(T)/100 + tf.nn.l2_loss(R)/100 + tf.nn.l2_loss(A)/10) / 4
    # if USE_REG:
    #     reg = reg_factor * l2_loss
    # else:
    #     reg = 0
    # loss = tf.contrib.losses.hinge_loss((y + 1) / 2, y_)
    # loss = tf.contrib.losses.hinge_loss(y_, y)
    loss = tf.maximum(0.0, 1.0 - (2.0*y_-1.0) * y)
    avg_loss = tf.reduce_mean(loss)

    # Training
    c_lr = LEARNING_RATE
    # train_step = tf.train.GradientDescentOptimizer(c_lr).minimize(loss)
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    sess.run(tf.global_variables_initializer())
    last_loss = 10 ** 10
    for step in range(num_epochs):
        if step % EPOCH_PRINT == EPOCH_PRINT - 1:
            print('epoch', step)
        num_batch = train_data[0].shape[0] // batch_size
        num_batch = 1
        cl = 0
        for _ in range(num_batch):
            idx = np.random.randint(train_data[0].shape[0], size=batch_size)
            # if step == num_steps / 2:
            #     train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

            _, loss_val = sess.run([train_step, avg_loss],
                                   feed_dict={p1: train_data[0][idx], mult_a1: train_data[1][idx], p2: train_data[2][idx],
                                              mult_a2: train_data[3][idx],
                                              y_: train_data[4][idx], keep_prob: dp})
            cl += loss_val
            # if last_loss < cl:
            #     c_lr /= 1.2
            #     train_step = tf.train.GradientDescentOptimizer(c_lr).minimize(loss)
            # last_loss = cl

            # train_step.run(
            #     feed_dict={p1: train_data[0], mult_a1: train_data[1], p2: train_data[2], mult_a2: train_data[3],
            #                y_: train_data[4]})
            # test_data = train_data

    prediction = tf.cast((tf.sign(y) + 1) // 2, tf.int64)
    correct_prediction = tf.equal(test_data[4], prediction)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc_eval = accuracy.eval(
        feed_dict={p1: test_data[0], mult_a1: test_data[1], p2: test_data[2], mult_a2: test_data[3], keep_prob: 1})
    return acc_eval

def small_set():
    path = r'data/test/'
    dirs = list(os.walk(path))[0][1]
    accs = []
    for dir in dirs:
        accuracy = event_ordering_model(path + dir + SLASH)
        accs.append(accuracy)
        print(dir, 'accuracy is %.3f' % accuracy)
    print('\naverage accuracy is %.3f\n' % (sum(accs) / len(accs)))

RECIPE_DIR_PATH = 'recipes'
TRAIN_SET_NAME = 'train_set'
TEST_SET_NAME = 'test_set'
SLASH = '/'


def recipes():
    scripts = recipes_embedding.read_recipe_file(RECIPE_DIR_PATH+SLASH+TRAIN_SET_NAME, NUM_OF_SCRIPTS)
    print('accuracy is: %.2f' % eom(scripts))


def main():
    # recipes()
    small_set()

if __name__ == "__main__":
    main()
