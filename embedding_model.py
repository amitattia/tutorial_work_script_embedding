import numpy as np
import tensorflow as tf

'''
TO DO:
Add dropout
Add regularization
Add decreasing learning rate
Fix feed_dict
'''

# Names
BATCH_SIZE = 'BATCH_SIZE'
EMBEDDING_SIZE = 'EMBEDDING_SIZE'
TRANSFORM_SIZE = 'TRANSFORM_SIZE'
LEARNING_RATE = 'LEARNING_RATE'
EPOCHS = 'EPOCHS'
TRAINING_PRINT_INTERVAL = 'TRAINING_PRINT_INTERVAL'

# Constants
MODEL_CONSTANTS = dict()
MODEL_CONSTANTS[BATCH_SIZE] = 10
MODEL_CONSTANTS[EMBEDDING_SIZE] = 10
MODEL_CONSTANTS[TRANSFORM_SIZE] = 10
TRAINING_CONSTANTS = dict()
TRAINING_CONSTANTS[LEARNING_RATE] = 0.005
# TRAINING_CONSTANTS[EPOCHS] = 2000
TRAINING_CONSTANTS[EPOCHS] = 100
TRAINING_CONSTANTS[TRAINING_PRINT_INTERVAL] = TRAINING_CONSTANTS[EPOCHS] / 10


def start_session():
    return tf.Session()


def close_session(session):
    session.close()
    return


def initialize_variables(session):
    session.run(tf.initialize_all_variables())
    # session.run(tf.global_variables_initializer())


def define_graph(dictionary_size, model_constants=MODEL_CONSTANTS):
    # Placeholders
    pred0 = tf.placeholder(tf.int32, shape=[model_constants[BATCH_SIZE]])
    pred1 = tf.placeholder(tf.int32, shape=[model_constants[BATCH_SIZE]])
    args0 = [tf.placeholder(tf.int32, shape=[None]) for _ in range(model_constants[BATCH_SIZE])]
    args1 = [tf.placeholder(tf.int32, shape=[None]) for _ in range(model_constants[BATCH_SIZE])]
    y_ = tf.placeholder(tf.float32, shape=[None, 1])

    # Embedding
    embedding_matrix = tf.Variable(tf.random_uniform([dictionary_size, model_constants[EMBEDDING_SIZE]], -1.0, 1.0))
    pred0_e = tf.nn.embedding_lookup(embedding_matrix, pred0)
    pred1_e = tf.nn.embedding_lookup(embedding_matrix, pred1)
    args0_e = [tf.nn.embedding_lookup(embedding_matrix, arg) for arg in args0]
    args1_e = [tf.nn.embedding_lookup(embedding_matrix, arg) for arg in args1]

    # Variables
    trans_pred = tf.Variable(
        tf.random_uniform([model_constants[EMBEDDING_SIZE], model_constants[TRANSFORM_SIZE]], -1.0, 1.0))
    trans_args = tf.Variable(
        tf.random_uniform([model_constants[EMBEDDING_SIZE], model_constants[TRANSFORM_SIZE]], -1.0, 1.0))
    summing = tf.Variable(tf.random_uniform([model_constants[TRANSFORM_SIZE], 1], -1.0, 1.0))

    # Dropout
    pass

    # Computations
    args0_mults = [tf.reduce_sum(tf.sigmoid(tf.matmul(arg, trans_args))) for arg in args0_e]
    args1_mults = [tf.reduce_sum(tf.sigmoid(tf.matmul(arg, trans_args))) for arg in args1_e]
    # hidden0 = tf.add_n(args0_mults) + tf.sigmoid(tf.matmul(pred0_e, trans_pred))
    # hidden1 = tf.add_n(args1_mults) + tf.sigmoid(tf.matmul(pred1_e, trans_pred))
    hidden0 = args0_mults + tf.sigmoid(tf.matmul(pred0_e, trans_pred))
    hidden1 = args1_mults + tf.sigmoid(tf.matmul(pred1_e, trans_pred))
    y = tf.nn.sigmoid(tf.matmul(hidden1, summing)) - tf.nn.sigmoid(tf.matmul(hidden0, summing))

    # Loss
    loss = tf.maximum(0.0, 1.0 - (2.0 * y_ - 1.0) * y)
    # loss = tf.contrib.losses.hinge_loss((y + 1) / 2, y_)
    avg_loss = tf.reduce_mean(loss)
    inputs = [pred0, pred1, args0, args1, y_, y]
    return inputs, loss, avg_loss


def train_model(session, inputs, loss, avg_loss, train_set, training_constants=TRAINING_CONSTANTS,
                model_constants=MODEL_CONSTANTS):
    train_step = tf.train.AdamOptimizer(training_constants[LEARNING_RATE]).minimize(loss)
    initialize_variables(session)
    # last_loss = 10 ** 10
    for step in range(training_constants[EPOCHS]):
        if (step + 1) % TRAINING_CONSTANTS[TRAINING_PRINT_INTERVAL] == 0:
            print(TRAINING_PRINT_INTERVAL, step+1, '%.2f' % ((step+1) / training_constants[EPOCHS]))
            print("loss is:", epoch_loss)
        num_batch = train_set[0].shape[0] // model_constants[BATCH_SIZE]
        epoch_loss = 0
        for _ in range(num_batch):
            idx = np.random.randint(train_set[0].shape[0], size=model_constants[BATCH_SIZE])
            feed = {inputs[0]: train_set[0][idx], inputs[1]: train_set[1][idx], inputs[4]: train_set[4][idx]}
            for i, d in zip(inputs[2], [train_set[2][i] for i in idx]):
                feed[i] = d
            for i, d in zip(inputs[3], [train_set[3][i] for i in idx]):
                feed[i] = d
            _, batch_loss = session.run([train_step, avg_loss],
                                        feed_dict=feed)
            epoch_loss += batch_loss
        # Optional decreasing learning rate
        # last_loss = epoch_loss
    return


def test_accuracy(session, inputs, test_set, training_constants=TRAINING_CONSTANTS, model_constants=MODEL_CONSTANTS):
    num_batch = test_set[0].shape[0] // model_constants[BATCH_SIZE]
    total_accuracy = 0

    num_batch = 100 // model_constants[BATCH_SIZE]

    for _ in range(num_batch):
        idx = np.random.randint(test_set[0].shape[0], size=model_constants[BATCH_SIZE])
        feed = {inputs[0]: test_set[0][idx], inputs[1]: test_set[1][idx]}
        for i, d in zip(inputs[2], [test_set[2][i] for i in idx]):
            feed[i] = d
        for i, d in zip(inputs[3], [test_set[3][i] for i in idx]):
            feed[i] = d
        prediction = tf.cast((tf.sign(inputs[5]) + 1) // 2, tf.int64)
        correct_prediction = tf.equal(test_set[4][idx], prediction)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_evaluation = session.run(accuracy, feed_dict=feed)
        total_accuracy += accuracy_evaluation
    return total_accuracy / num_batch

    feed = {inputs[0]: test_set[0], inputs[1]: test_set[1]}
    for i, d in zip(inputs[2], test_set[2]):
        feed[i] = d
    for i, d in zip(inputs[3], test_set[3]):
        feed[i] = d
    prediction = tf.cast((tf.sign(y) + 1) // 2, tf.int64)
    correct_prediction = tf.equal(test_set[4], prediction)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_evaluation = session.run(accuracy, feed_dict=feed)
    return accuracy_evaluation


def test_scenario(dictionary_size, train_set, test_set):
    session = start_session()
    inputs, loss, avg_loss = define_graph(dictionary_size)
    train_model(session, inputs, loss, avg_loss, train_set)
    accuracy = test_accuracy(session, inputs, test_set)
    close_session(session)
    # return 0
    return accuracy


def main():
    train_set, test_set = None, None
    session = start_session()
    inputs, loss, avg_loss = define_graph(train_set)
    train_model(session, inputs, loss, avg_loss, train_set)
    accuracy = test_accuracy(test_set)
    close_session(session)
    print('accuracy is: %.2f' % accuracy)
    return


if __name__ == '__main__':
    main()
