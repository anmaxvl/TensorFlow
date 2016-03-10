import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

import time

from tensorflow.models.rnn.rnn import *

def gen_seq():
    x = np.arange(0., np.pi*12, .03)
    bell = np.exp(-(np.sin(x-np.pi/2)-2*np.pi)**2/9.)
    y = 100*np.sin(8*x)*bell
    y = np.reshape(y, (len(x), 1))

    return x, y

def gen_input(y, n_steps, offset, seq_width=10, lag=60):
    seq_input = []
    seq_target = []

    for i in range(offset, offset+n_steps):
        window = []
        for j in range(seq_width):
            if i+j+seq_width<len(y):
                window.append(y[i+j+seq_width])
            else:
                window.append(0)
        seq_input.append(window)
        if i+lag+seq_width < len(y):
            seq_target.append(y[i+lag+seq_width])
        else:
            seq_target.append(0)

    return np.reshape(np.array(seq_input), (-1, seq_width)), np.reshape(np.array(seq_target), (-1, 1))

def gen_freerun_batch(y, net_y, n_steps, window_size=10, lag=60):
    if len(y.shape) > 1:
        y = np.reshape(y, (-1,))
    if len(net_y.shape) > 1:
        net_y = np.reshape(net_y, (-1,))

    seq_input = []
    seq_target = []
    seq_width = window_size
    for i in range(lag):
        window = []
        for j in range(seq_width):
            if -1-lag+i < 0:
                window.append(y[-1-lag+i+j])
            else:
                window.append(net_y[-1-lag+i+j])

        seq_input.append(window)
        seq_target.append(net_y[i])

    for i in range(n_steps-lag):
        window=[]
        for j in range(seq_width):
            window.append(net_y[i+j])
        seq_input.append(window)
        seq_target.append(net_y[i+lag])

    return np.reshape(np.array(seq_input), (-1, seq_width)), np.reshape(np.array(seq_target), (-1, 1))

def main(unused_args):
    print unused_args
    #Generating some data
    x, y = gen_seq()
    n_steps = len(x)/2
    plt.plot(x, y)
    plt.show()
    seq_width = 10
    num_hidden = 10

    ### Model initialiation

    #random uniform initializer for the LSTM nodes
    initializer = tf.random_uniform_initializer(-.1, .1)

    #placeholders for input/target/sequence_length
    seq_input = tf.placeholder(tf.float32, [n_steps, seq_width])
    seq_target = tf.placeholder(tf.float32, [n_steps, 1.])
    early_stop = tf.placeholder(tf.int32)

    #making a list of timestamps for rnn input
    inputs = [tf.reshape(i, (1, seq_width)) for i in tf.split(0, n_steps, seq_input)]

    #LSTM cell
    cell = rnn_cell.LSTMCell(num_hidden, seq_width, initializer=initializer)

    initial_state = cell.zero_state(1, tf.float32)
    #feeding sequence to the RNN
    outputs, states = rnn(cell, inputs, initial_state=initial_state, sequence_length=early_stop)

    #outputs is a list, but we need a single tensor instead
    outputs = tf.reshape(tf.concat(1, outputs), [-1, num_hidden])

    #softmax layer weights
    softmax_w = tf.get_variable('softmax_w', [num_hidden, 1])
    softmax_b = tf.get_variable('softmax_b', [1])

    #final prediction
    output = tf.matmul(outputs, softmax_w) + softmax_b

    #squared error
    error = tf.pow(tf.reduce_sum(tf.pow(tf.sub(output, seq_target), 2)), .5)

    lr = tf.Variable(0., trainable=False, name='lr')

    #optimizer setup
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(error, tvars), 5.)
    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.apply_gradients(zip(grads, tvars))
    ### Model initialization DONE

    ###Let the training begin
    init = tf.initialize_all_variables()

    session = tf.Session()
    session.run(init)

    #training and testing data
    train_input, train_target = gen_input(y, n_steps, offset=0, seq_width=10, lag=60)
    test_input, test_target = gen_input(y, n_steps, offset=n_steps, seq_width=10, lag=60)

    feed = {early_stop:n_steps, seq_input:train_input, seq_target:train_target}

    #initial predictions on untrained model
    outs = session.run(output, feed_dict=feed)
    plt.figure(1)
    plt.plot(x[:n_steps], train_target, 'b-', x[:n_steps], outs[:n_steps], 'r-')
    plt.ion()
    plt.show()

    tf.get_variable_scope().reuse_variables()

    session.run(tf.assign(lr, 1.))
    saver = tf.train.Saver()

    is_training = True

    if is_training:
        #Training for 100 epochs
        for i in range(100):
            new_lr = 1e-2
            if i > 25:
                new_lr = 1e-2
            elif i > 50:
                new_lr = 5e-3
            elif i > 75:
                new_lr = 1e-4
            session.run(tf.assign(lr, new_lr))

            err, outs, _ = session.run([error, output, train_op], feed_dict=feed)

            print ('Epoch %d done. Error: %1.5f') % (i+1, err)
            plt.clf()
            plt.plot(x[:n_steps], train_target, 'b-', x[:n_steps], outs[:n_steps], 'r-')
            plt.draw()
            time.sleep(.1)

        #saving the model variables
        saver.save(session, 'sine-wave-rnn-'+str(num_hidden) + '-' + str(seq_width), global_step = 0)

    if not is_training:
        saver.restore(session, 'sine-wave-rnn-'+str(num_hidden) + '-' + str(seq_width) + '-0')

    plt.ioff()
    plt.figure(1)
    plt.clf()
    #model prediction on training data
    train_outs = session.run(output, feed_dict = feed)
    plt.plot(x[:n_steps], train_target[:n_steps], 'b-', x[:n_steps], train_outs[:n_steps], 'g--')
    #model prediction on test data
    feed = {seq_input:test_input, seq_target:test_target, early_stop:n_steps}
    test_outs = session.run(output, feed_dict=feed)

    #plotting
    plt.plot(x[n_steps:2*n_steps], test_outs, 'r--')
    plt.plot(x[n_steps:2*n_steps], test_target, 'b-')
    plt.show()


if __name__=='__main__':
    tf.app.run()