import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

import time

from tensorflow.models.rnn.rnn import *

def main(unused_args):
    x = np.arange(0., np.pi*12, .03)
    bell = np.exp(-(np.sin(x-np.pi/2)-2*np.pi)**2/9.)

    y = 100*np.sin(4*x)*bell

    y = np.reshape(y, (len(x), 1))
    
    n_steps = len(x)/2
    seq_width = 10
    num_hidden = 20
    lag = 60

    yy = []
    ty = []
    for i in range(n_steps):
        window = []
        for j in range(seq_width):
            window.append(y[i+j])
        yy.append(window)
        ty.append(y[i+lag])

    yy = np.array(yy)
    yy = np.reshape(yy, (n_steps, seq_width))
    ty = np.array([ty,])
    ty = np.reshape(ty, (n_steps, 1))
    print ty.shape


    plt.plot(x[:n_steps], yy)
    plt.show()
    print yy.shape

    initializer = tf.random_uniform_initializer(-.1, .1)

    seq_input = tf.placeholder(tf.float32, [n_steps, seq_width])
    seq_target = tf.placeholder(tf.float32, [n_steps, 1.])
    early_stop = tf.placeholder(tf.int32)

    inputs = [tf.reshape(i, (1, seq_width)) for i in tf.split(0, n_steps, seq_input)]

    cell = rnn_cell.LSTMCell(num_hidden, seq_width, initializer=initializer)

    initial_state = cell.zero_state(1, tf.float32)
    outputs, states = rnn(cell, inputs, initial_state=initial_state, sequence_length=early_stop)

    outputs = tf.reshape(tf.concat(1, outputs), [-1, num_hidden])

    softmax_w = tf.get_variable('softmax_w', [num_hidden, 1])
    softmax_b = tf.get_variable('softmax_b', [1])

    output = tf.matmul(outputs, softmax_w) + softmax_b

    error = tf.pow(tf.reduce_sum(tf.pow(tf.sub(output, seq_target), 2)), .5)

    lr = tf.Variable(0., trainable=False, name='lr')

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(error, tvars), 5.)
    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    init = tf.initialize_all_variables()

    session = tf.Session()
    session.run(init)

    feed = {early_stop:n_steps, seq_input:yy[:n_steps], seq_target:ty[:n_steps]}
    outs = session.run(output, feed_dict=feed)
    plt.figure(1)
    plt.plot(x[:n_steps], ty[:n_steps], 'b-', x[:n_steps], outs[:n_steps], 'r-')
    plt.ion()
    plt.show()

    tf.get_variable_scope().reuse_variables()
    session.run(tf.assign(lr, 1.))

    for i in range(100):
        new_lr = 1e-2
        if i > 25:
            new_lr = 5e-3
        elif i > 50:
            new_lr = 1e-3
        elif i > 75:
            new_lr = 5e-4
        session.run(tf.assign(lr, new_lr))

        err, outs, _ = session.run([error, output, train_op], feed_dict=feed)

        print ('Epoch %d done. Error: %1.5f') % (i+1, err)
        plt.clf()
        plt.plot(x[:n_steps], ty[:n_steps], 'b-', x[:n_steps], outs[:n_steps], 'r-')
        plt.draw()
        time.sleep(.25)


    saver = tf.train.Saver()
    saver.save(session, 'sine-wave-rnn-'+str(num_hidden) + '-' + str(seq_width), global_step = 0)

    plt.ioff()
    plt.clf()

    plt.figure(1)
    plt.plot(x[:n_steps], ty[:n_steps], 'b-', x[:n_steps], outs[:n_steps], 'r-')

    outs, m_softmax_w = session.run([output, softmax_w], feed_dict=feed)

    plt.figure(2)
    plt.plot(x[:n_steps], ty[:n_steps], 'b-', x[:n_steps], outs[:n_steps], 'r-')
    plt.figure(3)
    plt.plot(np.arange(num_hidden), m_softmax_w)

    plt.show()


if __name__=='__main__':
    tf.app.run()