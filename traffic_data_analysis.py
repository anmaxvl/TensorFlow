import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import datetime
import time

from tensorflow.models.rnn.rnn import *

from sequence_rnn import SequenceRNN
from data_loaders import TrafficDataLoader

class TrafficRNN(SequenceRNN):
    def __init__(self, is_training, config):
        seq_width = config.seq_width
        n_steps = config.batch_size
        num_hidden = config.num_hidden
        num_relu = config.num_relu
        num_layers = config.num_layers

        #tensors for input, target and sequence length placeholders
        self._seq_input = tf.placeholder(tf.float32, [n_steps, seq_width])
        self._seq_target = tf.placeholder(tf.float32, [n_steps, 1])
        self._early_stop = tf.placeholder(tf.int32)

        #inputs should be a list of tensors at each timestamp
        inputs = [tf.reshape(data, (1, seq_width)) for data in tf.split(0, n_steps, self.seq_input)]
        initializer = tf.random_uniform_initializer(-.1, .1)

        cell = rnn_cell.LSTMCell(num_hidden, seq_width, initializer=initializer)
        if num_layers > 1:
            cell = rnn_cell.MultiRNNCell([cell]*num_layers)

        #initial state
        self._initial_state = cell.zero_state(1, tf.float32)

        outputs, states = rnn(cell, inputs, initial_state=self._initial_state, sequence_length=self._early_stop)
        
        #save final state of the rnn
        self._final_state = states[-1]

        #outputs originaly comes as a list of tensors, but we need a single tensor for tf.matmul
        outputs = tf.reshape(tf.concat(1, outputs), [-1, num_hidden])

        #rnn outputs
        W = tf.get_variable('W', [num_hidden, 1])
        b = tf.get_variable('b', [1])
        _output = tf.matmul(outputs, W) + b
        self._output = _output

        #ops for least squares error computation
        error = tf.pow(tf.reduce_sum(tf.pow(tf.sub(_output, self._seq_target), 2)), .5)
        tf.scalar_summary("error", error)

        self._error = error
        self._merge_summaries_op = tf.merge_all_summaries()

        if not is_training:
            return

        #learning rate
        self._lr = tf.Variable(0., trainable='False', name='lr')

        #trainable variables for gradient computation
        tvars = tf.trainable_variables()
        #compute gradients
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._error, tvars), config.max_grad_norm)

        #2 options here: either to use GradientDescentOptimizer (config.useGDO:True) or AdamOptimizer (config.useGDO:False)
        if config.useGDO:
            optimizer = tf.train.GradientDescentOptimizer(self._lr)
        else:
            optimizer = tf.train.AdamOptimizer(self._lr)

        #ops for training
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    @property
    def output(self):
        return self._output

    @property
    def merge_summaries_op(self):
        return self._merge_summaries_op

class TrafficDataConfig(object):
    """
    For data sampled in 5 min intervals:
        start:          0 hours (offset time)
        window_size:    2 hours
        lag:            6 hours
        batch_size:     4 hours
        n_steps:        168 hours or 7 days
    """
    start = 0
    window_size = 36
    n_steps = 2016
    use_1st_diffs = True
    use_2nd_diffs = False
    lag = 72
    batch_size = 24

class TestConfig(object):
    """
    For data sampled in 5 min intervals:
        start:          168 hours or 7 days (offset time)
        window_size:    2 hours
        lag:            6 hours
        batch_size:     4 hours
        n_steps:        840 hours or 35 days
    """
    start = 2016
    window_size = 36
    n_steps = 10080
    use_1st_diffs = True
    use_2nd_diffs = False
    lag = 72
    batch_size = 24

class TrafficRNNConfig(object):
    max_epoch = 250
    num_hidden = 150
    num_layers = 1
    useGDO = False
    max_grad_norm = 3.
    num_relu = 40
    def __init__(self, config):
        self.batch_size = config.batch_size
        if config.use_1st_diffs and config.use_2nd_diffs:
            self.seq_width = 3*config.window_size
        elif config.use_1st_diffs and not config.use_2nd_diffs:
            self.seq_width = 2*config.window_size
        elif config.use_2nd_diffs and not config.use_1st_diffs:
            self.seq_width = 2*config.window_size
        else:
            self.seq_width = config.window_size

def run_epoch(session, m, data, eval_op, config, writer=None):
    state = m.initial_state.eval()

    seq_input = data['seq_input']
    seq_target = data['seq_target']
    early_stop = data['early_stop']

    epoch_error = 0.
    rnn_outs = np.array([])
    for i in range(config.n_steps/config.batch_size):
        _seq_input = seq_input[i*config.batch_size:(i+1)*config.batch_size][:]
        _seq_target = seq_target[i*config.batch_size:(i+1)*config.batch_size][:]
        _early_stop = early_stop
        feed = {m.seq_input:_seq_input, m.seq_target:_seq_target, m.early_stop:_early_stop, m.initial_state:state}

        summary_str, step_error, state, step_outs, _ = session.run([m.merge_summaries_op, m.error, m.final_state, m.output, eval_op], feed_dict=feed)
        epoch_error += step_error
        rnn_outs = np.append(rnn_outs, step_outs)
        if writer is not None:
            writer.add_summary(summary_str, i)

    return epoch_error, rnn_outs

def main(unused_args):
    tdLoader = TrafficDataLoader('internet-data/data/internet-traffic-11-cities-5min.csv', max_norm=5.)
    tdConfig = TrafficDataConfig()
    tmConfig = TrafficRNNConfig(tdConfig)
    batch_size = tmConfig.batch_size

    seq_input, seq_target = tdLoader.get_rnn_input(tdConfig)

    print seq_input.shape, seq_target.shape
    data = dict()
    data['seq_input'] = seq_input
    data['seq_target'] = seq_target
    data['early_stop'] = tdConfig.batch_size

    is_training = True
    save_graph = False

    with tf.Graph().as_default(), tf.Session() as session:
        model = TrafficRNN(is_training=True, config=tmConfig)

        saver = tf.train.Saver()
        merged = None
        writer = None

        if is_training and save_graph:
            writer = tf.train.SummaryWriter('/tmp/rnn_logs', session.graph_def)

        tf.initialize_all_variables().run()

        # if is_training and save_graph:
        #     tf.train.write_graph(session.graph_def, '/tmp/rnn_graph/', 'traffic_rnn_graph')
        if is_training:
            for epoch in range(tmConfig.max_epoch):
                lr_value = 1e-4
                if epoch > 50:
                    lr_value = 5e-5
                elif epoch > 100:
                    lr_value = 1e-6
                elif epoch > 200:
                    lr_value = 1e-7
                # elif epoch > 75:
                #     lr_value = 1e-4
                # elif epoch > 100:
                #     lr_value = 5e-5
                # elif epoch > 225:
                #     lr_value = 1e-5

                # lr_value = 1e-4

                model.assign_lr(session, lr_value)

                net_outs_all = np.array([])

                error, net_outs_all = run_epoch(session, model, data, model.train_op, tdConfig)
                error, net_outs_all = run_epoch(session, model, data, tf.no_op(), tdConfig, writer)
                print net_outs_all.shape, seq_target.shape
                print ('Epoch %d: %s') % (epoch, error)

                if epoch == 0:
                    plt.figure(1, figsize=(20,10))
                    plt.ion()
                    plt.ylim([-1, 6])
                    plt.plot(xrange(tdConfig.n_steps), seq_target, 'b-', xrange(tdConfig.n_steps), net_outs_all, 'r-')
                    plt.show()
                    time.sleep(20)
                else:
                    plt.clf()
                plt.ylim([-1, 6])
                plt.plot(xrange(tdConfig.n_steps), seq_target, 'b-', xrange(tdConfig.n_steps), net_outs_all, 'r-')
                img_loc = 'out-img/epoch-%05d.png' % (epoch)
                plt.savefig(img_loc)
                plt.draw()
                time.sleep(.1)

                if epoch > 40 and epoch % 20 == 9:
                    outfile = 'internet-data/saved-models/traffic-rnn-hid-%d-batch-%d-window-%d-lag-%d.chkpnt' % (tmConfig.num_hidden, 
                                                                                                                    tdConfig.batch_size, 
                                                                                                                    tdConfig.window_size, 
                                                                                                                    tdConfig.lag)
                    saver.save(session, outfile, global_step=epoch)
        else:
            saved_vars = 'internet-data/saved-models/traffic-rnn-hid-%d-batch-%d-window-%d-lag-%d.chkpnt-%d' % (tmConfig.num_hidden, 
                                                                                                                    tdConfig.batch_size, 
                                                                                                                    tdConfig.window_size, 
                                                                                                                    tdConfig.lag,
                                                                                                                    tmConfig.max_epoch-1)
            saver.restore(session, saved_vars)


        train_error, train_outs_all = run_epoch(session, model, data, tf.no_op(), tdConfig)

        testDataConfig = TestConfig()
        test_seq_input, test_seq_target = tdLoader.get_rnn_input(testDataConfig)

        test_data = dict()        
        test_outs_all = np.array([])
        test_data['seq_input'] = test_seq_input
        test_data['seq_target'] = test_seq_target
        test_data['early_stop'] = testDataConfig.batch_size
        test_error, test_outs_all = run_epoch(session, model, test_data, tf.no_op(), testDataConfig)

        upper_curve = test_outs_all + .1*test_outs_all
        lower_curve = test_outs_all - .1*test_outs_all
        shift_left = np.zeros(test_outs_all.shape)
        shift_right = np.zeros(test_outs_all.shape)
        shift_left[:-18] = test_outs_all[18:]
        shift_right[18:] = test_outs_all[:-18]

        curve1 = np.maximum(upper_curve, shift_left)
        curve1 = np.maximum(curve1, shift_left)
        curve1 = np.maximum(curve1, shift_right)
        curve2 = np.minimum(lower_curve, shift_right)
        curve2 = np.minimum(curve2, upper_curve)
        curve2 = np.minimum(curve2, shift_left)

        print test_outs_all.shape

        x = xrange(len(test_outs_all))
        plt.figure(3, figsize=(20, 10))
        plt.ioff()

        plt.plot(x, test_outs_all, 'b-', alpha=1)
        plt.plot(x, test_seq_target, 'g-', alpha=1)
        plt.plot(x, curve1, 'r-', alpha=.1)
        plt.plot(x, curve2, 'r-', alpha=.1)
        plt.fill_between(x, curve1, curve2, color='grey', alpha=.3)
        plt.show()

        print 'Test error: %s' % test_error
        # plt.figure(2, figsize=(20,10))
        # plt.plot(xrange(tdConfig.n_steps), seq_target, 'b-', xrange(tdConfig.n_steps), train_outs_all, 'g--')
        # plt.plot(xrange(tdConfig.n_steps-24, tdConfig.n_steps+testDataConfig.n_steps-24), test_seq_target, 'b-')
        # plt.plot(xrange(tdConfig.n_steps-24, tdConfig.n_steps+testDataConfig.n_steps-24), test_outs_all, 'r--')
        # plt.show()
        # time.sleep(1)


if __name__=='__main__':
    tf.app.run()