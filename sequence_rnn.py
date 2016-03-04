import tensorflow as tf
import numpy as np
from tensorflow.models.rnn.rnn import *

class SequenceRNN(object):
    """
    Base class for sequenced data.
    """
    def __init__(self):
        self._early_stop = None
        self._seq_input = None
        self._seq_target = None
        self._initial_state = None
        self._final_state = None
        self._output = None
        self._train_op = None


    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self._lr, lr_value))

    @property
    def early_stop(self):
        return self._early_stop

    @property
    def seq_input(self):
        return self._seq_input

    @property
    def seq_target(self):
        return self._seq_target

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def final_state(self):
        return self._final_state

    @property
    def error(self):
        return self._error

    @property
    def train_op(self):
        return self._train_op

    @property
    def lr(self):
        return self._lr

