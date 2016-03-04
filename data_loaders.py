import numpy as np

import pandas.io.data as web
from pandas.io.api import read_csv

class StockDataLoader(object):
    pass


class TrafficDataLoader(object):
    def __init__(self, filename, max_norm):
        self._raw_data = None

        try:
            self._raw_data = read_csv(filename, header=0, index_col=0)
        except Exception as e:
            print str(e)
            return

        self._index = []
        self._values = []

        self._values = np.reshape(self._raw_data.values, (-1))
        self._index = self._raw_data.index

        self._normalize(max_norm)


    def _normalize(self, max_norm):
        logged = np.log(self._values)
        min_val = min(logged)
        max_val = max(logged)
        normalized = max_norm*(logged - min_val)/(max_val-min_val)
        self._values = normalized


    def eval_1st_diffs(self, start, n_steps):
        _1st_diffs = []

        for i in range(start, start+n_steps):
            if i < 1:
                _1st_diffs.append(0)
            else:
                _1st_diffs.append(self._values[i]-self._values[i-1])

        return _1st_diffs


    def eval_2nd_diffs(self, start, n_steps):
        _2nd_diffs = []

        for i in range(start, start+n_steps):
            if i < 2:
                _2nd_diffs.append(0)
            else:
                diff1 = self._values[i]-self._values[i-1]
                diff2 = self._values[i-1] - self._values[i-2]
                
                _2nd_diffs.append(diff1-diff2)

        return _2nd_diffs


    def get_rnn_input(self, config):
        start = config.start
        n_steps = config.n_steps
        use_1st_diffs = config.use_1st_diffs
        use_2nd_diffs = config.use_2nd_diffs
        window_size = config.window_size
        lag = config.lag

        seq_input = []

        seq_target = self._values[start+lag:start+lag+n_steps]

        for i in range(window_size):
            lagged_input = self._values[start:start+n_steps]
            seq_input.append(lagged_input)            
            if use_1st_diffs:
                _1st_diffs = self.eval_1st_diffs(start+i, n_steps)
                seq_input.append(_1st_diffs)
            if use_2nd_diffs:
                _2nd_diffs = self.eval_2nd_diffs(start+i, n_steps)
                seq_input.append(_2nd_diffs)

        seq_input_np = np.array(seq_input).T
        seq_target_np = np.array([seq_target,]).T

        return seq_input_np, seq_target_np


    @property
    def index(self):
        return self._index

    @property
    def values(self):
        return self._values

    @property
    def raw_data(self):
        return self._raw_data



