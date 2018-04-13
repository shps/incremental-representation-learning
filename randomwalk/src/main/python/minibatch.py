import numpy as np


class MiniBatch(object):
    def __init__(self, data, batch_size, seed):
        self.data_index = 0
        self.data_size = len(data)
        self.batch_size = batch_size
        np.random.seed(seed=seed)
        np.random.shuffle(data)
        self.batches = np.vsplit(data, self.data_size / self.batch_size)

    def next_batch(self):
        next_batch = self.batches[self.data_index]
        self.data_index += 1
        targets = next_batch[:, 0]
        contexts = next_batch[:, 1].reshape(len(next_batch), 1)
        return targets, contexts

    def end_of_batch(self):
        if self.data_index >= len(self.batches):
            return True
        else:
            return False

    def reset_batch_gen(self):
        self.data_index = 0

    def num_batches(self):
        return len(self.batches)
