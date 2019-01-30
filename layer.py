import numpy as np
import tensorflow as tf

class Layer:
    def __init__(self, size, num_out):
        self._size = size
        self._num_out = num_out
    
    def make(self, inp, batch_size):
        reshaped = tf.reshape(inp, (barch_size, -1,1, input.shape[-2].value, 1))