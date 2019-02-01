import tensorflow as tf
from layer import Layer
from utils import reshape

class Net:
    def __init__(self, width, height, channels):
        self._width = width
        self._height = height
        self._channels = channels
        self._graph = tf.Graph()
        primaryCaps = Layer(8, 32, with_routing=True)


def loss(m_plus, m_minus, length):
    max_len = tf.square(tf.maximum(0., m_plus - length))
    max_r = tf.square(tf.maximum(0., length - m_minus))
    max_len = reshape(max_len, (cfg.batch_size, -1))
    max_r = reshape(max_r, (cfg.batch_size, -1))
    