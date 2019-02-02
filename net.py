import tensorflow as tf
from layer import Layer
from utils import reshape

class Net:
    def __init__(self, width, height, channels):
        self._width = width
        self._height = height
        self._channels = channels
        self._reg_scale = 0.392
        self._graph = tf.Graph()
        primaryCaps = Layer(8, 32, with_routing=True)
    
    def loss(m_plus, m_minus, length, T_c):
        max_len = tf.square(tf.maximum(0., m_plus - length))
        max_r = tf.square(tf.maximum(0., length - m_minus))
        max_len = reshape(max_len, (cfg.batch_size, -1))
        max_r = reshape(max_r, (cfg.batch_size, -1))
        L_c = T_c * max_len + lambda_val * (1 - T_c) * max_r
        return self._margin_loss(L_c) + self._reconstruction_loss(L_c) * self._reg_scale
    
    def _margin_loss(self, X):
        return tf.reduce_mean(tf.reduce_sum(X, axis=1))
    
    def _reconstruction_loss(self, X):
        orgin = reshape(X, (batch_size, -1))
        return tf.reduce_mean(tf.square(self.decoded - orgin))
    