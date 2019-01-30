import tensorflow as tf
from layer import Layer

class Net:
    def __init__(self, width, height, channels):
        self._width = width
        self._height = height
        self._channels = channels
        self._graph = tf.Graph()
        primaryCaps = Layer(8, 32, with_routing=True)