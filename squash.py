import tensorflow as tf


class IncorrectType(Exception):
    '''definition for errors when input type is incorrect
    '''
    pass

def squash(t, dim=-1):
    if not isinstance(t, tf.Tensor):
        raise IncorrectType('input data should be as tf.Tensor')
    squared_norm = (t**2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * tensor / tf.math.sqrt(squared_norm)