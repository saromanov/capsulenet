import tensorflow as tf
from squash import squash

def routing(inp, bij, inp_size=10, out_size=10, iters=5):
    '''implementation of routing algorithm
    return: coupling coefficient
    '''
    inp = tf.tile(inp, [1, 1, num_dims * num_outputs, 1, 1])
    u_hat = reduce_sum(W * input, axis=3, keepdims=True)
    for i in range(iters):
        coeff = tf.softmax(bij)
        res = tf.math.reduce_sum(tf.multiply(coeff, u_hat), 1, keepdims=True)
        res_sq = squash(res)
    return res_sq

