from net import Net
import tensorflow as tf

def main():
    '''entry point of the app
    '''
    tf.logging.info('starting of the app')
    n = Net()
    sv = tf.train.Supervisor(graph=n.graph, logdir=cfg.logdir, save_model_secs=0)

if __name__ == "__main__":
    pass