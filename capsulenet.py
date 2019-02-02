from net import Net
import tensorflow as tf


def train(model, supervisor, num_label, epoches=100):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = supervisor.managed_session(config=config)
    for epoch in range(epochs):
        if supervisor.should_stop():
            break
def main():
    '''entry point of the app
    '''
    tf.logging.info('starting of the app')
    n = Net()
    sv = tf.train.Supervisor(graph=n.graph, logdir=cfg.logdir, save_model_secs=0)

if __name__ == "__main__":
    pass