import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from invNN_base import InvNN_base

#####################################
def gen_data(fn, multiple):
    ''' fn: filename for csv
        multiple: number of repeat, positive integer 
                  or corresponding list
    '''
    stk = pd.read_csv(fn, parse_dates=['Date'])
    d_logP_ = np.diff(np.log(stk.Close.values))

    def gen_data_int(m_):
        if m_ == 1:
            d_logP = d_logP_
        elif m_ > 1:
            d_logP = np.tile(d_logP_, m_)
        else:
            assert False, 'ERR: multiple should be int > 0'
        data_size = len(d_logP)
        norm_dist = tfd.Normal(loc=0, scale=1)
        norm_samples = norm_dist.sample(data_size)
        return tf.stack([d_logP, norm_samples], axis=1)

    if isinstance(multiple, int):
        data_ = gen_data_int(multiple)
    elif isinstance(multiple, (list, tuple)):
        assert all([isinstance(_, int) and _ > 0 for _ in multiple])
        data_ = [gen_data_int(_) for _ in multiple]
    else:
        assert False, 'Can only accept int or list of int'

    with tf.Session() as sess:
        return sess.run(data_)


def test_gen_data():
    fn_dir = 'data_options'
    fn_prefix = '0700_HK'
    fn = os.path.join(fn_dir, fn_prefix+'.csv')

    print('generating data... ', end='', flush=True)
    x_samples = gen_data(fn, multiple=3)
    print('finished')

    import matplotlib
    no_display = (os.environ.get('DISPLAY') is None)
    if no_display:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt 
     # input data, blue
    plt.scatter(x_samples[:,0], x_samples[:,1], color='b',s=1)

    plt.savefig('2Ddata.png')
    if not no_display:
        plt.show()


class InvNN_options(InvNN_base):
    def __init__(self, in_dim, **kwargs):
        super(InvNN_options, self).__init__(in_dim, **kwargs)
        self.build_encoder()
        if kwargs['fn_weights'] != None:
            self.build_decoder()
    def rand(self, gen_size):
        assert isinstance(gen_size, int)
        assert gen_size > 0

        if self.decoder == None:
            self.build_decoder()

        z_sample = np.array(np.random.randn(gen_size, self.in_dim))
        x_decoded = self.decoder.predict(z_sample)

        return x_decoded[:,0]
        # x_decoded[:,1] is the dummy normal distribution


if __name__ == '__main__':
    test_gen_data()


