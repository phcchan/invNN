import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# This enables a ctr-C without triggering errors
import signal, sys
signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

import numpy as np
import tensorflow as tf

from keras.layers import Layer, Dense, Lambda, Input
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.initializers import Constant


####################################################


class InvDense_LU(Layer):
    backwardPass = False # static variable for debug
    def __init__(self, isInverse=False, **kwargs):
        super().__init__(**kwargs)
        self.isInverse = isInverse
        self.kernel = None
    def initializer(self, shape):
        import scipy as sp
        import scipy.linalg
        random_matrix = sp.random.randn(shape[-1], shape[-1])
        random_orthogonal = sp.linalg.qr(random_matrix)[0]
        p, l, u = sp.linalg.lu(random_orthogonal)
        u_diag_sign = sp.sign(sp.diag(u))
        u_diag_abs_log = sp.log(abs(sp.diag(u))) 
        l_mask = 1 - sp.tri(shape[-1]).T
        u_mask = 1 - sp.tri(shape[-1])
        return p, l, u, u_diag_sign, u_diag_abs_log, l_mask, u_mask
    def build(self, input_shape):
        if self.kernel is None:
            (p, l, u, u_diag_sign, u_diag_abs_log,
                l_mask, u_mask) = self.initializer(input_shape)
            self.kernel_p = self.add_weight(name='kernel_p',
                                            shape=p.shape,
                                            initializer=lambda _: p,
                                            trainable=False)
            self.kernel_l = self.add_weight(name='kernel_l',
                                            shape=l .shape,
                                            initializer=lambda _: l,
                                            trainable=True)
            self.kernel_u = self.add_weight(name='kernel_u',
                                            shape=u.shape,
                                            initializer=lambda _: u,
                                            trainable=True)
            self.kernel_u_diag_sign = self.add_weight(name='kernel_u_diag_sign',
                                                         shape=u_diag_sign.shape,
                                                         initializer=lambda _: u_diag_sign,
                                                         trainable=False)
            self.kernel_u_diag_abs_log = self.add_weight(name='kernel_u_diag_abs_log',
                                                         shape=u_diag_abs_log.shape,
                                                         initializer=lambda _: u_diag_abs_log,
                                                         trainable=True)
            self.kernel_l = self.kernel_l * l_mask + K.eye(input_shape[-1])
            self.kernel_u = self.kernel_u * u_mask + K.tf.diag(
                self.kernel_u_diag_sign * K.exp(self.kernel_u_diag_abs_log))
            self.kernel = K.dot(K.dot(self.kernel_p, self.kernel_l),
                                self.kernel_u)
    def call(self, inputs):
        # dlogdet = tf.linalg.LinearOperator(self.kernel).log_abs_determinant()
        dlogdet = tf.cast(tf.log(abs(tf.matrix_determinant(
                    tf.cast(self.kernel, 'float64')))), 'float32')
        if not self.isInverse:
            x = tf.matmul(inputs, self.kernel)
            self.add_loss(-dlogdet)
        else:
            k_inv = tf.matrix_inverse(self.kernel)
            x  = tf.matmul(inputs, k_inv)

            # never used! keep for reference
            #self.add_loss( dlogdet) # pylint: disable=E1130
            assert not InvDense_LU.backwardPass, 'can only be used in backward pass!'
        return x 
    def inverse(self):
        layer = InvDense(not self.isInverse)
        layer.kernel = self.kernel
        # interesting! kernel can be assigned "="
        return layer

class InvDense(Layer):
    backwardPass = False # static variable for debug
    def __init__(self, isInverse=False, **kwargs):
        super().__init__(**kwargs)
        self.isInverse = isInverse
        self.kernel = None
    def build(self, input_shape):
        assert len(input_shape)==2
        k_shape = [input_shape[-1],input_shape[-1]]
        if self.kernel is None:
            k_init = np.linalg.qr(np.random.randn(
                *k_shape))[0].astype('float32')
            self.kernel = self.add_weight(name='kernel',
                                          shape=k_shape,
                                          initializer=lambda _: k_init,
                                          trainable=True)
    def call(self, inputs):
        # dlogdet = tf.linalg.LinearOperator(self.kernel).log_abs_determinant()
        dlogdet = tf.cast(tf.log(abs(tf.matrix_determinant(
                    tf.cast(self.kernel, 'float64')))), 'float32')
        if not self.isInverse:
            x = tf.matmul(inputs, self.kernel)
            self.add_loss(-dlogdet)
        else:
            k_inv = tf.matrix_inverse(self.kernel)
            x  = tf.matmul(inputs, k_inv)

            # never used! keep for reference
            #self.add_loss( dlogdet) # pylint: disable=E1130
            assert not InvDense.backwardPass, 'can only be used in backward pass!'
        return x 
    def inverse(self):
        layer = InvDense(not self.isInverse)
        layer.kernel = self.kernel
        # interesting! kernel can be assigned "="
        return layer



class Shuffle(Layer):
    ''' 2 shuffle modes: 'reverse' (default), 'random'
    '''
    def __init__(self, mode='reverse', **kwargs):
        super(Shuffle, self).__init__(**kwargs)
        self.idxs = None
        self.mode = mode
    def build(self, input_shape):
        in_dim = input_shape[-1]
        if self.idxs is None:
            if self.mode == 'reverse':
                self.idxs = self.add_weight(name='idxs',
                                            shape=(in_dim,),
                                            dtype='int32',
                                            initializer=self.reverse_initializer,
                                            trainable=False
                                            )
            elif self.mode == 'random':
                self.idxs = self.add_weight(name='idxs',
                                            shape=(in_dim,),
                                            dtype='int32',
                                            initializer=self.random_initializer,
                                            trainable=False
                                            )
            else:
                assert False, 'should NEVER get here!'
    def reverse_initializer(self, shape):
        idxs = list(range(shape[0]))
        return idxs[::-1]
    def random_initializer(self, shape):
        idxs = list(range(shape[0]))
        np.random.shuffle(idxs)
        return idxs
    def call(self, inputs):
        v_dim = K.int_shape(inputs)[-1]
        if self.idxs == None:
            self.idxs = list(range(v_dim))
            if self.mode == 'reverse':
                self.idxs = self.idxs[::-1]
            elif self.mode == 'random':
                np.random.shuffle(self.idxs)
        inputs = K.transpose(inputs)
        outputs = K.gather(inputs, self.idxs)
        # tf.gather gives UserWarning:
        # Converting sparse IndexedSlices to a dense Tensor of unknown shape. 
        # This may consume a large amount of memory.
        outputs = K.transpose(outputs)
        return outputs
    def inverse(self):
        in_dim = K.int_shape(self.idxs)[0]
        reverse_idxs = K.tf.nn.top_k(self.idxs, in_dim)[1][::-1]
        layer = Shuffle()
        layer.idxs = reverse_idxs
        return layer


class SplitVector(Layer):
    def __init__(self, **kwargs):
        super(SplitVector, self).__init__(**kwargs)
    def call(self, inputs):
        v_dim = K.int_shape(inputs)[-1]
        inputs = K.reshape(inputs, (-1, v_dim//2, 2))
        # -1 is batch_num in tensorflow
        return [inputs[:,:,0], inputs[:,:,1]]
    def compute_output_shape(self, input_shape):
        v_dim = input_shape[-1]
        return [(None, v_dim//2), (None, v_dim//2)]
        # None is batch_num in Keras
        # compute_output_shape() is derived from Keras::Layer
    def inverse(self):
        layer = ConcatVector()
        return layer


class ConcatVector(Layer):
    def __init__(self, **kwargs):
        super(ConcatVector, self).__init__(**kwargs)
    def call(self, inputs):
        # inputs is [x1, x2], each xi has dim (None, dim(xi))
        # expand_dims -> (None, dim(xi), 1)
        # concatenate -> (None, dim(x1)+dim(x2), 1)
        # reshape is similar to squeeze the last dim=1
        # it is simpler to use tf.stack and tf.squeeze
        assert len(inputs)==2, 'len(list)==2'
        inputs = [K.expand_dims(i, 2) for i in inputs]
        inputs = K.concatenate(inputs, 2)
        return K.reshape(inputs, (-1, np.prod(K.int_shape(inputs)[1:])))
    def compute_output_shape(self, input_shape):
        return (None, sum([i[-1] for i in input_shape]))
    def inverse(self):
        layer = SplitVector()
        return layer


class CoupleLayer(Layer):
    def __init__(self, isInverse=False, **kwargs):
        self.isInverse = isInverse
        super(CoupleLayer, self).__init__(**kwargs)
    def call(self, inputs):
        assert len(inputs)==4, 'len(list)==4'
        x1, x2, shift, log_s = inputs
        if not self.isInverse:
            logdet = -K.sum(K.mean(log_s, axis=0))
            self.add_loss(logdet)
            return [x1, x2*K.exp(log_s) + shift]
        else:
            logdet =  K.sum(K.mean(log_s, axis=0))
            self.add_loss(logdet)
            return [x1, (x2 - shift)*K.exp(-log_s)] 
    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1]]
    def inverse(self):
        return CoupleLayer(isInverse=not self.isInverse)


class CoupleWrapper():
    def __init__(self, isInverse=False):
        self.layer = CoupleLayer(isInverse=isInverse)
    def __call__(self, inputs, basicNN, inv=False):
        assert len(inputs)==2, 'len(list)==2'
        x1, x2 = inputs
        shift, log_s = basicNN(x1)
        if inv == False:
            return self.layer([x1, x2, shift, log_s])
        else:
            return self.layer.inverse()([x1, x2, shift, log_s])
    def inverse(self):
        return lambda inputs, basicNN: self(inputs, basicNN, inv=True)



class ShiftBias(Layer):
    def __init__(self, min_, s_, **kwargs):
        super().__init__(**kwargs)
        self._min = min_
        self._scale = s_
    def build(self, input_shape):
        self.bias = self.add_weight(
            name='bias',
            shape=input_shape[1:],
            initializer=Constant(value=self._min),
            trainable=False
        )
    def call(self, inputs):
        return K.tf.add(K.tf.multiply(inputs, self._scale), self.bias)

def build_NN(v_dim, 
    NN_depth = 3, NN_width = 256, 
    logs_min = -0.3, logs_max = 0.9
    # NN_width: better be power of 2, fit into RAM
):
    NN_scale = logs_max-logs_min
    NN_bias = logs_min
    _in = Input(shape=(v_dim,))
    x = _in
    for _ in range(NN_depth):
        x = Dense(NN_width, activation='relu')(x)
    shift = Dense(NN_width, activation='relu')(x)
    shift = Dense(v_dim)(shift)
    log_s = Dense(NN_width, activation='relu')(x)
    log_s = Dense(v_dim, activation='sigmoid')(log_s)
    log_s = ShiftBias(NN_bias, NN_scale)(log_s)
    return Model(_in, [shift, log_s])

def set_backwardpass(bool_var=True):
    InvDense.backwardpass = bool_var
    InvDense_LU.backwardpass = bool_var






