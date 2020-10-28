import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions
tfb = tfp.bijectors

class Squeeze(tfb.Bijector):
    def __init__(self, factor=2, name='squeeze', forward_min_event_ndims=0):
        self.factor = factor
        #self.name = name
        super(Squeeze, self).__init__(name=name, is_constant_jacobian = True, forward_min_event_ndims=forward_min_event_ndims)
        

    def _forward(self, x):
        (h,w,c) = x.shape[1:]
        batch_size = x.shape[0]
        temp_shape = (batch_size, h//self.factor, self.factor, w//self.factor, self.factor, c)
        output_shape = (batch_size, h//self.factor, w//self.factor, c*self.factor*self.factor)
        transpose_permutation = [0, 1, 3, 5, 2, 4]
        x = tf.reshape(x, temp_shape)
        x = tf.transpose(x, transpose_permutation)
        x = tf.reshape(x, output_shape)
        return x

    def _inverse(self, y):
        (h,w,c) = y.shape[1:]
        batch_size = y.shape[0]
        temp_shape = (batch_size, h, w, c//(self.factor*self.factor), self.factor, self.factor)
        output_shape = (batch_size, h*self.factor, w*self.factor, c//(self.factor*self.factor))
        transpose_permutation = [0, 1, 4, 2, 5, 3]
        y = tf.reshape(y, temp_shape)
        y = tf.transpose(y, transpose_permutation)
        y = tf.reshape(y, output_shape)
        return y

    def _forward_log_det_jacobian(self, x, event_ndims=0):
        return tf.constant(0.0)
    
    def _inverse_log_det_jacobian(self, x, event_ndims=0):
        return tf.constant(0.0)

def test_squeeze():
    factor = 2
    x = tf.Variable([[[1, 2, 5, 6], [3, 4, 7, 8], [9, 10, 13, 14],
                      [11, 12, 15, 16]]])
    x = tf.expand_dims(x, axis=-1)
    squeeze = Squeeze(factor=factor)
    y = squeeze.forward(x)
    z = squeeze.inverse(y)
    print(tf.reduce_sum(x - z))

    flow = tfd.TransformedDistribution(event_shape=[16, 16, 2],
                                       distribution=tfd.Normal(loc=0.,
                                                               scale=1.),
                                       bijector=squeeze)
    x = tf.random.normal([64, 16, 16, 2])
    y = flow.bijector.forward(x)
    log_prob = flow.log_prob(y)
    print(x.shape, y.shape, log_prob.shape)
    print(squeeze._forward_log_det_jacobian(x))
    
# test_squeeze()
