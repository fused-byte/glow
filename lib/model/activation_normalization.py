import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

# tf.random.set_seed(0)
class ACT(tfp.bijectors.Bijector):

    # Ref - https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/Bijector
    def __init__(self, channels, validate_args=False, name='actnorm', log_scale_factor = 1.0):
        super(ACT, self).__init__(
          validate_args=validate_args,
          forward_min_event_ndims=1,
          name=name)
        self.log_scale_factor = log_scale_factor
        self.initialized = False
        self.log_scale = tf.Variable( tf.random.normal([channels]), name="logScale")
        self.bias = tf.Variable(tf.random.normal([channels]),name="bias")

    def actnorm_mean_var(self, x):
        mean = tf.reduce_mean(x, axis=[0, 1, 2])
        var = tf.reduce_mean((x-mean) ** 2, axis=[0, 1, 2])
        stdVar = tf.math.sqrt(var) + 1e-6
        log_scale = tf.math.log(1./ stdVar / self.log_scale_factor) * self.log_scale_factor
        # print('log scale shape: ', log_scale.shape)
        self.bias = -mean
        self.log_scale = log_scale

    def _forward(self, x):
        if not self.initialized:
            self.actnorm_mean_var(x)
            self.initialized = True
        return (x + self.bias) * tf.exp(self.log_scale)

    def _inverse(self, y):
        if not self.initialized:
            self.actnorm_mean_var(y)
            self.initialized = True
        return y * tf.exp(-self.log_scale) - self.bias

    ### Formula the same as in paper but this is producing values like -1825236.9 something
    ### for log_det_jcb which seems like a pretty huge value for the below test values. 
    def _forward_log_det_jacobian(self, x):
        shape = x.get_shape()
        # log_det = int(shape[1]) * int(shape[2])
        # print(log_det)
        # return log_det * tf.reduce_sum(self.log_scale)
        return tf.reduce_sum(self.log_scale)

    def _inverse_log_det_jacobian(self, y):
        # shape = y.get_shape()
        # log_det = int(shape[1]) * int(shape[2])
        # return - log_det * tf.reduce_sum(self.log_scale)
        # print(self.log_scale)
        # print(tf.reduce_sum(self.log_scale))
        #print('Test inverse value: ', tf.reduce_sum(self.log_scale))
        return -tf.reduce_sum(self.log_scale)

def test_actnorm():
    actnorm = ACT(4)
    x = tf.random.normal([100, 16, 16, 4]) + 100
    y = actnorm.forward(x)
    z = actnorm.inverse(y)
    print('input: x', tf.reduce_mean(x, axis=[0, 1, 2]).numpy())
    print('output: y', tf.reduce_mean(y, axis=[0, 1, 2]).numpy())
    print('inverse: z', tf.reduce_mean(z, axis=[0, 1, 2]).numpy())
    print('log_det_jacobian: ', actnorm.forward_log_det_jacobian(y, event_ndims=3).numpy()/256) ## this prints very large values
    print('log_det_jacobian: ', actnorm.inverse_log_det_jacobian(y, event_ndims=3).numpy()/256)
    print(tf.shape(y))
    print(tf.size(tf.shape(y)))
test_actnorm()
        