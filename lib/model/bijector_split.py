import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors

class bijector_split(tfb.Bijector):
    def __init__(self,forward_min_event_ndims=0, name="split"):
        super(bijector_split, self).__init__(name=name, forward_min_event_ndims=forward_min_event_ndims,
            is_constant_jacobian = True)

    def _forward(self, x):
        z1, z2 = tf.split(x, 2, axis=-1)
        return z1,z2

    def _inverse(self, z):
        z = tf.concat([z[0],z[1]], axis=-1)
        return z

    def _forward_log_det_jacobian(self, x, event_ndims=0):
        return tf.constant(0.0)

        