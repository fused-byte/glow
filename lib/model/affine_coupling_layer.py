import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import sys
import keras.backend as K
tfd = tfp.distributions

class NN(keras.layers.Layer):
    def __init__(self,
            output_shape,
            activation="relu",
            kernel_size = [[3,3], [1,1]],
            n_hidden=[512,512], 
            stride = [1,1],
            padding="SAME",
            name="nn"):
        if name:
            super(NN,self).__init__(name=name)
        else:
            super(NN,self).__init__()

        ### official implementation uses different initialisation for weights for the first 2 layers.
        ### random_normal_initialiser. ours is Xavier. 
        ### We are using Xavier because the paper doesn't talk about their initialisation method benefits as such 
        # and Xavier is used in most of the networks. However, we dont know the significance of this difference.
        self.conv1 = keras.layers.Conv2D(n_hidden[0], kernel_size = kernel_size[0], strides=stride, padding=padding, activation=activation, name=name+"/conv_1")
        self.conv2 = keras.layers.Conv2D(n_hidden[1], kernel_size = kernel_size[1], strides=stride, padding=padding, activation=activation, name=name+"/conv_2")
        self.conv3 = keras.layers.Conv2D(output_shape, kernel_size=[3,3], strides = stride,  kernel_initializer="zeros", padding=padding, name=name+"/conv_3")
        # self.conv3 = keras.layers.Conv2D(output_shape, kernel_size=[3,3], strides = stride, padding=padding, name=name+"/conv_3")

    def call(self,x):
        # print("weights: ", self.conv1.weights)
        # print("conv1 input: ", x.shape)
        x = self.conv1(x)
        # print("conv2 input: ", x.shape)
        x = self.conv2(x)
        # print("conv3 input: ", x.shape)
        x = self.conv3(x)
        # print("output: ", x.shape)
        return x


class ACL(tfp.bijectors.Bijector):
    def __init__(self, output_shape, 
            forward_min_event_ndims = 3, 
            validate_args=False,
            name="acl",
            **kwargs):

        super(ACL, self).__init__(
            validate_args=validate_args,
            forward_min_event_ndims=forward_min_event_ndims,
            name=name)

        self.output_shape = output_shape
        self.nn_obj = NN(self.output_shape[-1], name=name+"/NN", **kwargs)
        k_inp = self.output_shape.copy()
        k_inp[-1] = self.output_shape[-1] // 2
        
        x = tf.keras.Input(k_inp)
        # print("keras input: ", x.shape)
        h = self.nn_obj(x)
        self.nn = tf.keras.Model(x, [h], name=self.name + "/nn")
        
    def _forward(self, x):
        # print()
        x_a, x_b = tf.split(x, 2, axis = -1)
        y_b = x_b
        h = self.nn(x_b)
        t = h[:,:,:,0::2]
        # scale = keras.activations.sigmoid(h[:,:,:,1::2] + 2.)
        log_s = keras.activations.tanh(h[:,:,:,1::2])
        scale = tf.math.exp(log_s)
        y_a = scale * (x_a + t)
        y = tf.concat([y_a,y_b], axis=-1)
        return y

    def _inverse(self, y):
        y_a, y_b = tf.split(y, 2, axis = -1)
        # print('This is y_b shape : ', y_b.shape)
        h = self.nn(y_b)
        t = h[:,:,:,0::2]
        # scale = keras.activations.sigmoid(h[:,:,:,1::2] + 2.)
        log_s = keras.activations.tanh(h[:,:,:,1::2])
        scale = tf.math.exp(log_s)
        x_a = (y_a/scale) - t
        x_b = y_b
        x = tf.concat([x_a, x_b], axis = -1)
        return x

    def _forward_log_det_jacobian(self, x, event_ndims=3):
        _ , x_b = tf.split(x, 2, axis = -1)
        h = self.nn(x_b)
        log_s = keras.activations.tanh(h[:,:,:,1::2])
        scale = tf.math.exp(log_s)
        # scale = keras.activations.sigmoid(h[:,:,:,1::2] + 2.)
        # log_s = tf.math.log(scale)

        # print("log scale values", scale)
        # print("test: ", K.eval(tf.reduce_sum(log_s, axis = [1,2,3])))
        return tf.reduce_sum(tf.math.log(tf.math.abs(scale)), axis = [1,2,3])


def nn_test():
    nn = NN(4, name="NN")
    x = tf.random.normal([3, 32, 32, 4])
    x = tf.keras.Input([32,32,2])
    print(x.shape)
    log_s = nn(x)
    print(log_s)
    # Non trainable params: -> Batch Normalization's params
    tf.keras.Model(x, [log_s], name="nn_test").summary()

# nn_test()

def realnvp_test():
    realnvp = ACL(output_shape=[16, 16, 4], n_hidden=[256, 256])
    x = tf.keras.Input([16, 16, 4])

    y = realnvp.forward(x)
    print("trainable_variables :", len(realnvp.trainable_variables))
    print('trainable variable NN: ', len(realnvp.nn.trainable_variables))
    
    flow = tfd.TransformedDistribution(
        event_shape=[16, 16, 4],
        distribution=tfd.Normal(loc=0.0, scale=1.0),
        bijector=realnvp,
    )
    x = flow.sample(5)
    print(x.shape)
    # y = realnvp.inverse(x)
    log_prob = flow.log_prob(y)
    print(realnvp.inverse_log_det_jacobian(x, event_ndims=3).numpy() )
    print(
        x.shape,
        # y.shape,
        log_prob,
        # -tf.reduce_mean(log_prob),
        # -tf.reduce_mean(flow.distribution.log_prob(x)),
        # -tf.reduce_mean(
        #     flow.bijector.forward_log_det_jacobian(
        #         x, event_ndims=flow._maybe_get_static_event_ndims()
        #     )
        # ),
        # -tf.reduce_mean(flow._log_prob(x)),
        # flow._finish_log_prob_for_one_fiber(
        #     y,
        #     x,
        #     flow.bijector.forward_log_det_jacobian(
        #         x, event_ndims=flow._maybe_get_static_event_ndims()
        #     ),
        #     flow._maybe_get_static_event_ndims(),
        #
        # ),
        # tf.reduce_sum(flow.distribution.log_prob(
        #     flow._maybe_rotate_dims(x, rotate_right=True)),
        #               axis=flow._reduce_event_indices)
    )

realnvp_test()