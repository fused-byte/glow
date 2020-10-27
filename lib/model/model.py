import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from . import squeeze, conv_1x1_inv, activation_normalization, affine_coupling_layer, bijector_split


tfb = tfp.bijectors

class model(tfb.Bijector):
    def __init__(self, L=3,
        K=32,
        epochs = 30,
        learning_rate = 1e-4,
        BATCH_SIZE=120,
        activation = "relu",
        input_size = [32,32,3],
        num_of_hidden = [512,512],
        name="model"
    ):

        super(model,self).__init__(
            validate_args=validate_args,
            forward_min_event_ndims=None,
            name=name)
        
        self.L = L
        self.K = K
        self.layers = []
        self.embedding_forward = []
        self.embedding_inverse = []
        self.input_shapes = self._get_input_shapes(input_size, L)
        self.unsqueeze_layers = []
        for i in range(L-1):

            squeeze = squeeze.Squeeze(2, name=name+"/squeeze_{}".format(i))
            self.layers.append(squeeze)
            flow_steps = self._flow_steps(K, input_shape=self.input_shapes[i], name=name+"/flow_step_{}".format(i))
            self.layers = self.layers + flow_steps
            if i < L-1:
                split = bijector_split.bijector_split(0, name=name+"/split_{}".format(i))
                self.layers.append(split)
        for i in range(L):
            squeeze_invert = tfb.Invert(squeeze.Squeeze(2, name=name+"/unsqueeze_{}".format(L-i)))
            self.unsqueeze_layers.append(squeeze_invert)

    def _forward(self, x):
        embedding = []
        for i in range(len(self.layers)):
            if i % K+1 == 0:
                z1, z2 = self.layers[i].forward(x)
                embedding.append(z1)
                x = z2
            else:
                x = self.layers[i].forward(x)

        self.embedding_forward = embedding
        
        for i in range(len(self.unsqueeze_layers)):
            x = self.unsqueeze_layers[i].forward(x)
            if i < self.L-1:
                x = tf.concat([embedding[self.L-2-i], x], axis=-1)
        return x

    def _inverse(self, x):
        embedding = []
        for i in range(len(self.unsqueeze_layers)):
            x = self.unsqueeze_layers[i].inverse(x)
            if i < L-1:
                z1, z2 = tf.split(x, 2, axis = -1)
                embedding.append(z1)
                x = z2
        
        self.embedding_inverse = embedding
        embedding_ptr = len(self.embedding_inverse)-1

        layers_len = len(self.layers)
        for i in range(layers_len):
            x = self.layers[layers_len-i-1].inverse(x)
            if (layers_len-i-1) % K+1 == 0:
                x = self.layers[layers_len-i-1].inverse(embedding[embedding_ptr])
                embedding_ptr -= 1

        return x

    #def _forward_log_det_jacobian(self, x):


    #def _inverse_log_det_jacobian(self, x):
        
    
    def _flow_steps(self, K, input_shape, name):
        flow_steps_list = []
        for i in range(K):
            act_norm = activation_normalization.ACT(input_shape[-1], False, name=name+"/actnorm_{}".format(i))
            inv_conv = conv_1x1_inv.invertible_1x1_conv_LU(input_shape[-1], name=name+"/inv_conv_1x1_{}".format(i))
            acl = affine_coupling_layer.ACL(input_shape[-1], name=name+"/acl_{}".format(i))
            flow_steps_list.append(act_norm)
            flow_steps_list.append(inv_conv)
            flow_steps_list.append(acl)
        
        return flow_steps_list
    
    def _get_input_shapes(self, input_shape, level):
        input_shapes = []
        for i in range(level):
            input_shape = [input_shape[0]//2,
                            input_shape[1]//2,
                            input_shape[2] * 4 - i * 8]
            input_shapes.append(input_shape)

        return input_shapes
    # model paramters
    

    