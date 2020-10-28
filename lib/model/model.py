import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import keras.backend as K
import squeeze, conv_1x1_inv, activation_normalization, affine_coupling_layer, bijector_split


tfb = tfp.bijectors
tfd = tfp.distributions

class Flow(tfb.Bijector):
    def __init__(self, L,
        K,
        activation,
        input_shape,
        num_of_hidden,
        name="flow_model"
    ):

        super(Flow, self).__init__(
            validate_args=False,
            forward_min_event_ndims=0,
            name=name)
        
        self.L = L
        self.K = K
        self.layers = []
        self.embedding_forward = []
        self.embedding_inverse = []
        self.fldjs = []
        self.ildjs = []
        self.input_shapes = self._get_input_shapes(input_shape, L)
        self.unsqueeze_layers = []

        print(self.input_shapes)

        for i in range(L):

            squeeze_layer = squeeze.Squeeze(2, name=name+"/squeeze_{}".format(i))
            self.layers.append(squeeze_layer)
            flow_steps = self._flow_steps(K, input_shape=self.input_shapes[i], name=name+"/flow_step_{}".format(i))
            self.layers = self.layers + flow_steps
            if i < L-1:
                split = bijector_split.bijector_split(0, name=name+"/split_{}".format(i))
                self.layers.append(split)
        for i in range(L):
            squeeze_invert = tfb.Invert(squeeze.Squeeze(2, name=name+"/unsqueeze_{}".format(L-i)))
            self.unsqueeze_layers.append(squeeze_invert)
        print("total layers: ", len(self.layers) + len(self.unsqueeze_layers))


    def _forward(self, x):
        embedding = []
        fldjs = []
        for i in range(len(self.layers)):
            # print(i, self.layers[i].name, x.shape)
            
            if "split" in self.layers[i].name:
            # if i % self.K+1 == 0:
                # print("inside if condition")
                z1, z2 = self.layers[i].forward(x)
                fldjs.append(self.layers[i].forward_log_det_jacobian(x, event_ndims=3))
                embedding.append(z1)
                x = z2
                print("embedding after split: ", z1.shape,z2.shape)
            else:
                # print("inside else ")
                x = self.layers[i].forward(x)
                fldjs.append(self.layers[i].forward_log_det_jacobian(x, event_ndims=3).numpy().tolist())

        self.embedding_forward = embedding
        
        for i in range(len(self.unsqueeze_layers)):
            x = self.unsqueeze_layers[i].forward(x)
            fldjs.append(self.unsqueeze_layers[i].forward_log_det_jacobian(x, event_ndims=3).numpy().tolist())
            if i < self.L-1:
                x = tf.concat([embedding[self.L-2-i], x], axis=-1)
        
        self.fldjs = fldjs
        return x

    def _inverse(self, x):
        # print("inverse is called: ", x.shape)
        embedding = []
        ildjs = []
        for i in range(len(self.unsqueeze_layers)):
            x = self.unsqueeze_layers[i].inverse(x)
            # print("shit", self.unsqueeze_layers[i].inverse_log_det_jacobian(x, event_ndims=3))
            ildjs.append(self.unsqueeze_layers[i].inverse_log_det_jacobian(x, event_ndims=3).numpy())
            if i < self.L-1:
                z1, z2 = tf.split(x, 2, axis = -1)
                embedding.append(z1)
                x = z2
        
        self.embedding_inverse = embedding
        embedding_ptr = len(self.embedding_inverse)-1

        layers_len = len(self.layers)
        for i in range(layers_len):
            # print(self.layers[layers_len-i-1].name)
            # if (layers_len-i-1) % self.K+1 == 0:
            if "split" in self.layers[layers_len-i-1].name:
                # print(x.shape, embedding[embedding_ptr].shape)
                x = self.layers[layers_len-i-1].inverse([embedding[embedding_ptr], x])
                ildjs.append(self.layers[layers_len-i-1].inverse_log_det_jacobian(embedding[embedding_ptr], event_ndims=3).numpy())
                embedding_ptr -= 1
                # print(x.shape)
            else:
                x = self.layers[layers_len-i-1].inverse(x)
                # print(tf.executing_eagerly())
                # print(tf.get_static_value(self.layers[layers_len-i-1].inverse_log_det_jacobian(x, event_ndims=3)))
                # print("shit 2: ", self.layers[layers_len-i-1].inverse_log_det_jacobian(x, event_ndims=3).numpy())
                if "inv_conv_1x1_" in self.layers[layers_len-i-1].name:
                    ildjs.append(self.layers[layers_len-i-1].inverse_log_det_jacobian(x, event_ndims=3).numpy()/(32*32))
                else:
                    ildjs.append(self.layers[layers_len-i-1].inverse_log_det_jacobian(x, event_ndims=3).numpy())
        
        self.ildjs = ildjs
        
        return x

    def _forward_log_det_jacobian(self, x):
        print("*************************called**********")
        return sum(self.fldjs)

    def _inverse_log_det_jacobian(self, x):
        sum_ildjs = 0
        for i in range(len(self.ildjs)):

            print("inverse ildjs: {}".format(i+1),self.ildjs[i], self.ildjs[i].shape)
            if self.ildjs[i].shape == (4,):
                if self.ildjs[i][0] == np.nan:
                    return sum_ildjs
                sum_ildjs += sum(self.ildjs[i])
            else:
                if self.ildjs[i] == np.nan:
                    return sum_ildjs
                sum_ildjs += self.ildjs[i]
        #         print(tf.get_static_value(self.ildjs[i], partial=True))
                # print(K.eval(self.ildjs[i]))
            
        print("inverse ildjs: ", sum_ildjs)
        return sum_ildjs
        
    
    def _flow_steps(self, K, input_shape, name):
        flow_steps_list = []
        for i in range(K):
            act_norm = activation_normalization.ACT(input_shape[-1], False, name=name+"/actnorm_{}".format(i))
            inv_conv = conv_1x1_inv.invertible_1x1_conv_LU(input_shape[-1], name=name+"/inv_conv_1x1_{}".format(i))
            acl = affine_coupling_layer.ACL(input_shape, name=name+"/acl_{}".format(i))
            flow_steps_list.append(act_norm)
            flow_steps_list.append(inv_conv)
            flow_steps_list.append(acl)
        
        return flow_steps_list
    
    def _get_input_shapes(self, input_shape, level):
        # print(level)
        input_shapes = []
        for i in range(level):
            input_shape = [input_shape[0]//2,
                            input_shape[1]//2,
                            (input_shape[2] * 4) - (i * 24)]
            input_shapes.append(input_shape)

        return input_shapes
    # model paramters
    
class Glow_Model(tf.keras.Model):
    def __init__(self, name):
        L=3
        K=32
        activation = "relu"
        input_shape = [32,32,3]
        num_of_hidden = [512,512]
        name="glow_model"
        print("inside glow model init")
        super(Glow_Model, self).__init__( name)

        self.flow_model = Flow(L, K, activation, input_shape, num_of_hidden, name=name+"/flow_bijector")

        self.flow_distribution_obj = tfd.TransformedDistribution(
                                event_shape=input_shape,
                                distribution=tfd.Normal(loc=0.0, scale=1.0),
                                bijector=self.flow_model                        
                            )
        
    def call(self, inputs):
        return self.flow_distribution_obj.bijector.forward(inputs)

    def log_prob(self, inputs):
        return self.flow_distribution_obj.log_prob(inputs)
    
    def getFlowSample(self, num):
        return self.flow_distribution_obj.sample(num)

    