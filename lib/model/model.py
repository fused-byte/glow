import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import keras.backend as K
import squeeze, conv_1x1_inv, activation_normalization, affine_coupling_layer, bijector_split, blockwise


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
        self.input_shapes = self._get_input_shapes(input_shape, L)

        #print(self.input_shapes)
        self.flow_network = tfb.Chain(list(reversed(self.add_flow(self.input_shapes, name=name+"/Flow_0"))))

    def add_flow(self, input_shape, name):
        flow_layers = []
        squeeze_layer = squeeze.Squeeze(2, name=name+"/squeeze_{}".format(self.L - len(input_shape)))
        flow_layers.append(squeeze_layer)
        flow_steps = self._flow_steps(self.K, input_shape=input_shape[0], name=name+"/flow_step_{}".format(self.L - len(input_shape)))
        flow_layers.append(flow_steps)
        if len(input_shape) != 1:
            #Blockwise
            blockwise_mod = blockwise.Blockwise(bijectors=[tfb.Identity(), tfb.Chain(
                            list(reversed(self.add_flow(input_shape[1:], name = name.split('_')[0]+"_{}".format(self.L-len(input_shape[1:])))))
                        )], name=name+"/blockwise_{}".format(self.L - len(input_shape)))
            flow_layers.append(blockwise_mod)
        squeeze_invert = tfb.Invert(squeeze.Squeeze(2, name=name+"/unsqueeze_{}".format(self.L - len(input_shape))))
        flow_layers.append(squeeze_invert)
        return flow_layers
    
    def _flow_steps(self, K, input_shape, name):
        flow_steps_list = []
        for i in range(K):
            act_norm = activation_normalization.ACT(input_shape[-1], False, name=name+"/actnorm_{}".format(i))
            inv_conv = conv_1x1_inv.invertible_1x1_conv_LU(input_shape[-1], name=name+"/inv_conv_1x1_{}".format(i))
            acl = affine_coupling_layer.ACL(input_shape, name=name+"/acl_{}".format(i))
            
            flow_steps_list.append(act_norm)
            flow_steps_list.append(inv_conv)
            flow_steps_list.append(acl)

        flow_steps = tfb.Chain(list(reversed(flow_steps_list)), validate_args=False, name=name)
        return flow_steps
    
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

    