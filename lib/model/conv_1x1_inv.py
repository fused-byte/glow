import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions
# Invertible 1x1 conv - Code from paper
def invertible_1x1_conv(z, logdet, forward=True): 
    # Shape
    h,w,c = z.shape[1:]
    # Sample a random orthogonal matrix to initialise weights
    w_init = np.linalg.qr(np.random.randn(c,c))[0]
    w = tf.Variable("W", initializer=w_init)
    # Compute log determinant
    dlogdet = h * w * tf.log(abs(tf.matrix_determinant(w)))
    if forward:
        # Forward computation
        _w = tf.reshape(w, [1,1,c,c])
        z = tf.keras.layers.conv2d(z, _w, strides = [1,1,1,1], padding = 'same')
        logdet += dlogdet
        return z, logdet
    else:
        # Reverse computation
        _w = tf.matrix_inverse(w)
        _w = tf.reshape(_w, [1,1,c,c])
        z = tf.keras.layers.conv2d(z, _w, strides = [1,1,1,1], padding = 'same')
        logdet -= dlogdet
        return z, logdet

def invertible_1x1_conv_LU(event_size, batch_shape=(), seed=None, dtype=tf.float32, name="lu_inv_conv"):

    with tf.name_scope(name or 'trainable_lu_factorisation'):
        event_size = tf.convert_to_tensor(
                        event_size, dtype_hint=tf.int32, name='event_size')
        batch_shape = tf.convert_to_tensor(
                        batch_shape, dtype_hint=event_size.dtype, name='batch_shape')
        random_matrix = tf.random.uniform(
                        shape=tf.concat([batch_shape, [event_size, event_size]], axis=0),
                        dtype=dtype,
                        seed=seed,
                        name='1x1_inv_conv_weights')
        #QR decomposition gives us 2 matrix
        # 0 index - orthogonal matrix which has orthornormal unit vector columns
        # 1 index Right upper triangular matrix.
        random_orthornormal = tf.linalg.qr(random_matrix)[0]

        # we do LU decomposition of orthogonal matrix 
        # 0th index gives lower_upper triangular matrix
        # 1 index give permutation matrix.
        lower_upper, permutation = tf.linalg.lu(random_orthornormal)

        lower_upper = tf.Variable(
                        initial_value=lower_upper,
                        trainable = True,
                        name='lower_upper')
        permutation = tf.Variable(
                        initial_value=permutation,
                        trainable=False,
                        name='permutation')

    inv_conv = tfb.MatvecLU(lower_upper, permutation, name=name)
    return inv_conv

def build_model(channels=3):
    # conv1x1 setup
    # t_lower_upper, t_permutation = invertible_1x1_conv_LU(channels)
    conv1x1 = invertible_1x1_conv_LU(channels, name='MatvecLU')
    # tfb.MatvecLU(t_lower_upper, t_permutation, name='MatvecLU')
    print('conv1x1 variable\n', conv1x1.variables)
    inv_conv1x1 = tfb.Invert(conv1x1)

    # forward setup
    fwd = tfp.layers.DistributionLambda(
        lambda x: conv1x1(tfd.Deterministic(x)))
    fwd.vars = conv1x1.trainable_variables

    # inverse setup
    inv = tfp.layers.DistributionLambda(
        lambda x: inv_conv1x1(tfd.Deterministic(x)))
    inv.vars = inv_conv1x1.trainable_variables
    
    x: tf.Tensor = tf.keras.Input(shape=[28, 28, channels])
    fwd_x: tfp.distributions.TransformedDistribution = fwd(x)
    rev_fwd_x: tfp.distributions.TransformedDistribution = inv(fwd_x)
    example_model = tf.keras.Model(inputs=x, outputs=rev_fwd_x, name='conv1x1')
    return example_model


def test_conv1x1():
    example_model = build_model()
    example_model.trainable = True
    example_model.summary()

    real_x = tf.random.uniform(shape=[2, 28, 28, 3], dtype=tf.float32)
    if example_model.weights == []:
        print('No Trainable Variable exists')
    else:
        print('Some Trainable Variables exist')

    with tf.GradientTape() as tape:
        tape.watch(real_x)
        out_x = example_model(real_x)
        out_x = out_x
        loss = out_x - real_x
    print(tf.math.reduce_sum(real_x - out_x))
    print(example_model.predict(real_x).shape)
    # => nealy 0
    # ex. tf.Tensor(1.3522818e-05, shape=(), dtype=float32)

    try:
        print(tape.gradient(loss, real_x).shape)
    except Exception as e:
        print('Cannot Calculate Gradient')
        print(e)
        
# test_conv1x1()