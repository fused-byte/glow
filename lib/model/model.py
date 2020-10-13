import tensorflow as tf
import numpy as np

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
        z = tf.keras.layers.conv2d(z, _w, strides = [1, 1, 1, 1], padding = 'same')
        logdet += dlogdet
        return z, logdet
    else:
        # Reverse computation
        _w = tf.matrix_inverse(w)
        _w = tf.reshape(_w, [1,1,c,c])
        z = tf.keras.layers.conv2d(z, _w, strides = [1,1,1,1], padding = 'same')
        logdet -= dlogdet
        return z, logdet