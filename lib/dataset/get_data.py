import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
dataset = 'mnist'


def make_iterator(flow):
    def itertor():
        return flow.next()

    return itertor


def get_data(dataset, batch_size):
    if dataset == "mnist":
        from keras.datasets import mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        y_train = y_train.reshape([-1])
        y_test = y_test.reshape([-1])

        x_train = np.lib.pad(x_train, ((0, 0), (2, 2), (2, 2)), 'minimum')
        x_test = np.lib.pad(x_test, ((0, 0), (2, 2), (2, 2)), 'minimum')

        x_train = np.tile(np.reshape(x_train, (-1, 32, 32, 1)), (1, 1, 1, 3))
        x_test = np.tile(np.reshape(x_test, (-1, 32, 32, 1)), (1, 1, 1, 3))

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        x_train = (x_train /(255.0/2)) - 1
        x_test = (x_test /(255.0/2)) - 1

    datagen_test = ImageDataGenerator()
    datagen_train = ImageDataGenerator(width_shift_range=0.1,
                                       height_shift_range=0.1)

    datagen_train.fit(x_train)
    datagen_test.fit(x_test)

    train_flow = datagen_train.flow(x_train, y_train, batch_size)
    test_flow = datagen_test.flow(x_test, y_test, batch_size, shuffle=False)

    train_iterator = make_iterator(train_flow)
    test_iterator = make_iterator(test_flow)

    return train_iterator, test_iterator

def get_data_alt(batch_shuffle_size, batch_size):
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    @tf.function
    def _parse_function(img, label):
        feature = {}
        img = tf.pad(img, paddings=[[2, 2], [2, 2]], mode="CONSTANT")
        img = tf.expand_dims(img, axis=-1)
        img = tf.reshape(img, [32, 32, 1])
        # print(img.shape)
        img = tf.tile(img, ( 1, 1, 3))

        img = tf.cast(img, dtype=tf.float32)
        img = (img / (255.0 / 2)) - 1
        feature["img"] = img
        feature["label"] = label
        return feature


    train_dataset_raw = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(
            _parse_function
            )
    test_dataset_raw = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(
            _parse_function
            )
    train_dataset = train_dataset_raw.shuffle(batch_shuffle_size).batch(batch_size)
    test_dataset = test_dataset_raw.shuffle(batch_shuffle_size).batch(batch_size)

    return train_dataset, test_dataset

#train_iterator, test_iterator = get_data(dataset, 32)
# print(train_iterator())
