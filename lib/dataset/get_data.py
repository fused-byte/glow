import numpy as np
from keras.preprocessing.image import ImageDataGenerator
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


train_iterator, test_iterator = get_data(dataset, 32)
print(train_iterator())
