import tensorflow as tf
import numpy as np

def model():
    # model paramters
    L=3
    K=32
    epochs = 30
    learning_rate = 1e-4
    BATCH_SIZE=120
    activation = "relu"
    input_size = [32,32,3]

    num_of_hidden = [512,512]

    