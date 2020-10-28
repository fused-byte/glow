from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import os
from pathlib import Path
from model import Glow_Model, Flow
import matplotlib.pyplot as plt
import sys

sys.path.insert(1, os.getcwd() + '/lib/dataset')
import get_data
tfd = tfp.distributions

print(os.getcwd())

def main():
    L=3
    K=32
    epochs = 30
    learning_rate = 1e-4
    BATCH_SIZE=4
    input_shape = [32,32,3]

    bpd_factor = np.log(2) * input_shape[0] * input_shape[1] * input_shape[2]

    # glow_model = Glow_Model(name="glow_model")
    # glow_model.build((1,32,32,3))
    # print(glow_model.summary())
    flow = Flow(L, K, "relu", [32,32,3], [512,512], name="test")
    flow_distribution = tfd.TransformedDistribution(
                                event_shape=input_shape,
                                distribution=tfd.Normal(loc=0.0, scale=1.0),
                                bijector=flow                       
                            )
    # x = tf.random.normal([3, 32, 32, 3])
    # y = flow_distribution.bijector.forward(x)
    # z = flow_distribution.bijector.inverse(y)
    fig = plt.figure(figsize=(8,8))
    fig.add_subplot(1,2,1)
    # plt.imshow(x[0])
    # # plt.show()
    # fig.add_subplot(1,2,2)
    # plt.imshow(z[0])
    # # plt.show()
    # print(tf.reduce_sum(z-x))
    # print(flow_distribution.log_prob(x).shape)

    # @tf.function
    def loss():
        x_ = np.clip(np.floor(x), 0, 255) /255.0
        x_.astype(np.float32)
        # print("input x: ", np.around(x_[0], 3))
        #plt.imshow(x[0])
        #plt.show()
        # print(np.max(x_[0,:,:,:]))
        print("bpd factor: ", bpd_factor, x_.shape)
        log_det = tf.reduce_mean(flow_distribution.log_prob(x_))
        print("log det: ", log_det)
        # if log_det.numpy() == np.nan:
        #     return log_det
        # else:
        #     return log_det / bpd_factor
        return tf.constant(1234)

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate) 
    log = tf.summary.create_file_writer('checkpoints')
    avg_loss = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
    dataset='mnist'
    train_iterator, test_iterator = get_data.get_data(dataset, BATCH_SIZE)
    train_its = int(60000/BATCH_SIZE)

    #checkpointing 
    checkpoint_path=Path('./checkpoints/flow_train')
    ckpt = tf.train.Checkpoint(model=flow, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt,
                                                checkpoint_path,
                                                max_to_keep=3)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('[Flow] Latest checkpoint restored!!')

    flag = False

    for e in range(1):
        if flag:
            break
        for i in range(train_its):
            x, y = train_iterator()
            # print(x.shape)
            with tf.GradientTape() as tape:
                log_prob_loss = loss()
                print('log prob loss : ', log_prob_loss)
            
            print('This is flow train vars : ', len(flow.trainable_variables))
            grads = tape.gradient(log_prob_loss, flow.trainable_variables)
            print('This is grad value :', grads)
            break
            optimizer.apply_gradients(zip(grads, flow.trainable_variables))
            print("returned log prob loss: ", log_prob_loss)
            if tf.math.is_nan(log_prob_loss):
                flag = True
                break
            avg_loss.update_state(log_prob_loss)
            
            if tf.equal(optimizer.iterations % 1000, 0):
                print("Step {} Loss {:.6f}".format(optimizer.iterations, avg_loss.result()))
            if tf.equal(optimizer.iterations % 100, 0):
                with log.as_default():
                    tf.summary.scalar("loss", avg_loss.result(), step=optimizer.iterations)
                    avg_loss.reset_states()
        
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(
                e + 1, ckpt_save_path))
    # print(train_iterator())

if __name__=="__main__":
    main()