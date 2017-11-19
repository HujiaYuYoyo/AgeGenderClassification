"""
Simple tester for the vgg16_trainable
"""

import tensorflow as tf

import vgg16_trainable as vgg16
import utils
import numpy as np


img1 = utils.load_image("./test_data/m1.jpg")
img2 = utils.load_image("./test_data/m2.jpg")
img3 = utils.load_image("./test_data/f.jpeg")

img1_true_result = [1 if i == 0 else 0 for i in range(2)]  # 1-hot result for gender
img2_true_result = [1 if i == 0 else 0 for i in range(2)]  # 1-hot result for gender
img3_true_result = [1 if i == 1 else 0 for i in range(2)]  # 1-hot result for gender
img_true_result = [img1_true_result, img2_true_result, img3_true_result]

batch1 = img1.reshape((1, 224, 224, 3))
batch2 = img2.reshape((1, 224, 224, 3))
batch3 = img3.reshape((1, 224, 224, 3))

batch = np.concatenate((batch1, batch2, batch3), 0)

with tf.device('/cpu:0'):
    sess = tf.Session()

    images = tf.placeholder(tf.float32, [3, 224, 224, 3])
    true_out = tf.placeholder(tf.float32, [3, 2])
    train_mode = tf.placeholder(tf.bool)

    vgg = vgg16.Vgg16('./vgg16.npy')
    vgg.build(images, train_mode)

    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    print(vgg.get_var_count())

    sess.run(tf.global_variables_initializer())

    # test classification
    prob = sess.run(vgg.prob, feed_dict={images: batch, train_mode: False})
    utils.print_prob(prob[0], './gender.txt')
    utils.print_prob(prob[1], './gender.txt')
    utils.print_prob(prob[2], './gender.txt')

    # simple 1-step training
    for i in range(10):
        print 'iter: ', i
        cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
        train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
        sess.run(train, feed_dict={images: batch, true_out: img_true_result, train_mode: True})

        # test classification again, should have a higher probability about tiger
        prob = sess.run(vgg.prob, feed_dict={images: batch, train_mode: False})
        utils.print_prob(prob[0], './gender.txt')
        utils.print_prob(prob[1], './gender.txt')
        utils.print_prob(prob[2], './gender.txt')

    # test save
    vgg.save_npy(sess, './test-save.npy')
