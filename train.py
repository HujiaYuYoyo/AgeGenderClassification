from facedataclass import facesdata
import tensorflow as tf

"""
Simple tester for the vgg16_trainable on facesdata
if have any quetsions - contact hujiay@stanford.edu
"""

import vgg16_trainable as vgg16
import utils
import numpy as np


# 'fold_1_data.txt', 'fold_2_data.txt', 'fold_3_data.txt', 'fold_4_data.txt'
filename = ['fold_0_data.txt']
batchsze = 50

faces = facesdata()
faces.getlabeldata(filename)
faces.labeling()

for userid in faces.userid:
	faces.getuserimage(userid)

faces.setbatchsize(batchsze)

genderlabel, agelabel, imagebatch = faces.loaddatabatch()

with tf.device('/cpu:0'):
    sess = tf.Session()

    images = tf.placeholder(tf.float32, [batchsze, 224, 224, 3])
    true_out = tf.placeholder(tf.float32, [batchsze, 2])
    train_mode = tf.placeholder(tf.bool)

    vgg = vgg16.Vgg16('./vgg16.npy')
    vgg.build(images, train_mode)

    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    print(vgg.get_var_count())

    sess.run(tf.global_variables_initializer())

    # test classification before starting to train
    prob = sess.run(vgg.prob, feed_dict={images: imagebatch , train_mode: False})
    for i in range(batchsze):
    	utils.print_prob(prob[i], './gender.txt') # print the top likely output label
    	print 'true-out: ', ['m' if genderlabel[i] == [1, 0] else 'f']

    # simple 1-step training
    for i in range(10):
        print 'iter: ', i
        cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
        train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
        sess.run(train, feed_dict={images: imagebatch, true_out: genderlabel, train_mode: True})

        # test classification again, should have a higher probability about each instance
        prob = sess.run(vgg.prob, feed_dict={images: imagebatch, train_mode: False})
        for i in range(batchsze):
    		utils.print_prob(prob[i], './gender.txt') # print the top likely output label

    # test save
    # name = 11180353 monthdayhourminutes
    timestr = time.strftime("%m%d%I%M")
	result = './result-save' + timestr + '.npy'
    vgg.save_npy(sess, result)
