#!/usr/bin/env python
import random
import numpy as np
from OLB import OLB
import OLButils as Utils
from OLBlearner import OLBlearner
from traindataclass import traindata
import os.path
import json

filename = ['fold_0_data.txt', 'fold_1_data.txt', 'fold_2_data.txt', 'fold_3_data.txt']
alpha = 0.8
gamma = 1

def createleaner(filename, alpha, gamma):
	data = traindata()
	data.gettraindata(filename)
	data.label('train')
	for userid in data.trainuserid:
		data.gettrainuserimage(userid)
	data.convertdata()
	dataconverted = data.traindataconverted
	numPCs = len(dataconverted)
	learner = OLBlearner(alpha, gamma, numPCs)
	learner._converttoOLBs(dataconverted)
	print 'successfully created a OLB learner class with converted OLBs!'
	return learner


def createlearnerfromsampledata(filename, alpha = 0.8, 
								gamma = 1, imgpergender = 10, 
								trainoutput = 'train.json', 
								testoutput = 'test.json', 
								numPCs = 10 * 2 * 7 * 0.6):
	if os.path.exists(trainoutput):
		print 'importing train and test dataset from existing files ...'
		train = json.load(open(trainoutput))
		test = json.load(open(testoutput))
		train = train['train']
		test = test['test']
	else:
		print 'creating new sampled train and test dataset ...'
		data = traindata()
		data.gettraindata(filename)
		data.label('train')
		for userid in data.trainuserid:
			data.gettrainuserimage(userid)
		data.convertdata()
		data.sampledata(imgpergender, trainoutput, testoutput)
		train = data.train['train']
		test = data.test['test']
	numPCs = len(train)
	try:
		assert numPCs == imgpergender * 14 * 0.6
		assert len(test) == imgpergender * 14 * 0.4
	except AssertionError:
		print 'length of train is: ', len(train)
		print 'length of test is: ', len(test)
	learner = OLBlearner(alpha, gamma, numPCs)
	learner._converttoOLBs(numPCs, train)
	print 'successfully created a OLB learner class with converted OLBs!'
	return learner, train, test

