#!/usr/bin/env python
import random
import numpy as np
from OLB import OLB
import OLButils as Utils
from OLBlearner import OLBlearner
from traindataclass import traindata
from learner import *
import matplotlib.pyplot as plt
import copy
import utils
import os


def _train_learning(k, learner, param, logfile, correctscore, punishscorem, train):
	print 'training started !'
	# OLB = random.choice(learner.OLBs)
	olbcounter = 0
	with open(logfile,'a') as f:
		f.write('olb\tepoch\titeration\treward\taccuracy\n')
	for OLB in learner.OLBs:
		olbcounter += 1
		best_l = learner._instantiate_Qtables(OLB)
		best = []
		besthandr = [0,0,0] # best h, epoch, and reward for that OLB 
		for e in range(learner.epochs):
			OLB.h = [0] * learner.N # initialize h
			OLB.chosenfbyorder = [] # initialize chosenf
			repeat = True
			# best_l = [random.choice(list(range(learner.N)))]
			curriter = 0
			while repeat:
				curriter += 1
				nextfeature = learner._chooseaction(OLB, best_l)
				learner._updatefeature(OLB, nextfeature)
				projected = learner._projectall(OLB)
				klabels = Utils._kmeans(projected, k)
				r, accu = learner._getreward(klabels, param, correctscore, punishscore)
				if r > besthandr[2]: 
					besthandr[0] = sum(OLB.h)
					besthandr[1] = e
					besthandr[2] = r
				with open(logfile,'a') as f:
					f.write('{}\t{}\t{}\t{}\t{}\n'.format(olbcounter, e, curriter, r, accu))
				if curriter % 50 == 0: 
					print 'For {} OLB, best h {}, at best epoch {} and best reward: {} '\
							.format(olbcounter, besthandr[0], besthandr[1], besthandr[2])
				best_l = learner._update_Qtable(OLB, r)
				if sum(OLB.h) == len(OLB.h) or r == correctscore * len(train):
					repeat = False
			best.append(copy.deepcopy(besthandr))
		if olbcounter == 1: # only plot once
			besth = [x[0] for x in best]
			bestr = [x[2] for x in best]
			# with open('besthr.txt','a') as f:
			# 		f.write('{}\n{}\n'.format(besth, bestr))
			fig = plt.figure()
			plt.plot(np.arange(learner.epochs), besth)
			plt.title('Best Feature Basis for One Image During Learning')
			plt.ylabel('Number of Feature Basis')
			plt.xlabel('Epoch')
			name = 'besth' + str(imgpergender) + str(correctscore) + str(punishscore) + '.png'
			fig.savefig(name)

			fig = plt.figure()
			plt.plot(np.arange(learner.epochs), bestr)
			plt.title('Best Reward for One Image During Learning')
			plt.ylabel('Reward')
			plt.xlabel('Epoch')
			name = 'bestr' + str(imgpergender) + str(correctscore) + str(punishscore) + '.png'
			fig.savefig(name)

		with open('train1211.txt','a') as f:
			f.write('For {} OLB, best h {}, at best epoch {} and best reward: {} '.format(olbcounter, besthandr[0], besthandr[1], besthandr[2]))
		learner._sortfeatures(OLB)
		print 'Finished sorting q table for current OLB !'
	print 'Successfully trained and updated and sorted all of OLB Q_tables! ready for testing ! '

def _test(learner, test, k, param, imgpergender, outputfile = './finalresult/testOLB.txt'):
	correct1 = 0
	correct2 = 0
	with open(outputfile,'a') as f:
		f.write('Accuracy for parameter {}, testing on {} test dataset:\n'.format(param, len(test)))
	for x in test:
		img = utils.load_image(x['imagepath'])
		if param == 'gender':
			top1 = learner._classify(img, k, param)
		else:
			top1, top2 = learner._classify(img, k, param)
		if top1 == x[param]:
			correct1 += 1
		if param == 'age':
			if x[param] in top2:
				correct2 += 1
	accu1 = correct1 * 1./len(test)
	accu2 = correct2 * 1./len(test)
	print 'param: {}, imgpergender: {}, numPCs: {}, correctscore: {}, punishscore: {}, top1: {}, top2: {} \n'\
			.format(param, imgpergender, learner.numPCs, correctscore, punishscore, accu1, accu2)
	with open('./finalresult/testOLB.txt','a') as f:
		f.write('param: {}, imgpergender: {}, numPCs: {}, correctscore: {}, punishscore: {}, top1: {}, top2: {} \n'\
			.format(param, imgpergender, learner.numPCs, correctscore, punishscore, accu1, accu2))


# filename = ['test245.txt']
# filename = ['fold_0_data.txt', 'fold_1_data.txt', 'fold_2_data.txt', 'fold_3_data.txt']
# filename = ['fold_0_data.txt']
filename = ['test.txt']
alpha = 0.8
gamma = 1
k = 2 # class label
param = 'gender'
imgpergender = 10
trainoutput = 'train.json'
testoutput = 'test.json' 
displayk = 3
logfile = './finalresult/log1214gender.txt'
# qtablefile = './finalresult/qtables1214gender.txt'
correctscore = 2
punishscore = 1
eigenfaceoutput = './finalresult/eigenfacepredict.txt'

if __name__ == '__main__':
	if not os.path.exists('./finalresult'):
		os.mkdir('./finalresult')
	for (k, param) in ((2, 'gender'), (7, 'age')):
		for imgpergender in [1, 3, 10, 15, 20]:
			for correctscore in [5, 10]:
				for punishscore in [1, 5, 10]:
					trainoutput = 'train' + str(imgpergender) + '.json' 
					testoutput = 'test' + str(imgpergender) + '.json' 
					logfile = './finalresult/log' + param + str(k) + str(imgpergender) + \
								str(correctscore) + str(punishscore) + '.txt'
					outputfile = './finalresult/' + param +'testOLB.txt'
					print 'Currently training results based on param: {}, k: {}, imgpergender: {}, reward: {}, punish: {}'\
						.format(param, k, imgpergender, correctscore, punishscore)
					learner, train, test = createlearnerfromsampledata(filename, alpha = alpha, 
												gamma = gamma, imgpergender = imgpergender, 
												trainoutput = trainoutput, 
												testoutput = testoutput, 
												numPCs = imgpergender * 2 * 7 * 0.6)
					print 'testing on a randomly chosen train data ...'
					Utils.test(learner, train, test, displayk, saveimg = 'after.png')
					print 'testing on eigenfaces without q learning on the test dataset ...'
					Utils.testeigenface(learner, train, test, param, outputfile = eigenfaceoutput)
					print 'now training q value tables for each face ..'
					_train_learning(k, learner, param, logfile, correctscore, punishscore, train)
					_test(learner, test, k, param, imgpergender, outputfile)

