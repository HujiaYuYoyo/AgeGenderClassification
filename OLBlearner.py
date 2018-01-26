#!/usr/bin/env python
import random
import numpy as np
from OLB import OLB
import OLButils as Utils
import operator
import collections
# from tqdm import tqdm


class OLBlearner(object):
	def __init__(self, alpha, gamma, numPCs, weights = None):
		self.OLBs = [] # list of tuple of (x_i, w_i, T_i) OLB objects
		self.numPCs = numPCs # N is the number of eigenfaces = numPCs
		self.N = numPCs
		self.m = len(self.OLBs) # size of OLB set - all training dataset
		self.alpha = alpha # learning rate 1 < alpha <= 1
		self.gamma = gamma # discount fatcor 0 <= gamma <= 1
		self.epsilon = 0.1 # greedy policy epsilon
		self.epochs = numPCs
		self.weights = weights
		self.evectors = []
		self.mean_img_col = []


	# this function converts training data list of dictionaries to OLBs
	# by taking one data - get its gender and age labels 
	# run pca on its image and initialize to be OLB and append to OLBs
	def _converttoOLBs(self, numPCs, traindataconverted):
		m = len(traindataconverted)
		gender, age, images = Utils.loadtraindatabatch(m, traindataconverted)
		#  weights (m, 146) evectors (m, 224*224)
		weights, mean_img_col, evectors= Utils._PCA(numPCs, m, images)
		self.mean_img_col = mean_img_col
		self.weights = weights
		self.evectors = evectors
		for i in range(len(gender)):
			w_gender_i = gender[i]
			w_age_i = age[i]
			x_i = weights[:,i]
			assert x_i.shape == (numPCs, 1)
			img_row = x_i.T.dot(evectors)
			assert img_row.shape == (1, 224*224) # a row vector of the original image after numPC transformation
			T_i = [0] * numPCs
			olb = OLB(x_i, w_age_i, w_gender_i, T_i, mean_img_col, pcs = evectors, img_col = img_row.T)
			# (self, x_i, w_age_i, w_gender_i, T_i, mu, pcs, img_col)
			self.OLBs.append(olb)
		print '%s number of OLBs have been added!' % len(self.OLBs)
		return weights, mean_img_col, evectors


	def _instantiate_Qtables(self, OLB):
		temp = []
		qtable0 = np.random.randn(self.N, 1)
		temp.append(qtable0)
		qtable1 = [qtable0] * self.N
		temp.append(qtable1)
		for _ in range(1, self.N):
			qtable1 = [qtable1] * self.N
			temp.append(qtable1)
		OLB.Q_tables = temp
		return OLB.Q_tables[0]

	# project all the representative points x_j, j = 1, .., m in training data into space defined 
	# by h using p_j = diag(x_j, h)
	# weights (topk, 146) evectors (topk, 224*224)
	def _projectall(self, OLB):
		h = np.diag(OLB.h)
		assert h.shape == (self.numPCs, self.numPCs) # shape is (topk, 1)
		projected = np.dot(self.weights.T, h) 
		assert projected.shape == (len(self.OLBs), self.numPCs) # every row represents a olb image
		return projected


	# this function returns reward after kmeans by summing up the correctly classified labels and 
	# minus the incorrectly classified ones
	def _getreward(self, labels_, param, correctscore = 1, punishscore = 1):
		reward = 0
		count = 0
		if param == 'age':
			file_path = 'age.txt'
			gt = [l.strip() for l in open(file_path).readlines()]
			for i in range(len(self.OLBs)):
				pred = gt[labels_[i]]
				if gt[np.argmax(self.OLBs[i].w_age_i)] == pred:
					reward += correctscore
					count += 1
				else: 
					reward -= punishscore
		else:
			file_path = 'gender.txt'
			gt = [l.strip() for l in open(file_path).readlines()]
			for i in range(len(self.OLBs)):
				pred = gt[labels_[i]]
				if gt[np.argmax(self.OLBs[i].w_gender_i)] == pred:
					reward += 1
					count += 1
				else: 
					reward -= 1	
		accu = count*1./len(self.OLBs)
		return reward, accu

	# this function updates the cell in the Q_table by looking ahead one step and 
	# calculates the step with max Q-value and updates the current cell
	# need to keep track of recently chosen feature as f
	def _update_Qtable(self, OLB, r): 
		j = sum(OLB.h) - 1
		if j == 0:
			qtable1 = OLB.Q_tables[1]
			i_0 = OLB.chosenfbyorder[0]
			tempmax = float('-inf')
			for l in range(1, self.N):
				if qtable1[i_0][l] > tempmax:
					tempmax = qtable1[i_0][l]
			OLB.Q_tables[0][i_0] = OLB.Q_tables[0][i_0] + self.alpha * (r + self.gamma * tempmax - OLB.Q_tables[0][i_0])
			nextfeaturesqval = qtable1[i_0]
		else:
			qtable1 = OLB.Q_tables[j]
			chosenfeatures = OLB.chosenfbyorder # all last chosen features
			idx = 0
			for idx in chosenfeatures[:-2]:
				qtable1 = qtable1[idx]
			tempmax = float('-inf')
			i_0 = chosenfeatures[-2]
			i_1 = chosenfeatures[-1] # currfeature # want to update [i_0][i_1]
			for l in range(1, self.N):
				if qtable1[i_1][l] > tempmax:
					tempmax = qtable1[i_1][l]
			qtable1[i_0][i_1] = qtable1[i_0][i_1] + self.alpha * (r + self.gamma * tempmax - qtable1[i_0][i_1])
			nextfeaturesqval = qtable1[i_1]
		return nextfeaturesqval


	def _updatefeature(self, OLB, f): 
		OLB.h[f] = 1
		assert len(OLB.chosenfbyorder) <= len(OLB.h)
		OLB.chosenfbyorder.append(f)

	def _chooseaction(self, OLB, best_row):
		if np.sum(OLB.h) == 0: 
			nextfeature = np.argmax(best_row)
		else:
			x = random.random()
			if x < self.epsilon:
				choices = []
				for f in range(len(OLB.h)):
					if OLB.h[f] != 1:
						choices.append(f)
				nextfeature = random.choice(choices)
			else:
				nextfeatureidx = np.argsort(np.reshape(best_row,(1,-1)))[::-1]
				for idx in nextfeatureidx[0]:
					if OLB.h[idx] != 1:
						nextfeature = idx
		return nextfeature

	def _sortfeatures(self, OLB):
		tmph = [1] * self.N
		fC = 1
		a_0 = np.argmax(OLB.Q_tables[0])
		ordered_features = [a_0]
		tmph[a_0] = 0
		cState = set([a_0])
		while fC < self.N:
			qtable = OLB.Q_tables[fC]
			while len(cState) > 0:
				idx = cState.pop()
				qtable = qtable[idx]
			row = qtable
			featureidx = np.argsort(np.reshape(row,(1,-1)))[::-1]
			for idx in featureidx[0]:
				if tmph[idx] == 1:
					a_fc = idx
					tmph[idx] = 0
					break
			ordered_features.append(a_fc)
			cState = set(ordered_features)
			fC += 1
		for f in ordered_features:
			assert OLB.T_i[f] == 0
			OLB.T_i[f] = 1 # change feature space
		OLB.cState = cState
		OLB.optimalf = ordered_features


	def _classify(self, query, k, param):
		# query in the format of row vector (1, 224x224)
		img = np.reshape(query, (-1,1)) # column vector
		img -= np.reshape(self.mean_img_col,(-1,1))
		similarity = []
		for olb in self.OLBs:
			t = np.diag(olb.T_i)
			S = t * self.evectors * img
			assert t.shape == (self.numPCs, self.numPCs) # shape is (number of features in featurespace, 1)
			projected = np.dot(self.weights.T, t) 
			assert projected.shape == (len(self.OLBs), self.numPCs) # every row represents a olb image in that space
			projected = np.vstack((projected, S.T)) # concatenate query data to the last input data
			labels_ = Utils._kmeans(projected, k)
			# calculating the reward
			reward = 0
			if param == 'age':
				file_path = 'age.txt'
				gt = [l.strip() for l in open(file_path).readlines()]
				querylabel = gt[labels_[-1]]
				for i in range(len(self.OLBs)):
					pred = gt[labels_[i]]
					gtlabel = gt[np.argmax(self.OLBs[i].w_age_i)]
					if gtlabel == pred:
						reward += 1
				pred = {'querylabel': querylabel, 'reward': reward }
				similarity.append({'olb': olb, 'pred':pred, 'class':gtlabel })
			else:
				file_path = 'gender.txt'
				gt = [l.strip() for l in open(file_path).readlines()]
				querylabel = gt[labels_[-1]]
				for i in range(len(self.OLBs)):
					pred = gt[labels_[i]]
					gtlabel = gt[np.argmax(self.OLBs[i].w_gender_i)]
					if gtlabel == pred:
						reward += 1
				pred = {'querylabel': querylabel, 'reward': reward }
				similarity.append({'olb': olb, 'pred':pred, 'class':gtlabel })

		classsimilarity = collections.defaultdict(int)
		for d in similarity:
			classsimilarity[d['class']] += d['pred']['reward']
		x = classsimilarity
		sorted_= sorted(x.items(), key = operator.itemgetter(1))[::-1]
		top1 = sorted_[0][0]
		if param == 'age':
			top2 = [sorted_[0][0], sorted_[1][0]]
			return top1, top2
		else:
			return top1


