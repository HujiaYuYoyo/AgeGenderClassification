#!/usr/bin/env python
import random
import numpy as np

class OLB(object):
	def __init__(self, x_i, w_age_i, w_gender_i, T_i, mu, pcs, img_col):
		# x_i is the representation of a sample face in the Eigenface space
		# T_i is the binary vector demonstrating which eigenfaces are the 
		# optimal subset to describe x_i
		# w_i is the OBL class label for age and gender respectively
		# print 'Initiated a new OLB object!'
		self.w_age_i = w_age_i
		self.w_gender_i = w_gender_i
		self.x_i = x_i
		self.T_i = T_i
		self.pcs= pcs
		self.selectedf = [] # list of selected features, features = eigenfaces interchangably, all features = x_i
		self.h = [0]*len(T_i) # indicator of whether this feature has already been selected
		self.Q_tables = []
		self.mu = mu # mean used to center the data before applying PCA
		self.img_col = img_col
		self.chosenfbyorder = [] # append chosen feature index sequentially
		self.cState = 0
		self.optimalf = 0

