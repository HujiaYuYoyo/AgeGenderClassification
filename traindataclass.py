import random
import numpy as np
from inputdata import inputdata
from inputimage import load
import collections
import os
import utils
import json

class traindata(object):
	def __init__(self):
		self.age = ['(0,2)', '(4,6)', '(8,12)', '(15,20)', '(25,32)', '(38,43)', '(48,53)', '(60,)']
		self.gender = ['m','f']
		self.traindata = []
		self.trainimagedata = []
		self.trainuserid = []
		self.valuserid = []
		self.valdata = []
		self.valfile = 'fold_4_data.txt'
		self.vallabel = []
		self.valimagedata = []
		self.traindataconverted = []
		self.valdataconverted = []
		self.train = []
		self.test = []

		# given userid and imgid userimagetoage[userid][imageid] = age of that image
		self.userimagetoage = {}
		self.userimagetogender = {}

	def gettraindata(self, filename):
		data = inputdata(filename)
		self.traindata = data
		self.trainuserid = data.keys()

	def gettrainuserimage(self, userid):
		data = load(userid)
		self.trainimagedata = data

	def label(self, typedata = 'train'):
		if typedata == 'train':
			data = self.traindata
		elif typedata == 'val':
			data = self.valdata
		for userid in data:
			if userid not in self.userimagetoage:
				self.userimagetoage[userid] = {}
				self.userimagetogender[userid] = {}
			for x in data[userid]:
				currimgid = x['imageid']
				currage = x['age']
				currgender = x['gender']
				self.userimagetoage[userid][currimgid] = currage
				self.userimagetogender[userid][currimgid] = currgender
		assert len(self.userimagetogender) == len(self.userimagetoage)
		print 'successfully labelled %s user images in label data' % len(self.userimagetogender)

	# this function converts all training data to be a list of [{},{},{}] image data objects
	# {userid, imageid, gender, genderlabel, age, agelabel, img}
	def convertdata(self):
		for userid in self.trainuserid:
			userimages = self.trainimagedata[userid]
			for i in range(len(userimages)):
				img = userimages[i]
				if img['imageid'] not in self.userimagetogender[userid]:
					continue
				temp = {}
				temp['userid'] = userid
				gender = self.userimagetogender[userid][img['imageid']]
				temp['imageid'] = img['imageid']
				temp['gender'] = gender
				genderlabel = [1 if self.gender[i] == gender else 0 for i in range(2)]
				temp['genderlabel'] = genderlabel
				age = self.userimagetoage[userid][img['imageid']]
				temp['age'] = age
				agelabel = [1 if self.age[i] == age else 0 for i in range(8)]
				temp['agelabel'] = agelabel
				# img = utils.load_image(img['image'])
				img = img['image'] # path to image
				# x = np.array(img)
				# img = x.flatten().reshape(1, -1)
				# img = img.reshape((1, 224x224))
				# assert img.shape == (1, 224*224)
				# assert np.sum(img) > 0
				temp['imagepath'] = img # 'imagepath' gives the path
				self.traindataconverted.append(temp)
		print 'traindataconverted size is: ', len(self.traindataconverted)

	# this function samples 140 images as train and test data
	# 80 m 80 f
	# 20 images per age group - 10 m 10 f
	# randomly select an image from converted all dataset
	def sampledata(self, imgpergender, trainfile, testfile):
		sampled = {}
		dataready = []
		t = True
		print len(self.traindataconverted)
		assert len(self.traindataconverted) != 0
		if len(trainfile) < 20:
			dataready = self.traindataconverted
		else:
			while t:
				x = random.choice(self.traindataconverted)
				if x in dataready: 
					continue
				else:
					if x['age'] in sampled:
						if len(sampled[x['age']][x['gender']]) == imgpergender:
							# print x['age'], x['gender'], 'has 10 samples already!'
							continue
						else:
							sampled[x['age']][x['gender']].append(x)
							dataready.append(x)
					else:
						sampled[x['age']] = {}
						sampled[x['age']]['m'] = []
						sampled[x['age']]['f'] = []
					if len(dataready) == 14 * imgpergender:
						t = False
			print 'successfully sampled %s data ...' % len(dataready)
		# train = random.sample(dataready, 96)
		# test = []
		# for x in dataready:
		# 	if x not in train:
		# 		test.append(x)
		random.shuffle(dataready) # inplace
		x = len(dataready) * 0.6
		train = dataready[:int(x)]
		test = dataready[int(x):]
		train = { 'train': train }
		test = { 'test': test }
		self.train = train
		self.test = test
		with open(trainfile, 'w') as f:
			json.dump(train, f)
		with open(testfile, 'w') as f:
			json.dump(test, f)
		return train, test



