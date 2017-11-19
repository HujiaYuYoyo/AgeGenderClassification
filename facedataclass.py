from inputdata import inputdata
from inputimage import load
import collections
import utils
import numpy as np

# filename = ['fold_0_data.txt', 'fold_1_data.txt', 'fold_2_data.txt', 'fold_3_data.txt', 'fold_4_data.txt']
# userid = '115321157@N03'

class facesdata(object):
	def __init__(self):
		self.age = ['(0,2)', '(4,6)', '(8,12)', '(15,20)', '(25,32)', '(38,43)', '(48,53)', '(60,)']
		self.gender = ['m','f']
		self.labeldata = []
		self.imagedata = []
		self.userid = []
		self.batchsize = 10
		self.loaded = collections.defaultdict(int)

		# given userid and imgid userimagetoage[userid][imageid] = age of that image
		self.userimagetoage = {}
		self.userimagetogender = {}

	def setbatchsize(self, num):
		self.batchsize = num

	# afer calling this function - we have inputted all the txt data from filenames to store in the 
	# self.labeldata in the form of list of img dict's corresponding to key userid
	def getlabeldata(self, filename):
		data = inputdata(filename)
		self.labeldata = data
		self.userid = data.keys()

	# after calling this function - we have imported all the images corresponding to userid - 
	# which is also the name of the path to be loaded 
	# notice we can import images by looping through userid in self.userid upon repuest
	def getuserimage(self, userid):
		data = load(userid)
		self.imagedata = data

	# must have imported txt data first before calling this function
	# this function consolidates and reorganizes the import labeldata by 
	# creating two dictionaries userimagetoage and userimagetogender
	# both of which can take [userid][imageid] and gives age and gender of that image of that user
	def labeling(self):
		for userid in self.labeldata:
			if userid not in self.userimagetoage:
				self.userimagetoage[userid] = {}
				self.userimagetogender[userid] = {}
			for x in self.labeldata[userid]:
				currimgid = x['imageid']
				currage = x['age']
				currgender = x['gender']
				self.userimagetoage[userid][currimgid] = currage
				self.userimagetogender[userid][currimgid] = currgender
		assert len(self.userimagetogender) == len(self.userimagetoage)
		print 'successfully labelled %s user images in label data' % len(self.userimagetogender)

	# can only be called after importing txt and image data
	# after calling this function - we have a batchsize of gender labels list (1-hot label)
	# and age labels list
	# and concatenated images of batch size
	# all in the format of ideal training format
	def loaddatabatch(self):
		sze = self.batchsize
		counter = 0
		loadedgenderlabel = []
		loadedagelabel = []
		loadedimage = []
		start = 0
		for userid in self.userid:
			if self.loaded[userid] == -1: # has been completely loaded before
				continue
			elif self.loaded[userid] > 0: # has been loaded partilly
				start = self.loaded[userid]
			userimages = self.imagedata[userid]
			for i in range(start, len(userimages)):
				img = userimages[i]
				gender = self.userimagetogender[userid][img['imageid']]
				currlabel = [1 if self.gender[i] == gender else 0 for i in range(2)]
				loadedgenderlabel.append(currlabel)
				age = self.userimagetoage[userid][img['imageid']]
				currage = [1 if self.age[i] == age else 0 for i in range(8)]
				loadedagelabel.append(currage)
				img = utils.load_image(img['image'])
				img = img.reshape((1, 224, 224, 3))
				if len(loadedimage) == 0: 
					loadedimage = img
				else:
					loadedimage = np.concatenate((loadedimage, img), 0)
				counter += 1
				if counter == sze:	
					self.loaded[userid] = i
					return loadedgenderlabel, loadedagelabel, loadedimage
			self.loaded[userid] = -1

