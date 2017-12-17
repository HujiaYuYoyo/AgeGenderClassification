#!/usr/bin/env python
import random
import numpy as np
from sklearn.cluster import KMeans
from OLB import OLB
import scipy
import skimage
import scipy.misc
from skimage import io
from matplotlib import pyplot as plt
from skimage.exposure import rescale_intensity
import skimage.transform
from traindataclass import traindata
import utils
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.misc import toimage
import json
# import Pycluster


# this function projects one representation points x_j into 
# space defined by h 
# returns the projected data p_j
def _projectone(OLB, x_j, h, mu):
	U = np.dot(OLB.pcs, h.T)
	p_j = np.dot(x_j - mu, U.T)
	return p_j

# this function takes in an face image X and transforms it to top num of principal components and 
# returns the transformed image as Z and saves all eigenvectors in self.features
# rowvar=False to indicate that columns represent variables.
def _PCA(topk, m, images): # taking in m batch of face images
	# right now assume face image is in dimension (224, 224, 1)
	# flatten to one row of (1,224*224*1)
	assert images.shape == (m, 224*224)
	L = images.T # each column is an image vector now shape is (50176 x m)
	mean_img_col = np.mean(L, axis=1) # sample mean of each feature vector 
	# print 'mean_img_col shape is ', mean_img_col.shape
	for j in range(m):   # subtract each column vector from its feature sample mean for all training images
		L[:, j] -= mean_img_col
	C = np.matrix(L.transpose()) * np.matrix(L) # m x m
	C /= m
	# print 'covariance matrix dimension is ', C.shape
	evalues, evectors = np.linalg.eig(C)   # m x m  
	# print evalues[:15]                     
	sort_indices = evalues.argsort()[::-1]  # getting their correct order - decreasing
	evalues = evalues[sort_indices]       
	evectors = evectors[sort_indices] 

	evalues_count = topk
	evalues = evalues[:evalues_count]
	evectors = evectors[:evalues_count]

	evectors = evectors.transpose() # change eigenvectors from rows to columns
	evectors = L * evectors  # left multiply to get the correct evectors (50176, m)
	# print 'correct evectors shape is ', evectors.shape
	norms = np.linalg.norm(evectors, axis=0) # find the norm of each eigenvector
	evectors = evectors / norms # normalize all eigenvectors
	weights = evectors.transpose() * L # computing the weights (numPCs, m)
	# print 'shape of evectors and L is: ', evectors.shape, L.shape 
	# print 'shape of weights (evectors * L) is ', weights.shape
	with open('./finalresult/weights.txt','w') as f:
		np.savetxt(f, weights)
	with open('./finalresult/meanimg.txt', 'w') as f:
		np.savetxt(f, mean_img_col)
	with open('./finalresult/eigenvectors.txt', 'w') as f:
		np.savetxt(f, evectors.transpose())
	print 'Finished writing weights and eigenvectors and mu to files ...'
	return weights, mean_img_col, evectors.transpose() # to rows # (m, 224*224)
# 	correct evectors shape is  (50176, 100)
#   shape of evectors and L is:  (50176, 100) (50176, 146)
#   shape of weights (evectors * L) is  (100, 146)


# this function finds the class labels of the kmeans of the projected data from p 
# it returns an binary array of size len(p) indicating the class label of each feature
def _kmeans(p, k):
	kmeans = KMeans(k, random_state=0).fit(p)
	# labels, error, nfound = Pycluster.kcluster(p, k)
	return kmeans.labels_


# can only be called after importing txt and image data and after calling converttraindata
# this function randomly samples batchsize of training data and returns gender labels, age 
# and images, all of batch size
def loadtraindatabatch(batchsize, traindataconverted):
	sze = batchsize
	loadedgenderlabel = []
	loadedagelabel = []
	loadedimage = []
	sampled = random.sample(traindataconverted, sze) # without replacement
	for x in sampled:
		loadedgenderlabel.append(x['genderlabel'])
		loadedagelabel.append(x['agelabel'])
		img = utils.load_image(x['imagepath'])
		x = np.array(img)
		img = x.flatten().reshape(1, -1)
		assert img.shape == (1, 224*224)
		assert np.sum(img) > 0
		if len(loadedimage) == 0: 
			loadedimage = img
		else:
			loadedimage = np.concatenate((loadedimage, img), 0) # stack rows
	print 'Loaded and concatenated image shape after loading {} number of training data is {}'.format(batchsize, loadedimage.shape)
	return loadedgenderlabel, loadedagelabel, loadedimage



def test(learner, train, test, displayk, saveimg = './finalresult/after.png'):
	# {userid, imageid, gender, genderlabel, age, agelabel, imagepath}
	# print 'train data looks like ', len(train), train[0]
	testimg = random.choice(train)
	weights, mean_img_col, evectors = learner.weights, learner.mean_img_col, learner.evectors
	# print weights.shape
	# print mean_img_col.shape
	# print evectors.shape
	# show the average face by adding up all the eigenfaces with weight = 1
	avgface = np.matrix.sum(evectors, axis = 0)
	avgfaceimg = np.reshape(avgface, (224,224))
	avgfaceimg = avgfaceimg.astype(complex).real
	scipy.misc.toimage(avgfaceimg).save('./finalresult/avgface.png')
	# plt.imshow(avgfaceimg, interpolation='nearest')
	# plt.title('Average Eigenface')
	# plt.show()

	# scipy.misc.toimage(avgface).save('averageface.png')
	# avgface = skimage.io.imread('averageface.png')

	# io.imshow(avgface)
	# io.show()

	print 'finished importing weights, mu, and eigenvectors of the sample data ...'
	testimg = utils.load_image(testimg['imagepath'])
	scipy.misc.toimage(testimg).save('./finalresult/testbefore.png')
	# io.imshow(img)
	# io.show()
	# plt.imshow(testimg, interpolation='nearest')
	# plt.title('TEST-Before Reconstruction')
	# plt.show()

	img = np.reshape(testimg, (-1,1)) # column vector
	img -= np.reshape(mean_img_col,(-1,1))
	S = evectors * img

	sortedidx= np.argsort(S)
	topkweights = S[sortedidx][:displayk]
	topkevecs = evectors[sortedidx][:displayk]
	reconstructed = 0
	for i in range(displayk):
		reconstructed += topkweights[i][0] * topkevecs[i]
	reconstructed = np.reshape(reconstructed, (224, 224))
	reconstructed = reconstructed.astype(complex).real
	scipy.misc.toimage(reconstructed).save(saveimg)
	# plt.imshow(reconstructed, interpolation='nearest')
	# plt.title('TEST-After Reconstructed')
	# plt.show()
	# scipy.misc.toimage(reconstructed).save(saveimg)

	# afterimg = skimage.io.imread(saveimg)
	# io.imshow(afterimg)
	# io.show()
	# testimg = mpimg.imread(testimg['imagepath'])
	print 'successfully saved the reconstructed output image (note that this is just top %s eigenfaces)' % displayk
	if displayk < 10:
		for idx, e in enumerate(topkevecs):
			e = np.reshape(e, (224, 224))
			fig = plt.figure()
			plt.imshow(e, interpolation='nearest')
			plt.title('%s eigenvector' % idx)
			name = '%s eigenvector.png' % idx
			fig.savefig(name)

# calculate class averages instead of averages of all training data
def classavg(learner, train):
	weights, mean_img_col, evectors = learner.weights, learner.mean_img_col, learner.evectors
	# show the average face by adding up all the eigenfaces with weight = 1

	avgface = np.matrix.sum(evectors, axis = 0)
	avgfaceimg = np.reshape(avgface, (224,224))
	avgfaceimg = avgfaceimg.astype(complex).real
	plt.imshow(avgfaceimg, interpolation='nearest')
	plt.title('Average Eigenface')
	plt.show()



# this function tests classification accuracy on the eigenface weights only
# by subtracting the image mean - multiple by the eigenvectors and 
# obtain the weights on each eigenvector
# and calculate which weights are the closest to the obtained image weight 
# and assign the image label to that weight's label
def testeigenface(learner, train, test, param, outputfile = './finalresult/eigenfacepredict.json'):
	# test dataset looks like: 
	# {userid, imageid, gender, genderlabel, age, agelabel, img}
	# img.shape = (1, 224*224)
	correct = 0
	for x in test:
		img = utils.load_image(x['imagepath'])
		img = np.reshape(img, (-1,1)) # column vector
		img -= np.reshape(learner.mean_img_col,(-1,1))
		S = learner.evectors * img
		# print 'S shape is ', S.shape
		# assert S.shape == (learner.m, 1)

		diff = learner.weights - S # finding the min ||W_j - S||
		norms = np.linalg.norm(diff, axis=0)
		closest_face_idx = np.argmin(norms) # idx corresponding to train data
		predict = train[closest_face_idx]
		x['predict'] = predict
		if predict[param] == x[param]:
			correct += 1
	accu = correct*1./len(test)
	test = { 'test': test }
	with open(outputfile, 'a') as f:
		json.dump(test, f)
	with open('./finalresult/testeigenface.txt','a') as f:
		f.write('Accuracy for testing on {} test dataset with just the eigenfaces for {} is {} \n'.format(len(test['test']), param, accu))
	return accu


