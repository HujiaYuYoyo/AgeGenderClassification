
from PIL import Image
from os import listdir
import os

# {
# '30601258@N03':[
# 		{
# 			imageid:
# 			image:
# 		}
# 		{
# 			imageid:
# 			image:
# 		}
# 		{
# 			imageid:
# 			image:
# 		}
# 	]
# }
# 'imageid': '10424815813_e94629b1ec_o.jpg'
#			  9505396598_2d1fb84849_o.jpg
# 'imageid': coarse_tilt_aligned_face.9.9505396598_2d1fb84849_o.jpg

def loadImages(userid, users):
	if userid not in users: 
		users[userid] = []
	path = os.path.join('./', userid)
	imagesList = listdir(path)
	loadedImages = []
	setids = set()
	for image in imagesList:
		if image.endswith('.jpg'):
			currimg = {}
			# img = Image.open(os.path.join(path, image))
			imgpath = os.path.join(path, image)
			st = str(image)
			imgidparsed = '.'.join(st.split('.')[-2:])
			if imgidparsed in setids: 
				continue
			else:
				setids.add(imgidparsed)
				currimg['imageid'] = imgidparsed
				currimg['image'] = imgpath
				loadedImages.append(currimg)
	print 'for user {}, we have loaded {} number of pictures'.format(userid, len(loadedImages))
	users[userid].extend(loadedImages)

users = {}
def load(userid):
	loadImages(userid, users)
	return users

# user = load('30601258@N03')

