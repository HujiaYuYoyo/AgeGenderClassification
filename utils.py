import skimage
import skimage.io
import skimage.transform
import numpy as np
from PIL import Image
import scipy.misc
from skimage.exposure import rescale_intensity




# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
    img = skimage.io.imread(path)
    # img = Image.open(path).convert('LA') # convert to grayscale
    # im = imread(im_path)
    img = skimage.color.rgb2gray(img)
    img = rescale_intensity(img, out_range=(0, 255))
    # scipy.misc.imsave('./test.png', img)
    # img = img / 255.0
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img


# returns the top1 string
def print_prob(prob, file_path):
    gender = [l.strip() for l in open(file_path).readlines()]

    pred = np.argsort(prob)[::-1]
    top1 = gender[pred[0]]
    # Get top5 label
    # top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    return top1


def load_image2(path, height=None, width=None):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx))


def test():
    img = skimage.io.imread("./test_data/puzzle.jpeg")
    ny = 300
    nx = img.shape[1] * ny / img.shape[0]
    img = skimage.transform.resize(img, (ny, nx))
    skimage.io.imsave("./test_data/test/output.jpg", img)


if __name__ == "__main__":
    test()
