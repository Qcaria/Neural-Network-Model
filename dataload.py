from mnist import MNIST
import numpy as np
import scipy
from scipy import ndimage

def load_mnistset(set):

    mndata = MNIST('samples')
    if set == "train":
        images, labels = mndata.load_training()
    elif set == "test":
        images, labels = mndata.load_testing()
    else:
        raise ValueError('Non valid set')

    X = np.array(images).T/255
    Y_raw = []

    for i in range(len(labels)):
        e = [0] * 10
        j = labels[i]
        e[j] = 1
        Y_raw.append(e)

    Y = np.array(Y_raw).T
    return X, Y


def load_image(image_name, pix_size):
    fname = "samples/" + image_name
    image = np.array(ndimage.imread(fname, flatten=True))
    my_image = scipy.misc.imresize(image, size=(pix_size, pix_size)).reshape((pix_size*pix_size, 1))
    my_image = my_image / 255.
    return my_image

