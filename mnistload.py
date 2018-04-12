from mnist import MNIST
import numpy as np

def load_set(set):

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



