""" 
This is an exploration in numpy and MNIST dataset
Using concepts generously from here: https://deepnotes.io/softmax-crossentropy
"""
import gzip
import numpy as np
import matplotlib.pyplot as plt 

import pdb

def softmax(z):
    """Returns a 10x1 matrix (original input size) of probabilities
        We'll use the safe version so that we don't have large exponents"""  
    s = np.exp(z - np.max(z))
    return s / np.sum(s)

def show_image(image_data):
    image = np.asarray(image_data).squeeze()
    plt.imshow(image)
    plt.show()

def one_hot(labels):
    """ Convert the y-labels to a one-hot vector of y-labels
        Will return a shape of 10 x num_labels """
    m = labels.shape[0]
    y = np.zeros((10, m))
    for i in range(0,m):
        y[labels[i],i] = 1
    return y

def cross_entropy(p, y):
    """ Defined as H(y,p) = -sum_i (y_i) * log(p_i)
        Here, the parameter y is labels as columns, 
        so, y.shape = (10 x num_images) 
    """
    m = y.shape[0]
    # for i in range(0,m):



f = gzip.open('data/train-images-idx3-ubyte.gz', 'r')

image_size = 28
num_images = 8
# num_images = 200

f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
batch_data = data.reshape(num_images, image_size, image_size, 1)

first_input = batch_data[0]

#-- Ravel input array:
input_vector = first_input.reshape(image_size * image_size, 1)
#-- Need to add a bias term:
input_vector = np.insert(input_vector,0,1,0) #-- 785 x 1

#-- Weight matrix (will be a 10 x 785 matrix)
#   Initialize with random numbers
weights = np.random.randn(10,image_size * image_size + 1)*0.01

z = weights.dot(input_vector) #-- 10 x 1

#-- Output of probabilities
h = softmax(z)

f_test = gzip.open('data/train-labels-idx1-ubyte.gz', 'r')

f_test.read(8)
buf_test = f_test.read(1*num_images)
labels = np.frombuffer(buf_test, dtype=np.uint8).astype(np.int64)
print(labels)

pdb.set_trace()