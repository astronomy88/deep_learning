""" 
This is an exploration in numpy and MNIST dataset
Using concepts generously from here: https://deepnotes.io/softmax-crossentropy
"""
import gzip
import numpy as np
import matplotlib.pyplot as plt 

import pdb

def create_input_vectors(batch_data, image_size):
    """Will create an num_features x num_images matrix of input vectors"""
    m = batch_data.shape[0]
    num_features = image_size * image_size + 1
    input_vectors = np.zeros(([num_features, m]))

    #-- Now create all the input vectors, column by column
    for i in range(0,m):
        #-- Ravel input array:
        input_vector = batch_data[i].reshape(image_size * image_size, 1)
        #-- Need to add a bias term:
        input_vector = np.insert(input_vector,0,1,0) #-- 785 x 1
        input_vectors[:,i] = input_vector.squeeze()

    return input_vectors

def get_train_data(num_images, image_size):
    """Return an array of image training data - image_size x image_size """
    f = gzip.open('data/train-images-idx3-ubyte.gz', 'r')
    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    batch_data = data.reshape(num_images, image_size, image_size, 1)
    return batch_data

def softmax(z):
    """Returns a 10 x num_images matrix (original input size) of probabilities
        We'll use the safe version so that we don't have large exponents"""
    softmax = np.zeros((z.shape))
    m = z.shape[1]
    for i in range(0,m):
        col_z = z[:,i]
        s = np.exp(col_z - np.max(col_z))
        s = s / np.sum(s)
        softmax[:, i] = s
    return softmax

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

def one_hot_index(y):
    """ Converts one-hot vectors back into ndx for that vector
        Will return a num_images x 1 array
    """
    indices = np.zeros(y.shape[1])
    pdb.set_trace()
    return indices

def grab_labels(num_images):
    f_test = gzip.open('data/train-labels-idx1-ubyte.gz', 'r')
    f_test.read(8)
    buf_test = f_test.read(1*num_images)
    labels = np.frombuffer(buf_test, dtype=np.uint8).astype(np.int64)
    return labels

def cross_entropy(probs, y):
    """ Defined as H(y,p) = 1/m * (-sum_i (y_i) * log(p_i))
        Here, the parameter y is labels as columns, 
        so, y.shape = (10 x num_images) 
        Since y is a one-hot vector, only the ndx where val = 1 remains

        p should be a 10 x num_images matrix, containing all probabilities

        We should return a single value, the loss
    """
    m = y.shape[1]
    loss = 0

    for i in range(0,m):
        y_temp = y[:,i]
        probs_temp = probs[:,i]
        temp_loss = probs_temp[y_temp == 1][0]

        #-- Need to add a hack here for when prob is 1 where it shouldn't be
        if temp_loss == 0:
            loss += -202.97
        else:
            loss += np.log(temp_loss)

    loss = -1 * loss / m
    return loss

def grad_cross_entropy(weights, input_vectors, y):    
    """ 
    dL/dW21 = z2(p1 - y1)
    We're going to get a grad for each example
    Then we will need to average them to apply for
    the next round of weights
    
    input_vectors is a 785 x num_images matrix

    h is a 10x8 matrix of probabilities using old weights

    This function should return 10 x 785 matrix
    """
    num_features = input_vectors.shape[0]
    m = input_vectors.shape[1]

    grad = np.zeros(([10,num_features]))

    z = np.dot(weights, input_vectors)
    probs = softmax(z)

    #-- Must loop through each image:
    m = input_vectors.shape[1]
    for i in range(0,m):
        input_vector = input_vectors[:,i]
        temp_input = np.array([input_vector,]*10)
        
        prob = probs[:,i]
        temp_prob = np.array([prob,] * 785)

        #-- Before multiplying by x, we need to subtract 1 from prob for when the label matches that digit
        temp_y = y[:,i]
        temp_prob[:,temp_y==1] -= 1

        #-- temp_grad holds each "digit" row multiplied by its prob of that digit
        temp_grad = np.multiply(temp_input, temp_prob.T) #-- 10 x 785 matrix

        grad += temp_grad

    #-- Now that we have grad summed, we need to get the average
    grad /= m

    return grad

def feedforward(weights, input_vectors, y):
    """
    Returns loss
    Parameters:
        - weights - the current weights that should be passed through
        - input_vectors - these are the input features (num_features x num_images)
        - y - one_hot vector of labels (num_digits x num_images) 
    """
    z = np.dot(weights, input_vectors)
    h = softmax(z)
    loss = cross_entropy(h, y)
    return loss

def predict(weights, feature_vector):
    """
    Should predict a label given a single feature vector
    Parameters:
        - weights is our model, and is 10 x 785 matrix
        - feature_vector is a 785 x 1 feature vector
    """
    z = np.dot(weights, feature_vector)
    probs = softmax(z)
    return np.argmax(probs)

#-- Works quite well for 8 images - only 4 iterations needed
#   The more images we had, the more iterations we needed. With 50 images, 10 iterations needed
#   With 500 images, 50 iterations needed. With 1000 images, 50 iterations only got 80% accuracy.
#   Then, with 100 iterations on 1000 images, w got 99.9% accuracy. This is all on training data btw.
num_images = 1000
image_size = 28

batch_data = get_train_data(num_images, image_size)
input_vectors = create_input_vectors(batch_data, image_size)

#-- Weight matrix (will be a 10 x 785 matrix)
#   Initialize with random numbers
weights = np.random.randn(10,image_size * image_size + 1)*0.01

labels = grab_labels(num_images)
y = one_hot(labels)

loss = feedforward(weights, input_vectors, y)
print(f"Initial Loss: {loss}")

learning_rate = 0.01 #-- Can test between 0.001 and 0.01
num_iterations = 100

for i in range(0,num_iterations):
    grad = grad_cross_entropy(weights, input_vectors, y)
    weights = weights - (learning_rate * grad)
    loss = feedforward(weights, input_vectors, y)
    print(f"Iteration {i}, loss = {loss}")

#-- By now, our weights should be good enough to predict.

count = 0
correct = 0
for which_ndx in range(0,num_images):

    feature_vector = input_vectors[:,which_ndx:which_ndx+1]

    prediction = predict(weights, feature_vector)

    # print(f"Our prediction: {prediction}")
    # print(f"Truth: {labels[which_ndx]}")
    if prediction == labels[which_ndx]:
        correct += 1
    count += 1
    # print()

print(f"We got {correct} correct out of {count} total images.")
print(f"That is {correct/count} accuracy")