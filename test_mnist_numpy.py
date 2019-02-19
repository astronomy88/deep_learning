""" These are test cases for the mnist_numpy functions """
from mnist_numpy import *
import pdb

def softmax_columns_sum_to_one():
    """ Does each column in the softmax matrix sum to one ? """
    num_images = 8
    image_size = 28

    batch_data = get_train_data(num_images, image_size)
    input_vectors = create_input_vectors(batch_data, image_size)

    weights = np.random.randn(10,image_size * image_size + 1)*0.01

    z = np.dot(weights, input_vectors) #-- 10 x 8

    #-- Should still be a 10 x 8 matrix and each column should sum to 1
    probs = softmax(z)
    for i in range(num_images):
        assert np.sum(probs[:,i]) - 1.0 < 0.0001
    return True

def main():
    assert softmax_columns_sum_to_one()

if __name__ == "__main__":
    main()