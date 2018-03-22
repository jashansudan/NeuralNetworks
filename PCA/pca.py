import numpy as np


# Main functions, calls the other methods
def main():
    data = read_data()
    weight_vector = train(data)
    output_to_csv(weight_vector, data)


# Reads in Data from sound.csv and converts it to a numpy array
def read_data():
    csv = np.genfromtxt('sound.csv', delimiter=",")
    return csv


# Trains the 2 node neural network using a PCA network as show in class
def train(data):
    # Sets arbitrary weights and a learning rate of 0.3
    weights = [0.1, 0.2]
    learning_rate = 0.3
    # Iterates through the data set, and trains the network with each value
    for vector in data:
        y = dot(weights, vector)
        delta_w = learning_rate * (vector * y - y * y * np.array(weights))
        weights = weights + delta_w
    return weights


# Calculates the dot product of the weights and a supplied vector
def dot(weight, vector):
    product = 0
    for i in range(len(weight)):
        product += weight[i] * vector[i]
    return product


# Using the final weight vector, iterates through the data and calculates a
# compressed data value and writes it to file
def output_to_csv(weight_vector, data):
    file = open("output.csv", "w")
    for row in data:
        product = dot(weight_vector, row)
        file.write('%s\n' % product)
    file.close()
    return


main()
