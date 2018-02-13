import random
from math import exp
from csv import reader


def getData(filename):
    data = []
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            data.append(row)
    data = data[1:]
    return data


def convertToFloat(data):
    for row in range(len(data)):
        for col in range(len(data[0])):
            data[row][col] = float(data[row][col])
        data[row][-1] = int(data[row][-1])
    return


def normalizeData(data):
    for col in range(len(data[0]) - 1):
        col_max = float("-inf")
        col_min = float("inf")
        for row in range(len(data)):
            if data[row][col] > col_max:
                col_max = data[row][col]
            if data[row][col] < col_min:
                col_min = data[row][col]

        for row in range(len(data)):
            data[row][col] = (data[row][col] - col_min) / (col_max - col_min)
    return


def standardizeDefiningLabel(data):
    for row in range(len(data)):
        if data[row][-1] == 5:
            data[row][-1] = 0
        elif data[row][-1] == 7:
            data[row][-1] = 1
        else:
            data[row][-1] = 2

def setupNetwork(num_input_nodes, num_hidden_nodes, num_output_nodes):
    network = []
    hidden_layer = [{'weights':[random.uniform(0, 1) for i in range(num_input_nodes + 1)]} for i in range(num_hidden_nodes)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random.uniform(0, 1) for i in range(num_hidden_nodes + 1)]} for i in range(num_output_nodes)]
    network.append(output_layer)
    return network


def calculateActivation(weights, input_vector):
    acitvation = weights[-1]
    for i in range(len(weights) - 1):
        acitvation += weights[i] * input_vector[i]
    return acitvation


def propogateForward(network, row):
    input_vector = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            acitvation = calculateActivation(neuron['weights'], input_vector)
            neuron['output'] = transfer(acitvation)
            new_inputs.append(neuron['output'])
        input_vector = new_inputs
    return input_vector


def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


def backwardPropogateError(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = []
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * derivative(neuron['output'])


def derivative(output):
    return output * (1.0 - output)


def updateWeights(network, row, learning_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += learning_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] + learning_rate * neuron['delta']


def trainNetwork(network, train, learning_rate, epochs, num_output_nodes):
    for epoch in range(epochs):
        sum_error = 0
        for row in train:
            outputs = propogateForward(network, row)
            expected = [0 for i in range(num_output_nodes)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i])**2 for i in range(len(expected))])
            backwardPropogateError(network, expected)
            updateWeights(network, row, learning_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learning_rate, sum_error))


data = getData("assignment2data.csv")
convertToFloat(data)
normalizeData(data)
standardizeDefiningLabel(data)
data = data[:1000]
input_nodes = len(data[0]) - 1
output_nodes = len(set([row[-1] for row in data]))
network = setupNetwork(input_nodes, 11, output_nodes)
trainNetwork(network, data, 0.5, 10000, output_nodes)
