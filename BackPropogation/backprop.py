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
    hidden_layer = []
    output_layer = []
    for i in range(num_hidden_nodes):
        random_weights = [random.uniform(0, 1) for j in range(num_input_nodes + 1)]
        node = {"weights": random_weights}
        hidden_layer.append(node)
    for i in range(num_output_nodes):
        random_weights = [random.uniform(0, 1) for j in range(num_hidden_nodes + 1)]
        node = {"weights": random_weights}
        output_layer.append(node)
    network.append(hidden_layer)
    network.append(output_layer)
    return network


def trainNetwork(network, training_data, learning_rate, epochs, num_output_nodes):
    for epoch in range(epochs):
        sum_error = 0
        for row in training_data:
            classification = classify(network, row)
            expected = [0] * num_output_nodes
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - classification[i])**2 for i in range(len(expected))])
            backwardPropogateError(network, expected)
            updateWeights(network, row, learning_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learning_rate, sum_error))


def classify(network, row):
    input_vector = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            acitvation = dotProduct(neuron['weights'], input_vector)
            neuron['output'] = transfer(acitvation)
            new_inputs.append(neuron['output'])
        input_vector = new_inputs
    return input_vector


def dotProduct(weights, input_vector):
    acitvation = weights[-1]
    for i in range(len(weights) - 1):
        acitvation += weights[i] * input_vector[i]
    return acitvation


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
                    error += (neuron['weights'][j] * neuron['change'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['change'] = errors[j] * derivativeOfLine(neuron['output'])


def derivativeOfLine(output):
    return output * (1.0 - output)


def updateWeights(network, row, learning_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += learning_rate * neuron['change'] * inputs[j]
            neuron['weights'][-1] += learning_rate * neuron['change']


def test_network(network, test_data, num_output_nodes):
    incorrect = 0
    for row in test_data:
        outputs = classify(network, row)
        prediction = outputs.index(max(outputs))
        if row[-1] != prediction:
            incorrect += 1
    print incorrect


data = getData("assignment2data.csv")
convertToFloat(data)
normalizeData(data)
standardizeDefiningLabel(data)
train_data = data[:500]
input_nodes = len(data[0]) - 1
output_nodes = len(set([row[-1] for row in data]))
network = setupNetwork(input_nodes, 25, output_nodes)
trainNetwork(network, train_data, 0.5, 500, output_nodes)
#test_data = data[1000:1200]
#test_network(network, test_data, output_nodes)
