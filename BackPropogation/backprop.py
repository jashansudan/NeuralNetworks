# importing the different modules needed for my code
import random
from math import exp
from csv import reader


# Function to get data from excel file and put into a python array
def getData(filename):
    data = []
    with open(filename, "r") as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            data.append(row)
    data = data[1:]
    return data


# Convert the string array values into floats, so data is adjustable
def convertToFloat(data):
    for row in range(len(data)):
        for col in range(len(data[0])):
            data[row][col] = float(data[row][col])
        data[row][-1] = int(data[row][-1])
    return


# Goes through data and scales all values to be in the range 0 - 1
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


# Changes our labelled data to be binary ranges
def standardizeDefiningLabel(data):
    for row in range(len(data)):
        if data[row][-1] == 5:
            data[row][-1] = 0
        elif data[row][-1] == 7:
            data[row][-1] = 1
        else:
            data[row][-1] = 2


# Sets up our 3 layer network, including input layer, hidden and output layer
def setupNetwork(num_input_nodes, num_hidden_nodes, num_output_nodes):
    network = []
    # Each layer is represented as an array of neurons, each storing a weight connection
    # to each previous layer's neurons
    hidden_layer = []
    output_layer = []

    # Initialize hidden layer
    for i in range(num_hidden_nodes):
        random_weights = [random.uniform(0, 1) for j in range(num_input_nodes + 1)]
        prev_change = [0] * (num_input_nodes + 1)
        node = {"weights": random_weights, "prev_change": prev_change}
        hidden_layer.append(node)

    # Initalize output layer
    for i in range(num_output_nodes):
        random_weights = [random.uniform(0, 1) for j in range(num_hidden_nodes + 1)]
        prev_change = [0] * (num_hidden_nodes + 1)
        node = {"weights": random_weights, "prev_change": prev_change}
        output_layer.append(node)

    # Add both layers to a single network
    network.append(hidden_layer)
    network.append(output_layer)
    return network


# Iterates through our training data to train the network
def trainNetwork(network, training_data, learning_rate, epochs, num_output_nodes):
    for epoch in range(epochs):
        sum_error = 0
        for row in training_data:
            # Compares predicted classification with expected classification then back propagates error
            classification = classify(network, row)
            expected = [0] * num_output_nodes
            expected[row[-1]] = 1
            backPropegate(network, expected)
            updateWeights(network, row, learning_rate)
        print "Epoch " + str(epoch) + " completed!"


# Attempts to classify the row data, by passing data through each of the layers in the network
def classify(network, row):
    input_vector = row
    hidden_layer_output = []
    for neuron in network[0]:
        activation = dotProduct(neuron["weights"], input_vector)
        neuron["output"] = sigmoid(activation)
        hidden_layer_output.append(neuron["output"])

    output_vector = []
    for neuron in network[1]:
        acitvation = dotProduct(neuron["weights"], hidden_layer_output)
        neuron["output"] = sigmoid(acitvation)
        output_vector.append(neuron["output"])
    return output_vector


# Calculates the activation at each neuron, by doing the dot product
def dotProduct(weights, input_vector):
    acitvation = weights[-1]
    for i in range(len(weights) - 1):
        acitvation += weights[i] * input_vector[i]
    return acitvation


# Calculates the sigmoid activation from the sum of the network output
def sigmoid(activation):
    return 1.0 / (1.0 + exp(-activation))


# Using the error detected, back propegates correction through the network
def backPropegate(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        # Calculates error for each neuron
        errors = []
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron["weights"][j] * neuron["change"])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron["output"])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron["change"] = errors[j] * derivativeOfLine(neuron["output"])


# Calculates the derivative of the sigmoid function
def derivativeOfLine(output):
    return output * (1.0 - output)


# Using the errors calculated, along with momentum, the weights are adjusted
def updateWeights(network, row, learning_rate):
    inputs = row[:-1]

    # Adjusting the hidden layer
    for neuron in network[0]:
        for j in range(len(inputs)):
            change = learning_rate * neuron["change"] * inputs[j]
            neuron["weights"][j] += change + 0.75 * neuron["prev_change"][j]
            neuron["prev_change"][j] = change
        neuron["weights"][-1] += learning_rate * neuron["change"]

    # Adjusting output layer weights
    hidden_layer_input = [neuron["output"] for neuron in network[0]]
    for neuron in network[1]:
            for j in range(len(hidden_layer_input)):
                change = learning_rate * neuron["change"] * hidden_layer_input[j]
                neuron["weights"][j] += change + 0.75 * neuron["prev_change"][j]
                neuron["prev_change"][j] = change
            neuron["weights"][-1] += learning_rate * neuron["change"]


# Tests the network and reports the number of incorrect classifcations
def testNetwork(network, test_data, num_output_nodes):
    incorrect = 0
    for row in test_data:
        classification = classify(network, row)
        prediction = classification.index(max(classification))
        if row[-1] != prediction:
            incorrect += 1
    return incorrect


# Outputs the network results to a text file
def logData(network, test_data, train_data, num_output_nodes):
    train_confusion_matrix = [[0 for i in range(num_output_nodes)] for i in range(num_output_nodes)]
    test_confusion_matrix = [[0 for i in range(num_output_nodes)] for i in range(num_output_nodes)]

    f = open("output.txt", "w")
    f.write("Expected\tPredicted\n")
    f.write("Training Data \n")

     # Writes the expected vs predicted outcome to file for training data
    for row in train_data:
        classification = classify(network, row)
        prediction = classification.index(max(classification))
        incrementMatrix(train_confusion_matrix, prediction, row[-1])
        f.write(str(convertToLabel(row[-1])) + "\t" + str(convertToLabel(prediction)) + "\n")

    #W Write the confusion matrix for training data to file
    for row in train_confusion_matrix:
        f.write("\n" + str(row))

    # Writes the expected vs predicted outcome to file for testing data
    f.write("\nTesting Data\n")
    for row in test_data:
        classification = classify(network, row)
        prediction = classification.index(max(classification))
        incrementMatrix(test_confusion_matrix, prediction, row[-1])
        f.write(str(convertToLabel(row[-1])) + "\t" + str(convertToLabel(prediction))+ "\n")


    # Write the confusion matrix for testing data to file
    for row in test_confusion_matrix:
        f.write("\n" + str(row))

    # Write our final weight vectors to the text file
    f.write("\nHidden Layer Node Weights")
    for i in range(len(network[0])):
        f.write("\nNode " + str(i) + " weights:")
        f.write("\n" + str(network[0][i]["weights"]))

    f.write("\nOutput Layer Node Weights")
    for i in range(len(network[1])):
        f.write("\nNode " + str(i) + " weights:")
        f.write("\n" + str(network[1][i]["weights"]))
    f.close()
    return


# increment confusion matrix counts
def incrementMatrix(matrix, prediction, actual):
    matrix[prediction][actual] += 1


# Convert our binary labels back to original labels
def convertToLabel(binary_value):
    if binary_value == 0:
        return 5
    elif binary_value == 1:
        return 7
    else:
        return 8


if __name__ == "__main__":
    # Set up variable constants for our network
    ERRORS = 200
    INPUT_NODES = 11
    OUTPUT_NODES = 3

    # Read data from exel file and mutate it as desired
    data = getData("assignment2data.csv")
    convertToFloat(data)
    normalizeData(data)
    standardizeDefiningLabel(data)

    # Split our data into train/test split and train our network
    number_of_training_data = int(len(data) * 0.8)
    train_data = data[:number_of_training_data]
    test_data = data[number_of_training_data:]
    network = setupNetwork(INPUT_NODES, 25, OUTPUT_NODES)
    trainNetwork(network, train_data, 0.5, 500, OUTPUT_NODES)

    # Calculate our desired error
    goal_error = (len(data) - number_of_training_data) * 0.20

    # Continuously loop until goal results are reached
    while ERRORS > goal_error:
        trainNetwork(network, train_data, 0.5, 10, OUTPUT_NODES)
        ERRORS = testNetwork(network, test_data, OUTPUT_NODES)

    # log the data to our textfiel
    logData(network, test_data, train_data, OUTPUT_NODES)
