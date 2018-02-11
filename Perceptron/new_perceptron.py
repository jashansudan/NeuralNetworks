# CISC 452
# Daniel Lawrence
# 10085513
# 12dl38

def readDataset(filename):
    dataset = []
    with open(filename) as f:
        dataset = f.readlines()
        for i in range(len(dataset)):
            dataset[i] = dataset[i].strip().split(',')
            for j in range(4):
                dataset[i][j] = float(dataset[i][j])
            dataset[i][4] = irisToVector(dataset[i][4])

    return dataset

def irisToVector(iris):
    if iris == "Iris-setosa" or iris == 0:
        return (1.0, 0.0, 0.0)
    if iris == "Iris-versicolor" or iris == 1:
        return (0.0, 1.0, 0.0)
    if iris == "Iris-virginica" or iris == 2:
        return (0.0, 0.0, 1.0)

# take the dot product of the weight vector and the row vector
def dot(row, weight):
    activation = weight[0]
    for i in range(len(row) - 1):
        activation += weight[i + 1] * row[i]
    return activation

# classify the iris based on the weight vector
def classify(row, weights):
    best = 0
    for i in range(1, len(weights)):
        if dot(row, weights[i]) > dot(row, weights[best]):
            best = i
    return irisToVector(best)

# Estimate perceptron weights using stochastic gradient descent
def trainWeights(dataset, learningRate, epoch):
    # initial weights
    # classifies the dataset as {Iris-setosa, Iris-virginica OR Iris-versicolor}
    weights1 = [1.0, 0.0, 0.0, 0.0, 0.0]
    # classifies the dataset as {Iris-versicolor, Iris-virginica OR Iris-setosa}
    weights2 = [2.0, 0.0, 0.0, 0.0, 0.0]
    # classifies the dataset as {Iris-virginica, Iris-setosa OR Iris-versicolor}
    weights3 = [3.0, 0.0, 0.0, 0.0, 0.0]
    weights = [weights1, weights2, weights3]

    for e in range(epoch):
        sumError = 0.0
        for row in dataset:
            prediction = classify(row, weights)
            expected = row[-1]
            predVal = 0
            expectVal = 0

            for i in range(len(prediction)):
                if prediction[i] == 1.0:
                    predVal = i
                if expected[i] == 1.0:
                    expectVal = i

            # if the prediction doesn't match the expected value
            error = 0
            if predVal != expectVal:
                punish = weights[predVal]
                reward = weights[expectVal]
                error = abs(row[-1][predVal]) + abs(prediction[predVal])

                punish[0] = punish[0] - learningRate * error
                reward[0] = reward[0] + learningRate * error
                for i in range(len(row) - 1):
                    punish[i + 1] = punish[i + 1] - learningRate * error * row[i]
                    reward[i + 1] = reward[i + 1] + learningRate * error * row[i]

            sumError += error ** 2

        #print('epoch=%d, lrate=%.3f, error=%.3f' % (e, learningRate, sumError))

    return weights


def main():
    train = readDataset("train.txt")
    test = readDataset("test.txt")
    weights = trainWeights(train, 0.1, 500)
    print(weights)
    for row in test:
        prediction = classify(row, weights)
        print("Expected: " + str(row[4]) + " Predicted: " + str(prediction))

main()