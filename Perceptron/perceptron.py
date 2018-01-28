def get_data(file_name):
    flower_data = []
    file = open(file_name, 'r')
    for line in file:
        temp_line = line.split(',')
        for i in range(len(temp_line)):
            if temp_line[i][0].isdigit():
                temp_line[i] = float(temp_line[i])
            else:
                temp_line[i] = temp_line[i].strip()
                temp_line[i] = convertStrToVector(temp_line[i])
        flower_data.append(temp_line)
    return flower_data


def convertStrToVector(string):
    if string == "Iris-setosa":
        return [1, 0, 0]
    elif string == "Iris-versicolor":
        return [0, 1, 0]
    else:
        return [0, 0, 1]


def convertVectorToStr(vector):
    if vector == [1, 0, 0]:
        return "Iris-setosa"
    elif vector == [0, 1, 0]:
        return "Iris-versicolor"
    else:
        return "Iris-virginica"


def classify(weight_vector, row_data):
    activation = weight_vector[0]
    for i in range(1, len(row_data)):
        activation += weight_vector[i] * row_data[i - 1]

    return 1 if activation >= 0 else 0


def train_classifier(train_data, epochs):
    learning_rate = 0.05
    weight_vector_1 = [1.0, 0.0, 0.0, 0.0, 0.0]
    weight_vector_2 = [2.0, 0.0, 0.0, 0.0, 0.0]
    weight_vector_3 = [3.0, 0.0, 0.0, 0.0, 0.0]

    for i in range(epochs):
        sum_error = 0
        for row in train_data:
            classification_1 = classify(weight_vector_1, row)
            classification_2 = classify(weight_vector_2, row)
            classification_3 = classify(weight_vector_3, row)

            error_1 = row[-1][0] - classification_1
            error_2 = row[-1][1] - classification_2
            error_3 = row[-1][2] - classification_3
            sum_error += error_1**2 + error_2**2 + error_3**2

            weight_vector_1[0] = weight_vector_1[0] + learning_rate * error_1
            weight_vector_2[0] = weight_vector_2[0] + learning_rate * error_2
            weight_vector_3[0] = weight_vector_3[0] + learning_rate * error_3

            for i in range(1, len(row)):
                weight_vector_1[i] = weight_vector_1[i] + learning_rate * error_1 * row[i - 1]
                weight_vector_2[i] = weight_vector_2[i] + learning_rate * error_2 * row[i - 1]
                weight_vector_3[i] = weight_vector_3[i] + learning_rate * error_3 * row[i - 1]
        #print sum_error
    print weight_vector_1, weight_vector_2, weight_vector_3
    return weight_vector_1, weight_vector_2, weight_vector_3


def test_classifier(test_data, weight_vector_1, weight_vector_2, weight_vector_3):
    for row in test_data:
        classification_1 = classify(weight_vector_1, row)
        classification_2 = classify(weight_vector_2, row)
        classification_3 = classify(weight_vector_3, row)

        print "Predicted:", [classification_1, classification_2, classification_3], "Actual:", row[-1]


train_data = get_data('train.txt')
weight_vector_1, weight_vector_2, weight_vector_3 = train_classifier(train_data, 100)
test_data = get_data('test.txt')
test_classifier(test_data, weight_vector_1, weight_vector_2, weight_vector_3)


