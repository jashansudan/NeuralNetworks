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
                if temp_line[i] == "Iris-setosa":
                    temp_line[i] = 1
                elif temp_line[i] == "Iris-versicolor":
                    temp_line[i] = 0
                else:
                    temp_line[i] = -1
        flower_data.append(temp_line)
    return flower_data


def train_model(train_data, epochs):
    weight_vector = [0, 0, 0, 0, 0]
    learning_rate = 0.0005
    for i in range(epochs):
        sum_error = 0
        for row in train_data:
            classifcation = classify(row, weight_vector)
            error = row[-1] - classifcation
            sum_error += error**2
            weight_vector[0] = weight_vector[0] + learning_rate * error
            for i in range(len(row) - 1):
                weight_vector[i + 1] = weight_vector[i + 1] + learning_rate * error * row[i]
    return weight_vector


def classify(row_data, weight_vector):
    activation = weight_vector[0]
    for i in range(len(row_data) - 1):
        activation += weight_vector[i + 1] * row_data[i]
    if activation > 0.333:
        return 1
    elif activation > -0.333:
        return 0
    else:
        return -1


def test_model(data, weight_vector):
    for row in data:
        classification = classify(row, weight_vector)
        print "Predicted:", get_flower(classification), "Actual:", get_flower(row[-1])

def get_flower(num):
    if num == 1:
        return "Iris-setosa"
    elif num == 0:
        return "Iris-versicolor"
    else:
        return "Iris-virginica"


train_data = get_data('train.txt')
weight_vector = train_model(train_data, 100)
test_data = get_data('test.txt')
test_model(test_data, weight_vector)




