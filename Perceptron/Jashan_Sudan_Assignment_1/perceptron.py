# Jashandeep Sudan
# NetID: 13jss 
# Student Number: 10103816


# Function to parse input from text file into an array
def get_data_from_file(file_name):
    flower_data = []
    file = open(file_name, 'r')
    for line in file:
        temp_line = line.split(',')
        # Iterates through each comma seperated item in the line
        for i in range(len(temp_line)):
            if temp_line[i][0].isdigit():
                temp_line[i] = float(temp_line[i])
            else:
                # If the line element is a classifcation for iris, convert it to a vector
                temp_line[i] = temp_line[i].strip()
                temp_line[i] = convertStrToVector(temp_line[i])
        flower_data.append(temp_line)
    return flower_data

# Convert iris type or index, into a vector
def convertStrToVector(string):
    if string == "Iris-setosa" or string == 0:
        return [1, 0, 0]
    elif string == "Iris-versicolor" or string == 1:
        return [0, 1, 0]
    else:
        return [0, 0, 1]

# Converts vector into iris type
def convertVectorToStr(vector):
    if vector == [1, 0, 0]:
        return "Iris-setosa"
    elif vector == [0, 1, 0]:
        return "Iris-versicolor"
    else:
        return "Iris-virginica"

# Predicts the type of iris, taking the highest dot product
def classify(weight_vectors, row_data):
    max_dot_product_index = 0
    max_dot_product = 0
    # Iterate through weight vectors looking for highest dot product
    for i in range(len(weight_vectors)):
        curr_dot_product = dotProduct(weight_vectors[i], row_data)
        if curr_dot_product > max_dot_product:
            max_dot_product = curr_dot_product
            max_dot_product_index = i
    return convertStrToVector(max_dot_product_index)

# Takes a weight vector and row data and computes the dot product of the two
def dotProduct(weight_vector, row_data):
    activation = weight_vector[0]
    for i in range(len(weight_vector) - 1):
        activation += weight_vector[i + 1] * row_data[i]
    return activation

# Iterates through the data set for the number of epochs provided, training the weight vectors
def train_classifier(train_data, epochs):
    # Sets initial learning rate and weight vectors arbitrarily
    learning_rate = 0.05
    weight_vector_1 = [1.0, 0.0, 0.0, 0.0, 0.0]
    weight_vector_2 = [2.0, 0.0, 0.0, 0.0, 0.0]
    weight_vector_3 = [3.0, 0.0, 0.0, 0.0, 0.0]
    weight_vectors = [weight_vector_1, weight_vector_2, weight_vector_3]

    # Iterate through the number of epochs
    for i in range(epochs):
        sum_error = 0
        # Iterate through each row in the training data
        for row in train_data:
            # Get the predicted classification
            classification = classify(weight_vectors, row)

            # Check if predicted classification equals expected
            if classification != row[-1]:
                predicted, expected = get_indexes(classification, row[-1])

                error = row[-1][predicted] + classification[predicted]
                learning_error = error * learning_rate

                # Decrease the incorrectly predicted weight vector
                change_weight_vector(weight_vectors[predicted], row, -learning_error)
                # Increase the expected weight vector
                change_weight_vector(weight_vectors[expected], row, learning_error)

            sum_error += error**2
    return weight_vectors

# Finds the index of the vector prediction
def get_indexes(classification, row):
    predicted, expected = 0, 0
    for i in range(len(classification)):
        if classification[i] == 1:
            predicted = i
        if row[i] == 1:
            expected = i
    return predicted, expected

# Either decreases or increases the weight vector, based on the error learning rate
def change_weight_vector(weight_vector, row, learning_rate):
    # Adjust bias
    weight_vector[0] += weight_vector[0] * learning_rate
    for i in range(len(weight_vector) - 1):
        # Adjust weight vector
        weight_vector[i + 1] += row[i] * learning_rate

# Tests the classifier agains the test training set
# Also outputs both the classification of the training data set and testing data set
def test_classifier(test_data, train_data, weight_vectors):
    f = open("output.txt", "w")
    f.write("Prediction   Expected\n")
    f.write("Training data predictions\n")
    # Output results from training data
    for row in train_data:
        predictediction = classify(weight_vectors, row)
        f.write(convertVectorToStr(predictediction) + "  " + convertVectorToStr(row[-1]) + "\n")
    f.write("Testing data predictions\n")
    # Output results from test data
    for row in test_data:
        predictediction = classify(weight_vectors, row)
        f.write(convertVectorToStr(predictediction) + "  " + convertVectorToStr(row[-1]) + "\n")
    return

# Function calls to parse data, train then test
train_data = get_data_from_file('train.txt')
weight_vectors = train_classifier(train_data, 200)
test_data = get_data_from_file('test.txt')
test_classifier(test_data, train_data, weight_vectors)
