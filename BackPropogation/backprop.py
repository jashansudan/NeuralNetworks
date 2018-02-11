from random import random
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
    return


def normalizeData(data):
    for col in range(len(data[0])):
        col_max = float("-inf")
        col_min = float("inf")
        for row in range(len(data)):
            if data[row][col] > col_max:
                col_max = data[row][col]
            if data[row][col] < col_min:
                col_min = data[row][col]
        print col_max, col_min

        for row in range(len(data)):
            data[row][col] = (data[row][col] - col_min) / (col_max - col_min)
    return


data = getData("assignment2data.csv")
convertToFloat(data)
normalizeData(data)
for i in data:
    print i
