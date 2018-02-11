from random import random
from csv import reader


def get_data(filename):
    data = []
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            data.append(row)
    return data


print get_data("assignment2data.csv")
