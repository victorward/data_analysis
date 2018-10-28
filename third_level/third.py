import pandas
import numpy as np


def read_data():
    global data
    data = pandas.read_csv(
        '../data/footballers.csv',
    )


if __name__ == "__main__":
    read_data()
