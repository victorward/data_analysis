import pandas
import numpy as np
from sklearn import linear_model as linm
import seaborn as sns
import matplotlib.pyplot as plt


def read_data():
    # http://archive.ics.uci.edu/ml/datasets/Auto+MPG
    global data
    data = pandas.read_csv(
        'data/auto-mpg.csv',
        header=None,
        names=['mpg', 'cylinders', 'engdispl', 'horsepower', 'weight', 'accel', 'year', 'origin', 'carname'],
        na_values='?'
    )


def print_missed_data_in_percents():
    table_length = data.shape[0]
    print("Table samples number - {}".format(table_length))
    missed_data = data.isnull().sum() / table_length * 100
    print("Missed data: \n{}".format(missed_data))


def clean_data():
    return data.drop(['horsepower'], axis=1)


def show_plots(cleaned):
    sns.pairplot(cleaned)
    plt.show()
    corr = cleaned.corr()
    sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                annot=True
                )
    plt.show()


def lin_regression(cleaned):
    # show_plots(cleaned)
    Y = cleaned['mpg']
    X = cleaned.drop(['mpg', 'carname'], axis=1)
    lm = linm.LinearRegression()
    lm.fit(X, Y)
    ar = lm.coef_[1]
    br = lm.intercept_
    Y_pred = lm.predict(X)
    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.show()
    print("ar {}, br {}".format(ar, br))


def start():
    # print_missed_data_in_percents()
    cleaned = clean_data()
    lin_regression(cleaned)


if __name__ == "__main__":
    read_data()
    start()
