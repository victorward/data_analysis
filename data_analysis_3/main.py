import random
import pandas
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
import statistics


def read_data():
    # http://archive.ics.uci.edu/ml/datasets/Auto+MPG
    global data
    data = pandas.read_csv(
        'data/auto-mpg.csv',
        header=None,
        names=['mpg', 'cylinders', 'engdispl', 'horsepower', 'weight', 'accel', 'year', 'origin', 'carname'],
        na_values='?'
    )


def get_missed_data_in_percents(data_with_na, name='', show=True):
    if show:
        print("Data information for {}:".format(name))
    table_length = data_with_na.shape[0]
    if show:
        print("Table samples number - {}".format(table_length))
    missed_data = data_with_na.isnull().sum() / table_length * 100
    if show:
        print("Missed data: \n{}\n".format(missed_data))

    return missed_data


def clean_data(dataset, column=False, row=False):
    new_data = dataset.copy()
    if column:
        return new_data.drop(['horsepower'], axis=1)
    if row:
        new_data.dropna(inplace=True)

    return new_data


def show_plots(cleaned):
    sns.pairplot(cleaned)
    plt.show()
    corr = cleaned.corr()
    sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                annot=True)
    plt.show()


def create_linear_regression_model(x, y):
    model = LinearRegression()
    model.fit(x, y)
    return model


def show_regression_parameters(model, X):
    print('Estimated intercept coefficient = {}'.format(model.intercept_))
    print('Number of coefficients = {}'.format(model.coef_))
    print('Columns: {}\n'.format(X.columns))


def fill_with_regression(model, dataset, X):
    copy = dataset.copy()
    copy = copy.drop(['horsepower', 'carname'], axis=1)

    for index, row in copy.iterrows():
        if np.isnan(dataset.at[index, 'horsepower']):
            prediction = model.predict(copy.iloc[[index]])
            dataset.at[index, 'horsepower'] = prediction[0][0]

    return dataset


def fill_na_by_mean_imputation(dataset):
    mean = dataset['horsepower'].mean()
    dataset.fillna(value=mean, inplace=True)

    return dataset


def print_comparision_information(dataset, isDataset=True, name=''):
    print("Comparision information for {}".format(name))
    if isDataset:
        copy = dataset.copy()
        without_string = copy.drop(['carname'], axis=1)
        for column in without_string.columns:
            print("Standard Deviation for column {} is {} s".format(column, statistics.stdev(without_string[column])))
            print("Mean for column {} is {} s".format(column, statistics.mean(without_string[column])))
            print("Quartiles for column {} is:\n {} s\n".format(column,
                                                                without_string[column].quantile([0.25, 0.5, 0.75])))
    else:
        print("Standard Deviation is {} s".format(statistics.stdev(dataset)))
        print("Mean is {} s".format(statistics.mean(dataset)))
        print("Quartiles is:\n{} s\n".format(dataset.quantile([0.25, 0.5, 0.75])))


def plot_linear_regression(model, x, y, dataset, name):
    plt.scatter(dataset['horsepower'], model.predict(x))
    plt.xlabel('Horsepower')
    plt.ylabel('Predicated horsepower')
    plt.title('Predicated horsepower for {}'.format(name))
    plt.show()


def linear_regression(dataset, name):
    print("Linear regression data for {}: ".format(name))
    y = dataset[['horsepower']]
    x = dataset.drop(['horsepower', 'carname'], axis=1)
    model = create_linear_regression_model(x, y)
    show_regression_parameters(model, x)
    plot_linear_regression(model, x, y, dataset, name)

    return model, x, y


def fill_with_hot_deck(dateset, cleaned):
    dateset_without_string = dateset.drop(['carname'], axis=1)
    cleaned_without_string = cleaned.drop(['carname'], axis=1)

    for index, row in dateset_without_string.iterrows():
        max_same_values = 0
        predicted_horsepower = 0
        for index2, row2 in cleaned_without_string.iterrows():
            same_values = 0

            for column in dateset_without_string:
                if column != 'horsepower':
                    diff = row2[column] - row[column]
                    if diff == 0:
                        same_values += 1

            if same_values > max_same_values:
                max_same_values = same_values
                predicted_horsepower = row2['horsepower']

        data.at[index, 'horsepower'] = predicted_horsepower

    return data


def fill_with_interpolation(dataset):
    return dataset.interpolate()


def add_missing_data(dataset, percentage):
    while True:
        dataset.at[random.randint(0, dataset.shape[0]), 'horsepower'] = np.NaN
        missed = get_missed_data_in_percents(dataset, show=False)
        if missed['horsepower'] >= percentage:
            break

    return dataset


def fill_data_with_regression_by_missed_data(dataset, percentage, ):
    missed = add_missing_data(dataset, percentage)
    # get_missed_data_in_percents(missed)
    print_comparision_information(missed['horsepower'], isDataset=False,
                                  name='Linear Regression, missed ~{}% before'.format(percentage))
    cleaned = clean_data(dataset, row=True)
    model, x, y = linear_regression(cleaned.copy(),
                                    'Linear Regression, missed ~{}%, before prediction'.format(percentage))
    predicted = fill_with_regression(model, dataset.copy(), x)
    print_comparision_information(predicted['horsepower'], isDataset=False,
                                  name='Linear Regression, missed ~{}%, after prediction'.format(percentage))
    linear_regression(predicted.copy(), 'Linear Regression, missed ~{}%, after prediction'.format(percentage))


def start():
    get_missed_data_in_percents(data, 'input data')
    cleaned = clean_data(data, row=True)
    get_missed_data_in_percents(cleaned, 'cleaned data')
    # show_plots(cleaned)

    # linear regression
    linear_regression(cleaned.copy(), 'cleaned data')

    # mean
    mean = fill_na_by_mean_imputation(data.copy())
    # print_missed_data_in_percents(mean, 'filled by mean')
    # print_comparision_information(mean)
    print_comparision_information(mean['horsepower'], isDataset=False, name='mean')
    linear_regression(mean, 'mean data')

    # predict horsepower
    model, x, y = linear_regression(cleaned.copy(), 'before prediction')
    predicted = fill_with_regression(model, data.copy(), x)
    print_comparision_information(predicted['horsepower'], isDataset=False, name='regression')
    # print_missed_data_in_percents(predicted, 'filled by linear regression')
    linear_regression(predicted.copy(), 'after prediction')

    # hot-deck
    # hot_deck = fill_with_hot_deck(data.copy(), cleaned.copy())
    # print_comparision_information(hot_deck['horsepower'], isDataset=False, name='hot deck')
    # linear_regression(hot_deck.copy(), 'after hot_deck')

    # interpolation
    interpolation = fill_with_interpolation(data.copy())
    print_comparision_information(interpolation['horsepower'], isDataset=False, name='interpolation')
    # print_missed_data_in_percents(interpolation, 'filled by interpolation')
    linear_regression(interpolation.copy(), 'after interpolation')

    # missing data
    fill_data_with_regression_by_missed_data(data.copy(), 15)
    fill_data_with_regression_by_missed_data(data.copy(), 30)
    fill_data_with_regression_by_missed_data(data.copy(), 45)


if __name__ == "__main__":
    read_data()
    start()
    print("Finish")
