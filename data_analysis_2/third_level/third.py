import pandas
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


parameter = 'Weight(pounds)'


def read_data():
    global data, Starting_Pitcher, Relief_Pitcher, Starting_Pitcher_Parameter, Relief_Pitcher_Parameter

    data = pandas.read_csv(
        '../data/footballers.csv',
        delimiter=";"
    )
    Starting_Pitcher = data[(data['Position'] == 'Starting_Pitcher')]
    Starting_Pitcher_Parameter = Starting_Pitcher[parameter]
    Starting_Pitcher_Parameter = Starting_Pitcher_Parameter[~np.isnan(Starting_Pitcher_Parameter)]
    Relief_Pitcher = data[(data['Position'] == 'Relief_Pitcher')]
    Relief_Pitcher_Parameter = Relief_Pitcher[parameter]


def check_weight_parameter():
    W, p = stats.shapiro(data[parameter])
    print("Data is normilized - {}".format(p > 0.05))
    make_levene_test()
    visualize_in_histogram()
    visualize_in_qplot()
    make_shapiro_wilk_test()
    # make_independent_ttest()
    make_wilcoxon_test()
    make_mannwhitney_test()
    show_histogram()
    show_gauss_approximation()


def make_levene_test():
    print(stats.levene(Relief_Pitcher_Parameter, Starting_Pitcher_Parameter))


def visualize_in_histogram():
    Relief_Pitcher_Parameter.plot(kind="hist", title="Relief_Pitcher")
    plt.xlabel(parameter)
    plt.show()

    Starting_Pitcher_Parameter.plot(kind="hist", title="Starting_Pitcher", color="red")
    plt.xlabel(parameter)
    plt.show()


def visualize_in_qplot():
    stats.probplot(Relief_Pitcher_Parameter, dist="norm", plot=plt)
    plt.title("Relief_Pitcher Q-Q Plot")
    plt.grid()
    plt.show()

    stats.probplot(Starting_Pitcher_Parameter, dist="norm", plot=plt)
    plt.title("Starting_Pitcher Q-Q Plot")
    plt.grid()
    plt.show()


def make_shapiro_wilk_test():
    print('Relief_Pitcher: ')
    print(stats.shapiro(Relief_Pitcher_Parameter))
    print('Starting_Pitcher: ')
    print(stats.shapiro(Starting_Pitcher_Parameter))


def make_independent_ttest():
    result = stats.ttest_ind(Relief_Pitcher_Parameter, Starting_Pitcher_Parameter)
    print("ttest_ind {}".format(result[1] * 100))

    if result[1] <= 0.05:
        print("Yes we can use this parameter")
    else:
        print("No, we can't use this parameter")


def make_wilcoxon_test():
    result = stats.ranksums(Relief_Pitcher_Parameter, Starting_Pitcher_Parameter)

    if result[1] <= 0.05:
        print("wilcoxon. Yes we can use this parameter")
    else:
        print("wilcoxon. No, we can't use this parameter")


def make_mannwhitney_test():
    result = stats.mannwhitneyu(Relief_Pitcher_Parameter, Starting_Pitcher_Parameter)

    if result[1] <= 0.05:
        print("mannwhitney. Yes we can use this parameter")
    else:
        print("mannwhitney. No, we can't use this parameter")


def show_histogram():
    data.hist()
    plt.show()


def show_gauss_approximation():
    own_data = data[parameter]
    own_data = own_data[~np.isnan(own_data)]
    x = np.linspace(0, len(own_data), len(own_data))
    y = own_data
    popt, pcov = curve_fit(gauss_func, x, own_data)

    plt.clf()
    plt.plot(x, y, 'g-', label='Data')
    plt.plot(x, gauss_func(x, *popt), 'r-', label='Gauss distribution')
    plt.title(parameter)
    plt.legend(loc='best')
    plt.show()


def gauss_func(x, a, b, c):
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2))


if __name__ == "__main__":
    read_data()
    check_weight_parameter()
