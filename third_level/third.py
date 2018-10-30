import pandas
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


data = None
versicolour = None
virginica = None


def read_data():
    global data, versicolour, virginica

    data = pandas.read_csv(
        '../data/iris.data',
        header=None,
        names=[
            'sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'
        ]
    )
    versicolour = data[(data['class'] == 'Iris-versicolor')]
    virginica = data[(data['class'] == 'Iris-virginica')]


def check_weight_parameter():
    W, p = stats.shapiro(data['sepal-width'])
    print("Data is normilized - {}".format(p > 0.05))
    makeLeveneTest()
    visualizeSpeciesInHistogram()
    visualizeSpeciesInQPlot()
    makeShapiroWilkTestForSpecies()
    makeIndependentTTest()
    showHistogram()
    showGaussApproximation()


def makeLeveneTest():
    print("Levene Test")
    print(stats.levene(virginica['sepal-width'], versicolour['sepal-width']))


def visualizeSpeciesInHistogram():
    virginica['sepal-width'].plot(kind="hist", title="Virginica Sepal Width")
    plt.xlabel("Length (units)")
    plt.savefig('Virginica_sepal_width')
    plt.show()

    versicolour['sepal-width'].plot(kind="hist", title="Versicolor Sepal Width", color="red")
    plt.xlabel("Length (units)")
    plt.savefig('Versicolor_sepal_width')
    plt.show()


def visualizeSpeciesInQPlot():
    stats.probplot(virginica['sepal-width'], dist="norm", plot=plt)
    plt.title("Virginica Sepal Width Q-Q Plot")
    plt.savefig("Virginica_qqplot.png")
    plt.show()

    stats.probplot(versicolour['sepal-width'], dist="norm", plot=plt)
    plt.title("Versicolor Sepal Width Q-Q Plot")
    plt.savefig("versicolor_qqplot.png")
    plt.show()


def makeShapiroWilkTestForSpecies():
    print('Virginical: ')
    print(stats.shapiro(virginica['sepal-width']))
    print('Versicolour: ')
    print(stats.shapiro(versicolour['sepal-width']))


def makeIndependentTTest():
    result = stats.ttest_ind(virginica['sepal-width'], versicolour['sepal-width']);
    print(result[1] * 100)
    if (result[1] <= 0.05):
        print("Yes we can use this parameter")
    else:
        print("No, we can't use this parameter")


def showHistogram():
    data.hist()
    plt.show()


def showGaussApproximation():
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']
    own_data = data[names[0]]
    x = np.linspace(0, 149, 150)
    y = own_data
    popt, pcov = curve_fit(gaussFunc, x, own_data, maxfev=1000000)

    plt.clf()
    plt.plot(x, y, 'g-', label='Data')
    plt.plot(x, gaussFunc(x, *popt), 'r-', label='Gauss distribution')
    plt.title(names[0])
    plt.legend(loc='best')
    plt.show()


def gaussFunc(x, a, b, c):
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2))


if __name__ == "__main__":
    read_data()
    check_weight_parameter()
