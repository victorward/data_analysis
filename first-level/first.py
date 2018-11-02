import numpy as np
import pandas
from scipy import stats
import matplotlib.pyplot as plt


def read_data():
    global data
    data = pandas.read_csv(
        '../data/abalone.data',
        header=None,
        names=[
            'Sex', 'Length', 'Diameter', 'Height', 'Whole weight',
            'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings'
        ]
    )


def mean_med_min_max():
    # print(data)
    length_min = np.min(data['Length'])
    length_max = np.max(data['Length'])
    length_median = np.median(data['Length'])
    diameter_min = np.min(data['Diameter'])
    diameter_max = np.max(data['Diameter'])
    diameter_median = np.median(data['Diameter'])
    height_min = np.min(data['Height'])
    height_max = np.max(data['Height'])
    height_median = np.median(data['Height'])
    whole_weight_min = np.min(data['Whole weight'])
    whole_weight_max = np.max(data['Whole weight'])
    whole_weight_median = np.median(data['Whole weight'])
    shucked_weight_min = np.min(data['Shucked weight'])
    shucked_weight_max = np.max(data['Shucked weight'])
    shucked_weight_median = np.median(data['Shucked weight'])
    viscera_weight_min = np.min(data['Viscera weight'])
    viscera_weight_max = np.max(data['Viscera weight'])
    viscera_weight_median = np.median(data['Viscera weight'])
    shell_weight_min = np.min(data['Shell weight'])
    shell_weight_max = np.max(data['Shell weight'])
    shell_weight_median = np.median(data['Shell weight'])
    rings_min = np.min(data['Rings'])
    rings_max = np.max(data['Rings'])
    rings_median = np.median(data['Rings'])
    sex_mode = stats.mode(data["Sex"])[0][0]

    print("1 zadanie")
    print("Length min - {}".format(length_min))
    print("Length max - {}".format(length_max))
    print("Length median - {}".format(length_median))
    print("Diameter min - {}".format(diameter_min))
    print("Diameter max - {}".format(diameter_max))
    print("Diameter median - {}".format(diameter_median))
    print("Height min - {}".format(height_min))
    print("Height max - {}".format(height_max))
    print("Height median - {}".format(height_median))
    print("Whole weight min - {}".format(whole_weight_min))
    print("Whole weight max - {}".format(whole_weight_max))
    print("Whole weight median - {}".format(whole_weight_median))
    print("Shucked weight min - {}".format(shucked_weight_min))
    print("Shucked weight max - {}".format(shucked_weight_max))
    print("Shucked weight median - {}".format(shucked_weight_median))
    print("Viscera weight min - {}".format(viscera_weight_min))
    print("Viscera weight max - {}".format(viscera_weight_max))
    print("Viscera weight median - {}".format(viscera_weight_median))
    print("Shell weight min - {}".format(shell_weight_min))
    print("Shell weight max - {}".format(shell_weight_max))
    print("Shell weight median - {}".format(shell_weight_median))
    print("Rings min - {}".format(rings_min))
    print("Rings max - {}".format(rings_max))
    print("Rings median - {}".format(rings_median))
    print("Sex mode - {}\n".format(sex_mode))


def plot_histogram():
    whole_weight_histogram = data['Whole weight']
    shucked_weight_histogram = data['Shucked weight']
    legend = ['Whole weight', 'Shucked weight']
    plt.hist([whole_weight_histogram, shucked_weight_histogram], color=['orange', 'green'])
    plt.xlabel('Weight [grams]')
    plt.ylabel('Amount')
    plt.title('Whole weight and Shucked weight')
    plt.grid()
    plt.legend(legend)
    plt.show()


if __name__ == "__main__":
    read_data()
    mean_med_min_max()
    plot_histogram()
