import pandas
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

data = None


def read_data():
    global data
    data = pandas.read_csv(
        '../data/Births.csv',
    )


def avg_day_birth():
    # w_value, p_value = stats.shapiro(data['date'])
    grouped_by_wday = data.groupby('wday').sum()
    amount_of_data = data.groupby('wday').count()
    # print(grouped_by_wday['births'])
    # print(amount_of_data['date'])
    days_birth = grouped_by_wday['births']/amount_of_data['date']
    tval, pval = stats.ttest_1samp(days_birth, 10000)
    hipoteza_is_ok = pval > 0.05
    print("Hipoteza is ok - {}. Pvalue = {}".format(hipoteza_is_ok, pval))
    print(days_birth)
    print("Average day birth - {}".format(np.mean(days_birth)))
    plot(days_birth)


def plot(days_birth):
    plt.bar(days_birth.keys().tolist(), days_birth)
    plt.axhline(y=10000, color='r', linestyle='-')
    plt.xlabel('Days')
    plt.ylabel('Births')
    plt.title('Births by day')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    read_data()
    avg_day_birth()
