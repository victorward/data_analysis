import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf

# https://github.com/LamaHamadeh/Pima-Indians-Diabetes-DataSet-UCI/blob/master/prima_indians_diabetes.py
# https://github.com/mcclymont-k/diabetes_ML/blob/master/diabetes.py
# https://scikit-learn.org/stable/modules/cross_validation.html

# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap


def read_data():
    global data
    data = pandas.read_csv(
        'data/pima-indians-diabetes.data',
        header=None,
        names=['No_pregnant', 'Plasma_glucose', 'Blood_pres', 'Skin_thick',
               'Serum_insu', 'BMI', 'Diabetes_func', 'Age', 'Class'],
    )


def knn_clasifier():
    X = np.array(data.drop(['Class'], axis=1))
    y = np.array(data['Class'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                        random_state=7)
    Knn = KNeighborsClassifier(n_neighbors=2)
    Knn.fit(X_train, y_train)
    accuracy = Knn.score(X_test, y_test)
    print('accuracy of the model is: ', accuracy)  # 0.69


def start():
    print(data.head())
    knn_clasifier()


if __name__ == "__main__":
    read_data()
    start()
    print("Finish Last")
