import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# https://github.com/LamaHamadeh/Pima-Indians-Diabetes-DataSet-UCI/blob/master/prima_indians_diabetes.py
# https://github.com/mcclymont-k/diabetes_ML/blob/master/diabetes.py
# https://scikit-learn.org/stable/modules/cross_validation.html

n_nodes_hidden_layer_1 = 10
n_nodes_hidden_layer_2 = 10

n_classes = 2
batch_size = 20
names = ['No_pregnant', 'Plasma_glucose', 'Blood_pres', 'Skin_thick',
         'Serum_insu', 'BMI', 'Diabetes_func', 'Age', 'Class']


def read_data():
    global data
    data = pandas.read_csv(
        'data/pima-indians-diabetes.data',
        header=None,
        names=names,
    )


def knn_clasifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                        random_state=7)
    Knn = KNeighborsClassifier(n_neighbors=2)
    Knn.fit(X_train, y_train)
    accuracy = Knn.score(X_test, y_test)
    print('accuracy of the model is: ', accuracy)  # 0.69


def ml_classification(X, y, learning_rate, epochs):
    start = time.process_time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                        random_state=7)
    mlp = MLPClassifier(hidden_layer_sizes=(100,), learning_rate_init=learning_rate)
    scores_train = []
    scores_test = []

    for epch in range(epochs):
        mlp.partial_fit(X_train, y_train, classes=np.unique(y_train))
        scores_train.append(mlp.score(X_train, y_train))
        scores_test.append(mlp.score(X_test, y_test))

    print("Accuracy score: " + str(accuracy_score(y_test, mlp.predict(X_test))))
    print("Classification took " + str(time.process_time() - start) + "s")


def start():
    X = np.array(data.drop(['Class'], axis=1))
    y = np.array(data['Class'])
    ml_classification(X, y, 0.001, 1000)
    # knn_clasifier(X, y)


if __name__ == "__main__":
    read_data()
    start()
    print("Last Finished")
