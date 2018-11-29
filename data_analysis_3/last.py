import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2, SelectKBest
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle
from scipy.cluster.hierarchy import linkage, dendrogram

names = ['No_pregnant', 'Plasma_glucose', 'Blood_pres', 'Skin_thick',
         'Serum_insu', 'BMI', 'Diabetes_func', 'Age', 'Class']


def read_data():
    global data
    data = pandas.read_csv(
        'data/pima-indians-diabetes.data',
        header=None,
        names=names,
    )


def mlp_classification(X, y, learning_rate, epochs, plot_name, normalize_data=True):
    start = time.process_time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                        random_state=7)
    if normalize_data:
        X_train, X_test = normalize(X_train, X_test)

    mlp = MLPClassifier(hidden_layer_sizes=(100,), learning_rate_init=learning_rate)
    train_scores = []
    test_scores = []
    errors = []

    for epch in range(epochs):
        mlp.partial_fit(X_train, y_train, classes=np.unique(y_train))
        train_scores.append(mlp.score(X_train, y_train))
        test_scores.append(mlp.score(X_test, y_test))
        errors.append(mlp.loss_)

    plot_scores(train_scores, test_scores, errors, epochs, plot_name)
    print(
        "[MLP {}] | Accuracy | score for test data: {}".format(plot_name, accuracy_score(y_test, mlp.predict(X_test))))
    print("[MLP {}] Classification took {}".format(plot_name, time.process_time() - start) + "s")


def plot_scores(scores_train, scores_test, errors, epochs, name):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(scores_train, alpha=0.8, label='Train')
    plt.plot(scores_test, alpha=0.8, label='Test')
    plt.title(name + ": accuracy/epochs", fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend(loc='lower right')
    plt.subplot(2, 1, 2)
    plt.title(name + ": semilogx error for test data", fontsize=14)
    plt.semilogx(range(epochs), errors, label='Errors')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend(loc='upper right')
    plt.show()


def pca_for_all(X):
    pca = PCA(copy=True)
    pca.fit(X)
    explained_variance_ratio_ = pca.explained_variance_ratio_
    print("[All principal components] Variance: {}".format(explained_variance_ratio_))
    print("[All principal components] Sum of variances ratios: {}".format(sum(explained_variance_ratio_)))


def pca_fot_2_best(X):
    pca = PCA(n_components=2, copy=True)
    pca.fit(X)
    explained_variance_ratio_ = pca.explained_variance_ratio_
    X_transformed = pca.transform(X)
    print("[2 best principal components] Variance: {}".format(explained_variance_ratio_))
    print("[2 best principal components] Sum of variances ratios: {}".format(sum(explained_variance_ratio_)))

    return X_transformed


def pca_fot_2_worst(X):
    pca = PCA(copy=True)
    pca.fit(X)
    explained_variance_ratio_ = pca.explained_variance_ratio_
    explained_variance_ratio_ = explained_variance_ratio_[-2:]
    X_transformed = pca.transform(X)
    X_transformed = X_transformed[:, -2:]
    print("[2 worst principal components] Variance: {}".format(explained_variance_ratio_))
    print("[2 worst principal components] Sum of variances ratios: {}".format(sum(explained_variance_ratio_)))

    return X_transformed


def normalize(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def chi_2(X, y):
    k_best = SelectKBest(score_func=chi2, k=2)
    fit = k_best.fit(X, y)
    features = fit.transform(X)

    return features


def tsne_clustering(X, y, name):
    model = TSNE(learning_rate=100)
    transformed = model.fit_transform(X)
    x_axis = transformed[:, 0]
    y_axis = transformed[:, 1]

    plt.scatter(x_axis, y_axis, c=y, alpha=0.7)
    plt.colorbar()
    plt.title("[TSNE clustering] {}".format(name), fontsize=14)
    plt.show()


def means_hift(X, y, name):
    bandwidth = estimate_bandwidth(X)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    # ms = MeanShift()
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("[Meanshift] number of estimated clusters : %d" % n_clusters_)

    plt.figure(1)
    plt.clf()

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
    plt.title('[' + str(name) + '] Estimated number of clusters: %d' % n_clusters_)
    plt.show()


def start():
    X = np.array(data.drop(['Class'], axis=1))
    y = np.array(data['Class'])
    # MLP
    mlp_classification(X, y, 0.001, 1000, "Only normalized")
    tsne_clustering(X, y, "Only normalized")
    means_hift(X, y, "Only normalized")
    # PCA all
    pca_for_all(X)
    # PCA two best
    X_after_pca_for_two_best = pca_fot_2_best(X)
    mlp_classification(X_after_pca_for_two_best, y, 0.001, 1000, "For 2 best principals")
    tsne_clustering(X_after_pca_for_two_best, y, "For 2 best principals")
    means_hift(X_after_pca_for_two_best, y, "For 2 best principals")
    # PCA two worst
    X_after_pca_for_two_worst = pca_fot_2_worst(X)
    mlp_classification(X_after_pca_for_two_worst, y, 0.001, 1000, "For 2 worst principals")
    tsne_clustering(X_after_pca_for_two_worst, y, "For 2 worst principals")
    means_hift(X_after_pca_for_two_worst, y, "For 2 worst principals")
    # Chi 2
    X_after_chi_2 = chi_2(X, y)
    mlp_classification(X_after_chi_2, y, 0.001, 1000, "For chi 2")
    tsne_clustering(X_after_chi_2, y, "For chi 2")
    means_hift(X_after_chi_2, y, "For chi 2")


if __name__ == "__main__":
    read_data()
    start()
