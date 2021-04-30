from sklearn.utils import shuffle
import validation
import random
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import preprocess
import warnings
import sys
from plot import hyperparam_plot
import numpy as np
import matplotlib.pyplot as plt

def create_hyperparameter_plots():
    total_err, total_err2, err_dict_svm, err_dict_knn = run_single(1000)
    hyperparam_plot('SVM Error for different slack values (C)', 'C', err_dict_svm)
    hyperparam_plot('KNN Error for different nearest-neighbor values (k)', 'k', err_dict_knn)


def create_size_plots():
    # dictionaries for different graphs of C hyperparameters and K hypyerpamaters
    err_C_01 = {}
    err_C_1 = {}
    err_C_10 = {}
    err_K_5 = {}
    err_K_10 = {}
    err_K_15 = {}

    size = 0
    total_errSVC = np.empty(5)
    total_errKNN = np.empty(5)
    sample_sizes = np.array([200, 400, 600, 800, 1000])
    # get error cross validation between sizes 200-1000 at intervals of 200
    for i in range(5):
        size += 200

        # get the svm for error dictionary
        total_errSVC[i], total_errKNN[i], err_dict_svm, err_dict_knn = run_single(size)

        # assign key, value pair for current size, svm error for each C hyperparameter
        for k, v in err_dict_svm.items():
            if k == 0.1:
                err_C_01[size] = v
            elif k == 1:
                err_C_1[size] = v
            else:
                err_C_10[size] = v

        # perform same process for KNN
        for k, v in err_dict_knn.items():
            if k == 5:
                err_K_5[size] = v
            elif k == 10:
                err_K_10[size] = v
            else:
                err_K_15[size] = v

    # print("FINAL: ")
    # print(err_C_01)
    # print(err_C_1)
    # print(err_C_10)

    # plot the graphs with different values of hyperparameters]

    hyperparam_plot('SVM Error for slack value 0.1 for different sizes', 'Size', err_C_01)
    hyperparam_plot('SVM Error for slack value 1 for different sizes', 'Size', err_C_1)
    hyperparam_plot('SVM Error for slack value 10 for different sizes', 'Size', err_C_10)
    hyperparam_plot('KNN Error for K value 5 for different sizes', 'Size', err_K_5)
    hyperparam_plot('KNN Error for K value 10 for different sizes', 'Size', err_K_10)
    hyperparam_plot('KNN Error for K value 15 for different sizes', 'Size', err_K_15)

def run_single(size):
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    # fix seed for testing purposes
    random.seed(10)

    # run the preprocessing to get the subset of data
    X, y = preprocess.run("review_polarity.tar.gz", 0.05, size, binary=False)
    y = y.to_numpy(dtype=int)

    # shuffle data
    X, y = shuffle(X, y)

    # number of CV folds
    k = 10

    # test svm (perform nested k fold cross validation)
    best_C, best_err, fold_err, total_err, err_dict_svm = validation.kfold(k, X, y, LinearSVC, {"C": [.1, 1, 10]})
    best_C2, best_err2,fold_err2, total_err2, err_dict_knn = validation.kfold(k, X, y, KNeighborsClassifier, {"n_neighbors": [5,10,15]})

    # return the svm for error dictionary
    return total_err, total_err2, err_dict_svm, err_dict_knn


if __name__ == "__main__":
    # turn off convergence warnings, 
    # or turn back on using -W when running program ie python3 -Wmain.py
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    # fix seed for testing purposes
    random.seed(10)

    # generate 1st experimental results
    create_hyperparameter_plots()
    # generate 2nd experiment results
    create_size_plots()

    plt.show()