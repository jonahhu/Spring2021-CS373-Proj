from sklearn.utils import shuffle
import validation
import random
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import preprocess
import warnings
import sys
from plot import hyperparam_plot

if __name__ == "__main__":
    # turn off convergence warnings, 
    # or turn back on using -W when running program ie python3 -Wmain.py
    if not sys.warnoptions:
        warnings.simplefilter("ignore")


    # fix seed for testing purposes
    random.seed(10)

    X, y = preprocess.run("review_polarity.tar.gz", 0.05, binary=False)
    y = y.to_numpy(dtype=int)

    # shuffle data
    X,y = shuffle(X, y)

    # constants 
    # number of CV folds
    k = 10

    # test svm (perform nested k fold cross validation)
    best_C, best_err, fold_err, total_err, err_dict_svm = validation.kfold(k, X, y, LinearSVC, {"C": [.1,1,10]})
    print("SVM Results:")
    print(best_C)
    print(best_err)
    print(fold_err)
    print(total_err)
    print(err_dict)

    # test knn (perform nested k fold cross validation)
    best_C, best_err,fold_err, total_err, err_dict_knn = validation.kfold(k, X, y, KNeighborsClassifier, {"n_neighbors": [5,10,15]})
    print("KNN Results:")
    print(best_C)
    print(best_err)
    print(fold_err)
    print(total_err)
    print(err_dict)

    # plot results
    hyperparam_plot('SVM Error for different slack values (C)', 'C', err_dict_svm)



