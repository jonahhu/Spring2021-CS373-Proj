from sklearn.utils import shuffle
import validation
import random
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import preprocess
import warnings
import sys
from plot import hyperparam_plot

def run_single(size):
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    # fix seed for testing purposes
    random.seed(10)

    X, y = preprocess.run("review_polarity.tar.gz", 0.05, size, binary=False)
    y = y.to_numpy(dtype=int)

    # shuffle data
    X, y = shuffle(X, y)

    # constants
    # number of CV folds
    k = 10

    # test svm (perform nested k fold cross validation)
    best_C, best_err, fold_err, total_err, err_dict_svm = validation.kfold(k, X, y, LinearSVC, {"C": [.1, 1, 10]})
    print("SVM Results:")
    print(err_dict_svm)

    # hyperparam_plot('SVM Error for different slack values (C)', 'C', err_dict_svm)


    print("TEST: ")
    print(err_dict_svm)

    # hyperparam_plot('SVM Error for different slack values (C)', 'C', curr_err)
    return err_dict_svm

def run():
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    # fix seed for testing purposes
    random.seed(10)

    err_C_01 = {}
    err_C_1 = {}
    err_C_10 = {}

    size = 0
    for i in range(5):
        size += 200
        err_dict_svm = run_single(size)

        for k, v in err_dict_svm.items():
            print(k)
            print(v)

            if k == 0.1:

                err_C_01[size] = v
            elif k == 1:

                err_C_1[size] = v
            else:

                err_C_10[size] = v



    #err_C_01 = [item for sublist in err_C_01 for item in sublist]
    #err_C_1 = [item for sublist in err_C_1 for item in sublist]
    #err_C_10 = [item for sublist in err_C_10 for item in sublist]

    print("FINAL: ")
    print(err_C_01)
    print(err_C_1)
    print(err_C_10)

    hyperparam_plot('SVM Error for slack value 0.1', 'C', err_C_01)
    hyperparam_plot('SVM Error for slack value 1', 'C', err_C_1)
    hyperparam_plot('SVM Error for slack value 10', 'C', err_C_10)




def run_multiple():
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    # fix seed for testing purposes
    random.seed(10)

    all_errs = {}

    size_200 = {}
    size_400 = {}
    size_600 = {}
    size_800 = {}
    size_1000 = {}


    size = 0
    for i in range(5):
        size += 200

        X, y = preprocess.run("review_polarity.tar.gz", 0.05, size, binary=False)

        y = y.to_numpy(dtype=int)

        # shuffle data
        X, y = shuffle(X, y)

        # constants
        # number of CV folds
        k = 10

        # test svm (perform nested k fold cross validation)
        best_C, best_err, fold_err, total_err, err_dict_svm = validation.kfold(k, X, y, LinearSVC, {"C": [.1, 1, 10]})
        print("SVM Results:")
        print(best_C)
        print(best_err)
        print(fold_err)
        print(total_err)
        print(err_dict_svm)

        #hyperparam_plot('SVM Error for different slack values (C)', 'C', err_dict_svm)

        for k, v in err_dict_svm.items():
            print(k, v)
            if k not in all_errs:
              all_errs[k] = [v]
            else:
              #all_errs[k] = all_errs[k].append(v)
              all_errs[k].append(v)

    print(all_errs)

    for k, v in all_errs.items():
        flat_list = []
        for sublist in v:
            for item in sublist:
                flat_list.append(item)

        all_errs[k] = flat_list


    print("TEST: ")
    print(all_errs)

    hyperparam_plot('SVM Error for different slack values (C)', 'C', all_errs)

run()
