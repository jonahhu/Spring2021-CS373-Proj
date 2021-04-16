import numpy as np
import math
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier


def bootstrap(B,X_subset,y_subset, C_dict, Classifier):
  n = len(X_subset)
  bs_err = np.zeros(B)
  for b in range(B):
    train_samples = list(np.random.randint(0,n,n))
    test_samples = list(set(range(n)) - set(train_samples))
    #alg = LinearSVC(C=C)
    #alg = KNeighborsClassifier(n_neighbors=C)
    alg = Classifier(**C_dict)
    alg.fit(X_subset[train_samples], y_subset[train_samples])
    bs_err[b] = np.mean(y_subset[test_samples] != alg.predict(X_subset[test_samples]))
  err = np.mean(bs_err)
  return err

def kfold(k, X, y, Classifier, arg_dict):
    #C_vals = [5, 10, 15]
    hyperparam_name = list(arg_dict.keys())[0]
    B = 10
    n, d = X.shape
    z = np.zeros((k, 1))
    best_C = np.zeros(k, dtype= type(list(arg_dict.values())[0][0])) # use float or int
    best_err = np.full(k, 1.1)
    fold_err = np.zeros(k)
    #print(best_err)
    for i in range(k):
        bot = (n * i) / k
        top = ((n * (i + 1)) / k)
        T = set([j for j in range(math.floor(bot), math.floor(top))])
        S = set([num for num in range(n)]) - T
        arr_S = np.fromiter(S, int)
        arr_T = np.fromiter(T, int)
        # create training / test partitions for this fold
        X_train = X[arr_S]
        y_train = y[arr_S]
        X_test = X[arr_T]
        y_test = y[arr_T]
        # iterate through hyperparameters, performing boostrapping on training set to determine best C
        for C in list(arg_dict.values())[0]:
            C_dict = {hyperparam_name: C}
            err = bootstrap(B, X_train, y_train, C_dict, Classifier)
            if err < best_err[i]:
                best_err[i] = err
                best_C[i] = C
        # evaluate fold error using best C, add to total err
        alg = Classifier(**{hyperparam_name: best_C[i] })
        alg.fit(X_train, y_train)
        fold_err[i] = np.mean(y_test != alg.predict(X_test))
    total_err = np.mean(fold_err)
    return best_C, best_err, fold_err, total_err