import numpy as np
import math
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier


# performs boostrapping subset of data, with specific hyperparameters
def bootstrap(B,X,y, C_dict, Classifier):
  n = len(X)
  bs_err = np.zeros(B)
  # perform boostrapping B times
  for b in range(B):
    # sample training set, use non-sampled data as test set
    train_samples = list(np.random.randint(0,n,n))
    test_samples = list(set(range(n)) - set(train_samples))
    # fit classifier with name as the key in c_dict and value as the key's value
    alg = Classifier(**C_dict)
    alg.fit(X[train_samples], y[train_samples])
    # compute error on test sample
    bs_err[b] = np.mean(y[test_samples] != alg.predict(X[test_samples]))
  # return mean error across all boostrapped samples
  err = np.mean(bs_err)
  return err

# implements k-fold cross validation with nested boostrapping for hyperparameter tuning
def kfold(k, X, y, Classifier, arg_dict):
    hyperparam_name = list(arg_dict.keys())[0]
    B = 10
    n, d = X.shape
    z = np.zeros((k, 1))

    best_C = np.zeros(k, dtype= type(list(arg_dict.values())[0][0])) # use float or int
    best_err = np.full(k, 1.1)
    fold_err = np.zeros(k)

    for i in range(k):
        # compute trainig / test set indexes
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