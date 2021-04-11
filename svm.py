import numpy as np
import math
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier


def bootstrap(B,X_subset,y_subset,C):
  n = len(X_subset)
  bs_err = np.zeros(B)
  for b in range(B):
    train_samples = list(np.random.randint(0,n,n))
    test_samples = list(set(range(n)) - set(train_samples))
    #alg = LinearSVC(C=C)
    alg = KNeighborsClassifier(n_neighbors=C)
    alg.fit(X_subset[train_samples], y_subset[train_samples])
    bs_err[b] = np.mean(y_subset[test_samples] != alg.predict(X_subset[test_samples]))
  err = np.mean(bs_err)
  return err

def kfold(k, X, y):
    C_vals = [5, 10, 15]
    B = 10
    n, d = X.shape
    z = np.zeros((k, 1))
    best_C = np.zeros(k)
    best_err = np.full(k, 1.1)
    print(best_err)
    for i in range(k):
        bot = (n * i) / k
        top = ((n * (i + 1)) / k)
        T = set([j for j in range(math.floor(bot), math.floor(top))])
        S = set([num for num in range(n)]) - T
        arr_S = np.fromiter(S, int)
        X_train = X[arr_S]
        y_train = y[arr_S]
        for C in C_vals:
            err = bootstrap(B, X_train, y_train, C)
            if err < best_err[i]:
                best_err[i] = err
                best_C[i] = C
    return best_C, best_err