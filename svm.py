import numpy as np
import math
from sklearn.svm import LinearSVC

def bootstrap(B,X,y,C):
    n, d = X.shape
    z = np.zeros((B, 1))
    for i in range(B):
        u = np.zeros(n, dtype=int)
        S = set()
        for j in range(n):
            k = np.random.randint(0, n)
            u[j] = k
            S = S.union({k})
        T = set([num for num in range(n)]).difference(S)
        X_train = X[u]
        y_train = y[u]
        alg = LinearSVC(C=C)
        alg.fit(X_train, y_train)
        z[i] = np.mean(y[list(T)] != alg.predict(X[list(T)]))
    return np.mean(z)

def kfold(k, X, y):
    C_vals = [0.1, 1.0, 10.0]
    B = 10
    n, d = X.shape
    z = np.zeros((k, 1))
    best_C = np.zeros(k)
    best_err = np.zeros(k)
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
                best_C = C
    return C, err