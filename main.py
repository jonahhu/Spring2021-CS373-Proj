from sklearn.utils import shuffle
import svm
import random
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import preprocess

if __name__ == "__main__":
    # fix seed for testing purposes
    random.seed(10)

    X, y = preprocess.run("review_polarity.tar.gz", 0.05, binary=False)
    y = y.to_numpy(dtype=int)

    # shuffle data
    X,y = shuffle(X, y)

    # constants 
    k = 10

    # test svm
    best_C, best_err, fold_err, total_err = svm.kfold(k, X, y, LinearSVC, {"C": [.1,1,10]})
    print("SVM Results:")
    print(best_C)
    print(best_err)
    print(fold_err)
    print(total_err)

    # test knn
    best_C, best_err,fold_err, total_err = svm.kfold(k, X, y, KNeighborsClassifier, {"n_neighbors": [5,10,15]})
    print("KNN Results:")
    print(best_C)
    print(best_err)
    print(fold_err)
    print(total_err)

