import tarfile
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
import svm
import random
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier


def run(fname, min_df, binary=True):
    df = extract_data(fname)
    X = bag_of_words(df, min_df, binary)
    y = df['Label']
    return X, y

# extracts the raw data from public data set
def extract_data(fname):
    df = pd.DataFrame(index=list(range(0, 2000)), columns=['Content', 'Label'])
    i = 0

    if fname.endswith("tar.gz"):
        tar = tarfile.open(fname, "r:gz")

        for member, tarinfo in zip(tar.getmembers(), tar):
            file = tar.extractfile(member)
            if file is not None:
                contents = re.sub(r'\n', '', file.read().decode('utf-8'))

                if "neg" in tarinfo.name:
                    df.loc[i] = [contents, -1]
                elif "pos" in tarinfo.name:
                    df.loc[i] = [contents, 1]

            i = i + 1

        tar.close()

    return df


def bag_of_words(df, min_df, binary=True):
    cv = CountVectorizer(min_df=min_df, binary=binary)
    X = cv.fit_transform(df['Content'].tolist()).toarray()

    return X

if __name__ == "__main__":
    # fix seed for testing purposes
    random.seed(10)

    X, y = run("review_polarity.tar.gz", 0.05, binary=False)
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