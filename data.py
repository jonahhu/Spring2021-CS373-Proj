import pandas as pd
import preprocess
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def mod_bag_of_words(df, min_df, binary):
    cv = CountVectorizer(min_df=min_df, binary=binary)
    X = cv.fit_transform(df['Content'].tolist()).toarray()

    return X, cv


if __name__ == "__main__":

    sample_sizes = [200, 400, 600, 800, 1000]
    for size in sample_sizes:
        X, y = preprocess.run("review_polarity.tar.gz", 0.05, size, binary=False)
        df = pd.DataFrame(X)
        y = np.array(y)
        df['label'] = y
        df.to_csv(f'./preprocessed/data_size_{size}.csv')
        print(df)
