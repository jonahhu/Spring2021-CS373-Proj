import tarfile
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer


def run(fname, min_df):
    df = extract_data(fname)
    X = bag_of_words(df, min_df)
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


def bag_of_words(df, min_df):
    cv = CountVectorizer(min_df=min_df, binary=True)
    X = cv.fit_transform(df['Content'].tolist()).toarray()

    return X


X, y = run("review_polarity.tar.gz", 0.01)
print(X)
print(y)
