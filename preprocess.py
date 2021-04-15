import tarfile
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer

# main run function for preprocessing data
def run(fname, min_df, binary):
    df = extract_data(fname)
    X = bag_of_words(df, min_df, binary)
    y = df['Label']
    return X, y

# extracts the raw data from public data set and consolidates it
# into one data set
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

# runs the bag of words function to get a table of binary values
# that tells us about the occurrence of each word
def bag_of_words(df, min_df, binary):
    cv = CountVectorizer(min_df=min_df, binary=binary)
    X = cv.fit_transform(df['Content'].tolist()).toarray()

    return X