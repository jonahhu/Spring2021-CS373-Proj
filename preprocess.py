import tarfile
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
import random


# main run function for preprocessing data
def run(fname, min_df, size, binary):
    df = extract_data(fname)

    # subset data if specified size is less than max
    if size < 2000:
        df = data_subset(df, size)

    X = bag_of_words(df, min_df, binary)
    y = df['Label']
    return X, y


# extracts the raw data from public data set and consolidates it
# into one data set
def extract_data(fname):
    # create a dataframe with enough elements to hold the data
    # 2000 is the size of the original dataset
    df = pd.DataFrame(index=list(range(0, 2000)), columns=['Content', 'Label'])
    i = 0

    # check if file has the tar.gz flag which should be present in the data set
    if fname.endswith("tar.gz"):
        # open the tar file
        tar = tarfile.open(fname, "r:gz")

        # iterate through all elements and update the dataframe as needed
        for member, tarinfo in zip(tar.getmembers(), tar):
            file = tar.extractfile(member)
            if file is not None:
                # substitute and decode binary data into utf-8
                contents = re.sub(r'\n', '', file.read().decode('utf-8'))

                # assign review a negative (-1) or positive (+1) label based on data
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


# helper function to randomly select n pos or neg values from dataset
def select_rows(df, num_values, start_bound, end_bound):
    subset = pd.DataFrame()
    elements = []
    count = 0

    # keep looping until n elements are in subset
    while count < num_values:
        # randomly select a number between start/end bound (both included)
        n = random.randint(start_bound, end_bound)

        # if current row isn't already in subset, add it
        if n not in elements:
            elements += [n]
            subset = subset.append(df.iloc[n])
            count += 1

    return subset


# subsets the original dataset into specified size
# size = # positive + # negative elements, so if you wanted a size of 1000
# you would get 500 pos. + 500 neg. reviews
def data_subset(df, size):
    # Note: df always size 2000, neg reviews 0-999, pos reviews 1000-1999

    subset = pd.DataFrame()
    # get negative reviews
    subset = subset.append(select_rows(df, size / 2, 0, 999))
    # get positive reviews
    subset = subset.append(select_rows(df, size / 2, 1000, 1999))

    return subset
