import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import preprocess
from sklearn.feature_extraction.text import CountVectorizer


def mod_bag_of_words(df, min_df, binary):
    cv = CountVectorizer(min_df=min_df, binary=binary)
    X = cv.fit_transform(df['Content'].tolist()).toarray()

    return X, cv

if __name__ == "__main__":
    # fname = "review_polarity.tar.gz"
    # df = preprocess.extract_data(fname)

    # import pdb;pdb.set_trace()

    # count_neg = df.query('Label == -1').Content.str.split(expand=True).stack().value_counts().to_frame().reset_index()
    # count_neg[0] /= count_neg[0].sum()
    # count_neg = count_neg.sort_values(by=0, ascending=False)


    # count_pos = df.query('Label == 1').Content.str.split(expand=True).stack().value_counts().to_frame().reset_index()
    # count_pos[0] /= count_pos[0].sum()
    # count_pos = count_pos.sort_values(by=0, ascending=False)


    # merged = count_pos.merge(count_neg, how='outer', on='index').dropna()
    # merged['diff'] = np.abs(merged['0_x'] - merged['0_y'])
    # merged = merged.sort_values(by='diff', ascending=False)

    # start = 0
    # num_values = 30
    # plt.bar(count_neg['index'].values[start:num_values], count_neg[0].values[start:num_values])
    # plt.bar(count_pos['index'].values[start:num_values], count_pos[0].values[start:num_values])

    
    # #plt.bar(merged['index'].values[start:num_values], merged['diff'].values[start:num_values])
    # plt.show()

    sample_sizes = [200, 400, 600, 800, 1000]
    for size in sample_sizes:
        X, y = preprocess.run("review_polarity.tar.gz", 0.05, size, binary=False)
        df = pd.DataFrame(X)
        df['label'] = y
        df.to_csv(f'./preprocessed/data_size_{size}')
        print(df)
    
