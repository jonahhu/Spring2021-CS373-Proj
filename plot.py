import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd


def sample_plot(x, y, title):
    cinterval = 1.95 * np.std(y)/np.mean(y)

    plt.plot(x, y, '-o')
    plt.xlabel('sample sizes')
    plt.ylabel('cross-val error')
    plt.fill_between(x, (y-cinterval), (y+cinterval), alpha=0.05)
    #plt.show()

def hyperparam_plot(title, xlab, err_dict):
    fig, ax = plt.subplots()
    ax.set_title(title)
    
    params = list(err_dict.keys())
    data = list(err_dict.values())

    ax.boxplot(data)
    ax.set_xticklabels(params)
    ax.set_ylabel('Error')
    ax.set_xlabel(xlab)

    #plt.show()

    # return incase we want to make changes after function call
    return fig,ax

# used for testing purposes
if __name__ == "__main__":
    # testing hyperparam plot
    #matplotlib.use('TkAgg')
    #plt.ioff()
    test_data = {0.1: [0.23879590879814067, 0.21828255587819517, 0.23336295833375126, 0.23124525612438757, 0.21880039855057182, 0.22149232135831323, 0.21274808702709, 0.22439998088104285, 0.2127951384813252, 0.21943764304218724], 1: [0.2417847378484875, 0.2187509616870608, 0.23008012190106536, 0.23213278992380357, 0.2314014446806329, 0.22886685756119274, 0.21190215375435956, 0.23146733255126808, 0.21574453495003493, 0.21104110080193542], 10: [0.23804803714600595, 0.21516937954155427, 0.2276390166379559, 0.2360583875095882, 0.22920916033398414, 0.2270999823200929, 0.22669965978808598, 0.22178080381315893, 0.2191448904888333, 0.22111348181901117]}
    hyperparam_plot(title='Test Plot', xlab='C', err_dict=test_data)
    hyperparam_plot(title='Test Plot', xlab='C', err_dict=test_data)
    plt.show()

    x = np.array([1, 3, 5, 7])
    y = np.array([5, 7, 3, 5])
    #sample_plot(x, y, "random")
    #plt.close()