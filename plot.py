import matplotlib.pyplot as plt


# used for generating boxplots from a dictionary, err_dict
# where the keys of the dictionary are the x-axis
# and the values for the boxplots on the y-axis
def hyperparam_plot(title, xlab, err_dict):
    fig, ax = plt.subplots()
    ax.set_title(title)

    params = list(err_dict.keys())
    data = list(err_dict.values())

    ax.boxplot(data)
    ax.set_xticklabels(params)
    ax.set_ylabel('Error')
    ax.set_xlabel(xlab)

    # return incase we want to make changes after function call
    return fig, ax
