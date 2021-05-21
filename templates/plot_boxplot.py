#exec(open('.\\templates\\plot_boxplot.py').read())
import subprocess as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

if __name__ == '__main__':
    sp.call('cls', shell = True)

    # close all figures
    plt.close('all')

    with open('.\\data\\iris\\iris.data', 'rt') as fl:
        df = pd.read_csv(fl
            ,names = ['sepal_length','sepal_width','petal_length','petal_width','class']
            ,header = None
            ,index_col = False)

    # create a new figure
    fig = plt.figure(figsize = (14.4, 9))

    # add subplot to the figure
    ax = fig.add_subplot(1, 1, 1)

    # create the boxplot with seaborn
    ax = sns.boxplot(data = df)

    # title of the plot
    ax.set_title('Iris Dataset Boxplot')

    # save the plot as a file
    fig.savefig('.\\iris_boxplot.png', format = 'png')
    os.remove('.\\iris_boxplot.png')

    # show the plot
    plt.show()
    plt.close('all')