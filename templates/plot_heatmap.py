#exec(open('.\\templates\\plot_heatmap.py').read())
import subprocess as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

if __name__ == '__main__':
    sp.call('cls', shell = True)

    # close all plots
    plt.close('all')

    with open('.\\data\\iris\\iris.data', 'rt') as fl:
        df = pd.read_csv(fl
            ,names = ['sepal_length','sepal_width','petal_length','petal_width','class']
            ,header = None
            ,index_col = False)

    # get all numeric columns
    numeric = df.select_dtypes([np.number]).columns
    df = df.loc[:, numeric]

    # specify the column names to be used in rows
    rownames = numeric

    # specify the column names to be used in columns
    colnames = numeric

    # calculate correlations for the specified rownames and column names
    dfcorr = df.corr()
    dfcorr = dfcorr.loc[rownames, colnames]

    # specify a threshold correlation
    corr = 0.5

    # set all values below threshold correction to nan
    mask = dfcorr.gt(corr) | dfcorr.lt(corr * -1)
    dfcorr = dfcorr.where(mask)

    # heat map of correlations
    fig = plt.figure(figsize = (14.4, 9))
    ax = fig.add_subplot(1,1,1)

    # plot the heatmap
    ax = sns.heatmap(data = dfcorr
        ,vmin = -1      # minimum value in range
        ,vmax = 1       # maximum value in range
        ,annot = True   # label the value
        ,annot_kws = dict([('fontsize', 6)]))

    # format ticks and labels
    for ticklabel in ax.get_xticklabels():
        ticklabel.set_rotation_mode(None)
        ticklabel.set_horizontalalignment('center')
        ticklabel.set_rotation(30)
        ticklabel.set_fontsize(10)
    for ticklabel in ax.get_yticklabels():
        ticklabel.set_rotation_mode(None)
        ticklabel.set_verticalalignment('center')
        ticklabel.set_rotation(30)
        ticklabel.set_fontsize(10)

    # save the plot as a file
    plt.savefig('.\\iris_heatmap.png', format = 'png')
    os.remove('.\\iris_heatmap.png')

    # show the plot
    plt.show()
    plt.close('all')