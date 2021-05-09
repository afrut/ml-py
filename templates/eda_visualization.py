#exec(open('templates\\eda_visualization.py').read())
import subprocess as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == '__main__':
    sp.call('cls', shell = True)
    plt.close('all')

    # load some data
    with open('.\\data\\iris\\iris.data', 'rt') as fl:
        df = pd.read_csv(fl
            ,names = ['sepal_length','sepal_width','petal_length','petal_width','class']
            ,header = None
            ,index_col = False)

    # get numeric types
    numerics = df.select_dtypes([np.number]).columns.to_numpy()

    # get non-numeric types
    nonnum = list(set(df.columns) - set(numerics))
    nonnum = np.array(nonnum)

    # determine layout of univariate plots
    numrows = math.sqrt(len(numerics))
    if numrows - int(numrows) > 0:
        numcols = numrows + 1
    else:
        numcols = int(numrows)
    numrows = int(numrows)
    layout = (numrows, numcols)

    # matrix of histograms
    ret = df.hist()

    # matrix of probability density functions
    ret = df.plot(kind = 'density', subplots = True, layout = layout, sharex = False)
    
    # box and whisker plot for all numeric quantities
    ret = df.plot(kind = 'box', subplots = True, layout = layout, sharex = False, sharey = False)
    
    # matrix/heatmap of correlations
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    dfnumeric = df.loc[:, df.select_dtypes([np.number]).columns]

    # plot the matrix of correlations, returns the colorbar image
    axesimage = ax.matshow(dfnumeric.corr(), vmin = -1, vmax = 1)

    # plot the colorbar image
    fig.colorbar(axesimage)

    # set x and y-axis ticks and ticklabels
    ticks = np.arange(0, len(dfnumeric.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(dfnumeric.columns)
    ax.set_yticklabels(dfnumeric.columns)

    plt.show()
    plt.close('all')