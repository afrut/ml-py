#exec(open('eda.py').read())
import subprocess as sp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import importlib as il

import main

if __name__ == '__main__':
    sp.call('cls', shell = True)
    plt.close('all')
    il.reload(main)

    # ----------------------------------------
    # Data loading and formatting
    # ----------------------------------------
    df, dfTest = main.loadData()

    # Count number of rows with missing values.
    tplTrain = main.countNan(df)
    tplTest = main.countNan(dfTest)
    print('Training set missing values: {0}/{1} = {2:.6f}'.format(tplTrain[0], tplTrain[1], tplTrain[2]))
    print('Test set missing values: {0}/{1} = {2:.6f}'.format(tplTest[0], tplTest[1], tplTest[2]))
    print('')

    # Load data with feature engineering, imputation, and type conversions.
    df, dfTest = main.preprocData(df, dfTest)
    
    # ----------------------------------------
    # Constants
    # ----------------------------------------
    np.set_printoptions(precision = 4, suppress = True)
    pd.options.display.float_format = '{:10,.4f}'.format
    seed = 29

    # ----------------------------------------
    # Descriptive statistics
    # ----------------------------------------
    print('Number of Rows: {0}'.format(df.shape[0]))
    print('Number of Columns: {0}'.format(df.shape[1]))
    print('')

    print('Column Names:')
    for col in df.columns:
        print('    ' + col)
    print('')

    print('First 20 rows:')
    print(df.head(20))
    print('')

    print('Last 20 rows:')
    print(df.tail(20))
    print('')

    print('Data types:')
    datatypes = df.dtypes
    for idx in datatypes.index:
        print('    {0} - {1}'.format(idx, datatypes[idx]))
    print('')

    print('Statistical Summary:')
    print(df.describe())
    print('')

    print('Correlations between variables:')
    print(df.corr())
    print('')

    # numbers closer to 0 mean the distribution is closer to Gaussian
    print('Skew of variables:')
    print(df.skew())
    print('')

    # Number of unique cell names.
    unqCellNames, unqCnt = np.unique(df.loc[:, 'CellName'], return_counts = True)
    print('Number of unique CellNames: {0}'.format(len(unqCellNames)))
    print('Counts of each unique CellName: {0}'.format(unqCnt))

    # NOTE: Training set is 36815 x 14. Number of samples >> number of columns.
    # NOTE: All features are numerical except Time and CellName. Time can be
    # decomposed by replacing it with an identifier that uniquely identifies the
    # specific 15-minute chunk of the day. CellName can be decomposed into both
    # the cell number and base station. Both non-numeric features can also be
    # one-hot encoded.
    
    # ----------------------------------------
    # Descriptive plots
    # ----------------------------------------
    # get numeric types
    numerics = df.select_dtypes([np.number]).columns.to_numpy()

    # get non-numeric types
    nonnum = list(set(df.columns) - set(numerics))
    nonnum = np.array(nonnum)

    # determine layout of univariate plots
    numvar = len(numerics)
    numrows = int(math.sqrt(numvar))
    numcols = numrows
    while(numrows * numcols < numvar):
        numcols = numcols + 1
    layout = (numrows, numcols)

    # get only numeric data
    dfnumeric = df.loc[:, numerics]

    # matrix of histograms
    ret = dfnumeric.hist()

    # matrix of probability density functions
    ret = dfnumeric.plot(kind = 'density', subplots = True, layout = layout, sharex = False)
    
    # box and whisker plot for all numeric quantities
    ret = dfnumeric.plot(kind = 'box', subplots = True, layout = layout, sharex = False, sharey = False)
    
    # matrix/heatmap of correlations
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

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

    # scatter matrix plot
    ret = pd.plotting.scatter_matrix(dfnumeric)
    for cntPlot in range(len(dfnumeric.columns)):
        ylabel = ret[cntPlot][0].axes.get_ylabel()
        ret[cntPlot][0].axes.set_ylabel(ylabel, rotation = 0)

    plt.show()
    plt.close('all')