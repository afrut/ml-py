#exec(open('templates\\proj_template_regression.py').read())
import subprocess as sp
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import math
import pandas as pd
import sklearn.model_selection as sms
import sklearn.pipeline as pipeline
import sklearn.linear_model as slm
import sklearn.neighbors as neighbors
import sklearn.svm as svm
import sklearn.ensemble as ensemble
import sklearn.tree as tree
import sklearn.preprocessing as pp
import sklearn.kernel_ridge as kr
import sklearn.gaussian_process as gp
import sklearn.cross_decomposition as cd
import sklearn.neural_network as nn

if __name__ == '__main__':
    sp.call('cls', shell = True)
    plt.close('all')

    # ----------------------------------------
    # Data loading and formatting
    # ----------------------------------------
    with open('.\\data\\bostonHousing.pkl', 'rb') as fl:
        df = pk.load(fl)
    # check that there are no missing values
    assert(np.all(np.logical_not(np.isnan(df.values)))), 'Nan values present'

    # ----------------------------------------
    # Constants
    # ----------------------------------------
    np.set_printoptions(precision = 4, suppress = True)
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
    pd.plotting.scatter_matrix(dfnumeric)

    #plt.show()
    plt.close('all')

    # ----------------------------------------
    # Specify variables and target
    # ----------------------------------------
    ycols = ['PRICE']
    xcolsnumeric = list(set(df.select_dtypes([np.number]).columns) - set(ycols))
    xcolsnonnumeric = list(set(df.select_dtypes([object]).columns) - set(ycols))
    xcols = xcolsnumeric + xcolsnonnumeric
    X = df.loc[:, xcols].values
    y = np.ravel(df.loc[:, ycols].values)
    
    # ----------------------------------------
    # Validation set
    # ----------------------------------------
    validationSize = 0.2
    Xtrain, Xvalid, ytrain, yvalid = sms.train_test_split(X, y
        ,test_size = validationSize
        ,random_state = seed)

    # ----------------------------------------
    # Try different piplines
    # - base model
    # - standardized/normalized/min-max-scaled
    # - one-hot encoding
    # - feature selection pipeline
    # - tuning pipeline
    # ----------------------------------------
    # TODO: never include SGDRegressor without scaling
    plt.close('all')
    models = dict()
    models['LR'] = (slm.LinearRegression, {})
    models['RIDGE'] = (slm.Ridge, {})
    models['LASSO'] = (slm.Lasso, {})
    #models['MTLASSO'] = (slm.MultiTaskLasso, {})
    models['EN'] = (slm.ElasticNet, {})
    #models['MTEN'] = (slm.MultiTaskElasticNet, {})
    models['LARS'] = (slm.Lars, {})
    models['LASSOLARS'] = (slm.LassoLars, {})
    models['OMP'] = (slm.OrthogonalMatchingPursuit, {})
    models['BRIDGE'] = (slm.BayesianRidge, {})
    models['TW'] = (slm.TweedieRegressor, {'max_iter': 10000})
    models['SGD'] = (slm.SGDRegressor, {}) # always standard-scale this
    models['PA'] = (slm.PassiveAggressiveRegressor, {})
    models['HUBER'] = (slm.HuberRegressor, {'max_iter': 10000})
    models['RANSAC'] = (slm.RANSACRegressor, {})
    models['TH'] = (slm.TheilSenRegressor, {})
    models['KRR'] = (kr.KernelRidge, {})
    models['GPR'] = (gp.GaussianProcessRegressor, {})
    models['PLS'] = (cd.PLSRegression, {}) # don't include this in the voting regressor
    models['KNN'] = (neighbors.KNeighborsRegressor, {})
    models['CART'] = (tree.DecisionTreeRegressor, {})
    models['SVM'] = (svm.SVR, {})
    models['AB'] = (ensemble.AdaBoostRegressor, {})
    models['GBM'] = (ensemble.GradientBoostingRegressor, {})
    models['RF'] = (ensemble.RandomForestRegressor, {})
    models['ET'] = (ensemble.ExtraTreesRegressor, {})
    models['NN'] = (nn.MLPRegressor, {'max_iter': 10000})

    # create a voting regressor out of all the regressors
    estimators = list()
    for entry in models.items():
        name = entry[0]
        model = entry[1][0]
        args = entry[1][1]
        if name != 'PLS' and name != 'SGD':
            estimators.append((name, model(**args)))
    models['VOTE'] = (ensemble.VotingRegressor, {'estimators': estimators})

    pipelines = dict()
    for entry in models.items():
        name = entry[0]
        model = entry[1][0]
        args = entry[1][1]
        if name != 'SGD':   # SGD always needs scaling
            pipelines[name] = pipeline.Pipeline([(name, model(**args))])
        pipelines['Scaled' + name] = pipeline.Pipeline([('Scaler', pp.StandardScaler()), (name, model(**args))])
    
    # specify cross-validation
    k = 10                                                                   # number of folds
    cvsplitter = sms.KFold(n_splits = k, shuffle = True, random_state = 0)   # cross-validation splitter
    foo = ensemble.VotingRegressor(estimators = [('LR', slm.LinearRegression()), ('RIDGE', slm.Ridge())])
    score = -1 * sms.cross_val_score(foo, Xtrain, ytrain, cv = cvsplitter, scoring = 'neg_mean_absolute_error')

    # fit and compute scores
    scoring = 'neg_mean_absolute_error'
    algs = list()
    scores = list()
    for entry in pipelines.items():
        score = -1 * sms.cross_val_score(entry[1], Xtrain, ytrain, cv = cvsplitter, scoring = scoring)
        scores.append(score)
        algs.append(entry[0])
        print('{0} - {1:.4f} - {2:.4f}'.format(entry[0], np.mean(score), np.std(score, ddof = 1)))

    # boxplot of results
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.boxplot(scores)
    ax.set_xticklabels(algs)
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Mean Absolute Error of Different Algorithms')

    # format every xticklabel
    for ticklabel in ax.get_xticklabels():
        ticklabel.set_horizontalalignment('right')  # center, right, left
        ticklabel.set_rotation_mode('anchor')       # None or anchor
        ticklabel.set_rotation(60)                  # angle of rotation
        ticklabel.set_fontsize(12)                  # float

    fig.tight_layout()

    # table of results
    scores = np.array(scores)
    dfScores = pd.DataFrame(index = algs)
    dfScores['mean'] = np.mean(scores, axis = 1)
    dfScores['std'] = np.std(scores, ddof = 1, axis = 1)
    print('Mean and standard deviation of MSE for different algorithms:')
    print(dfScores.sort_values(by = ['mean']))

    plt.show()
    plt.close('all')

# TODO: extend with polynomials
# TODO: add tuning pipeline
# TODO: feature selection pipeline