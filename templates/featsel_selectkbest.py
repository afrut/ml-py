#exec(open('templates\\featsel_selectkbest.py').read())
import subprocess as sp
import pandas as pd
import sklearn.feature_selection as sf
import numpy as np

if __name__ == '__main__':
    sp.call('cls', shell = True)

    # load some data
    with open('.\\data\\iris\\iris.data', 'rt') as fl:
        df = pd.read_csv(fl
            ,names = ['sepal_length','sepal_width','petal_length','petal_width','class']
            ,header = None
            ,index_col = False)

    # specify the x and y matrices
    xcols = ['sepal_length','sepal_width','petal_length','petal_width']
    ycols = ['class']
    X = df.loc[:, xcols].values
    y = df.loc[:, ycols].values

    # select the 2 best features by using chi-squared tests
    # chi-squared tests are usually only for categorical target y
    k = 2
    selector = sf.SelectKBest(score_func = sf.chi2, k = k).fit(X, y)

    # display scores of every feature
    np.set_printoptions(precision = 3)
    print('Chi-squared scores:')
    for cnt in range(len(xcols)):
        print('    {0}: {1:.4f}'.format(xcols[cnt], selector.scores_[cnt]))
    print('')

    # find the k-best feature names
    srs = pd.Series(selector.scores_, index = xcols)
    bestFeatures = srs.sort_values(ascending = False)[0:2].index.values

    # extract the k-best features
    features = selector.transform(X)
    print('Best feature values:\n{0}\n'.format(features[0:5,:]))