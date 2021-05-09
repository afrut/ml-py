#exec(open('templates\\featsel_rfe.py').read())
# (R)ecursive (F)eature (E)limination using logistic regression
import subprocess as sp
import pandas as pd
import sklearn.feature_selection as sf
import sklearn.linear_model as sl
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
    y = np.ravel(df.loc[:, ycols].values)

    # specify the model to use and perform rfe to get the 2 best features
    k = 2
    model = sl.LogisticRegression()
    selector = sf.RFE(model, n_features_to_select = k).fit(X, y)

    # find the k-best feature names
    print('{0}-best selected features:'.format(selector.n_features_))
    bestFeatures = np.array(xcols)[selector.support_]
    for cnt in range(len(bestFeatures)):
        print('    ' + bestFeatures[cnt])
    print('')

    # ranking of the features
    print('Feature rankings:')
    srs = pd.Series(selector.ranking_, index = xcols)
    for tpl in srs.sort_values().items():
        print('    {0} - {1}'.format(tpl[1], tpl[0]))