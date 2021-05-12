#exec(open('templates\\featsel_feature_importance.py').read())
# selection of features using ExtraTreesClassifier and RandomForestClassifier
import subprocess as sp
import pandas as pd
import sklearn.ensemble as se
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

    model = se.ExtraTreesClassifier(random_state = 0)
    model.fit(X, y)
    print('ExtraTreesClassifier feature importance rankings:')
    srs = pd.Series(model.feature_importances_, index = xcols)
    srs.sort_values(ascending = False, inplace = True)
    for tpl in srs.items():
        print('    {0}: {1:.4f}'.format(tpl[0], tpl[1]))
    print('')

    model = se.RandomForestClassifier(random_state = 0)
    model.fit(X, y)
    print('RandomForestClassifier feature importance rankings:')
    srs = pd.Series(model.feature_importances_, index = xcols)
    srs.sort_values(ascending = False, inplace = True)
    for tpl in srs.items():
        print('    {0}: {1:.4f}'.format(tpl[0], tpl[1]))