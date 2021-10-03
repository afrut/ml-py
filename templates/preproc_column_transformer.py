#exec(open('.\\templates\\preproc_column_transformer.py').read())
import subprocess as sp
import importlib as il
import pickle as pk
import numpy as np
import sklearn.compose as sc
import sklearn.preprocessing as pp

import datacfg

if __name__ == '__main__':
    sp.call('cls', shell = True)
    il.reload(datacfg)

    with open(datacfg.datacfg()['adult']['filepath'], 'rb') as fl:
        df = pk.load(fl)

    # Set feature and target columns.
    ycols = set(['class'])
    xcols = set(df.columns) - ycols

    # Set numeric and non-numeric columns.
    numerics = set(df.select_dtypes([np.number]).columns)
    nonnumerics = xcols - numerics
    xcols = list(xcols)
    idxnumerics = [xcols.index(col) for col in numerics]
    idxnonnumerics = [xcols.index(col) for col in nonnumerics]

    # Designate data.
    X = df.loc[:, xcols].values
    y = df.loc[:, ycols].values

    # Apply a transformation for each column.
    transformers = list()
    transformers.append(('StandardScaler', pp.StandardScaler(), idxnumerics))
    transformers.append(('OneHotEncoder', pp.OneHotEncoder(sparse = False, drop = 'first'), idxnonnumerics))
    ct = sc.ColumnTransformer(transformers, remainder = 'passthrough')
    ct.fit(X)
    Xtransformed = ct.transform(X)
    print('Feature Names: {0}'.format(ct.get_feature_names_out()))