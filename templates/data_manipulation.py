#exec(open('.\\templates\\data_manipulation.py').read())
# TODO: np.squeeze
# TODO: np.ravel
# TODO: np.newaxis
# TODO: np.concatenate
# TODO: np.stack
# TODO: np.hstack
# TODO: np.vstack
# TODO: mask
# TODO: df.gt,ge,eq,lt,le
# TODO: df.where
import subprocess as sp
import pandas as pd
import numpy as np
import random

def threshold(a: float):
    "Return 0 if value is greater than 5, else return value"
    if a > 5:
        return 0
    else:
        return a

if __name__ == '__main__':
    sp.call('cls', shell = True)

    # load some data
    with open('.\\data\\iris\\iris.data', 'rt') as fl:
        df = pd.read_csv(fl
            ,names = ['sepal_length','sepal_width','petal_length','petal_width','class']
            ,header = None
            ,index_col = False)

    # get all numeric columns
    numeric = df.select_dtypes([np.number]).columns
    nonNumeric = df.select_dtypes([object]).columns
    print('Numeric columns:')
    for val in numeric:
        print('  ' + val)
    print('Non-numeric columns:')
    for val in nonNumeric:
        print('  ' + val)
    print('')

    # generate a random number in index to set to nan
    row = random.randint(df.index[0], df.index[-1])
    col = random.randint(0, len(df.columns) - 1)
    df.iloc[row, col] = np.nan

    # check if any value in the dataframe is na
    print('df.isna():\n{0}\n'.format(df.isna()))

    # for every column, check if there is at least one value that is na
    print('df.isna().any():\n{0}\n'.format(df.isna().any()))

    # for every row, check if there is at least one value that is na
    print('df.isna().any(axis = 1):\n{0}\n'.format(df.isna().any(axis = 1)))

    # boolean indexing
    srs = df.loc[df.loc[:, 'sepal_length'] < 5, :]
    print('rows with sepal_length < 5:\n{0}\n'.format(srs))

    # convert a pandas series to a numpy array
    sepal = df.loc[:, 'sepal_length'].copy().to_numpy()
    print('numpy array of sepal lengths:\n{0}\n'.format(sepal))

    # convert numpy array to Python list
    lssepal = sepal.tolist()
    print('list of sepal lengths:\n{0}\n'.format(lssepal))

    # find row and column with nan value
    row = df.isna().any(axis = 1)
    row = row[row].index[0]
    col = df.isna().any()
    col = col[col].index[0]
    print('row {0} and column {1} has value: {2}\n'.format(row, col, df.loc[row, col]))
    df.loc[row, col] = 0

    # find first index of an element in a Python list
    print('index of sepal_length == 5.4: {0}\n'.format(lssepal.index(5.4)))

    # indexes of a numpy array that fit a criteria
    print('indexes of values of sepal_length < 5:\n{0}\n'.format(np.where(sepal < 5)[0]))

    # apply a function to a series or dataframe
    print('columns without null values:\n{0}\n'.format(df.isna().any().apply(np.logical_not)))
    
    # define and use a vectorized user-defined function and return results as float
    f = np.vectorize(threshold, otypes = [float])
    print('using vectorize function to set all values of sepal_length > 5 to 0:\n{0}\n'.format(f(sepal)))
    
    # update the values of a DataFrame based on a second one
    # create a second dataframe with only odd rows
    df2 = pd.DataFrame(f(sepal), columns = ['sepal_length'])
    df2 = df2.loc[(df2.index % 2) == 1, :]
    df.update(df2)
    print('updated dataframe using another dataframe:\n{0}\n'.format(df))

    # merge one dataframe with another
    srs = df.loc[:, 'petal_length'].copy()
    srs[srs.index % 2 == 0] = 0
    df2 = pd.DataFrame(srs, columns = ['petal_length'])
    df = df.merge(df2
        ,how = 'left'
        ,left_index = True
        ,right_index = True
        #,left_on = columnname
        #,right_on = columnname
        ,suffixes = ['_old', '_new']
        ,copy = False)
    print('merged dataframe:\n{0}\n'.format(df))

    # create an array by specifying a start, stop and interval
    arr = np.arange(0, 5, 0.5)
    print('arr:\n{0}\n'.format(arr))

    # create an array by specifying a start, stop, and the number of
    # equally-spaced values
    arr = np.linspace(0, 4, 17)
    print('arr:\n{0}\n'.format(arr))

    # group by/aggregate the data on a column(s)
    grouped = df.groupby(['class'])
    print('number of elements in each class:\n{0}\n'.format(grouped.size()))