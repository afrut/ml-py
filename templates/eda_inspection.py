#exec(open('templates\\eda_inspection.py').read())
import subprocess as sp
import pandas as pd

if __name__ == '__main__':
    sp.call('cls', shell = True)
    print("hello world")

    # load some data
    with open('.\\data\\iris\\iris.data', 'rt') as fl:
        df = pd.read_csv(fl
            ,names = ['sepal_length','sepal_width','petal_length','petal_width','class']
            ,header = None
            ,index_col = False)

    # shape of the dataset
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
    print('Statistical Summary:')
    print(df.describe())