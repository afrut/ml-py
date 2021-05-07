#exec(open('.\\templates\\plot_formatxticklabels.py').read())
#set_ytick
#set_xtick
#set_yticklabels
#set_xticklabels
import subprocess as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

if __name__ == '__main__':
    sp.call('cls', shell = True)
    plt.close('all')

    with open('.\\data\\iris\\iris.data', 'rt') as fl:
        df = pd.read_csv(fl
            ,names = ['sepal_length','sepal_width','petal_length','petal_width','class']
            ,header = None
            ,index_col = False)

    fig = plt.figure(figsize = (14.4, 9))
    ax = fig.add_subplot(1, 1, 1)
    ax = sns.boxplot(data = df)
    ax.set_title('Unformatted xticklabels')

    fig = plt.figure(figsize = (14.4, 9))
    ax = fig.add_subplot(1, 1, 1)
    ax = sns.boxplot(data = df)
    ax.set_title('Formatted xticklabels')

    # format every xticklabel
    for ticklabel in ax.get_xticklabels():
        ticklabel.set_horizontalalignment('right')  # center, right, left
        ticklabel.set_rotation_mode('anchor')       # None or anchor
        ticklabel.set_rotation(30)                  # angle of rotation
        ticklabel.set_fontsize(12)                  # float

    plt.show()
    plt.close('all')