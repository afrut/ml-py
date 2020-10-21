#exec(open('visualization.py').read())
import importlib as il
import subprocess as sp
import numpy as np
import pandas as pd
import pickle as pk
import stemgraphic as st
import math
import plots
import datacfg
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sp.call('cls', shell = True)
    il.reload(plots)
    il.reload(datacfg)
    plt.close()

    # create DataFrame from data
    datacfg = datacfg.datacfg()
    df = pk.load(open(datacfg['iris']['filepath'], 'rb'))

    # define a column of interest
    col = 'petal_width'

    vals = dfIris.loc[:, col]
    fig, ax = plots.stemleaf(vals
        ,title = 'Stem and Leaf Plot of {0}'.format(col))





    plt.show()
