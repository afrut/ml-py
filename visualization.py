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

    datacfg = datacfg.datacfg()
    for datasetname in datacfg.keys():
        print(datasetname)
        df = pk.load(open(datacfg[datasetname]['filepath'], 'rb'))

        plots.stemleaf(df
            ,title = 'Stem and Leaf'
            ,save = True
            ,savepath = '.\\png\\plots\\stemleaf\\' + datasetname + '.txt')

        plots.histogram(df
            ,save = True
            ,savepath = '.\\png\\plots\\histogram\\' + datasetname + '.png'
            ,close = True)

        plots.boxplot(df
            ,save = True
            ,savepath = '.\\png\\plots\\boxplot\\' + datasetname + '.png'
            ,close = True);





    plt.show()
