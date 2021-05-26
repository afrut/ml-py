#exec(open('plots.py').read())
import subprocess as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle as pk

# general plotting flow
# - create figure: fig = plt.figure(figsize = figsize)
# - add subplots: ax = fig.add_subplot(nrows, ncols, index)
# - plot on axes: ax.plotting_function ie. boxplot
# - set axis title: ax.set_title(some_string)
# - set xlims: ax.set_xlim(xmin, xmax)
# - set_ylims: ax.set_ylim(ymin, ymax)
# - set xticks: ax.set_xticks
# - set xticklabels: ax.set_xticklabels
# - format xticklabels: formatxticklabels(ax)
# - set yticks: ax.set_yticks
# - set yticklabels: ax.set_yticklabels
# - format y ticklabels: formatxticklabels(ax)
# - set x-axis label: ax.set_xlabel(xlabel)
# - set y-axis label: ax.set_ylabel(ylabel)
# - add a grid: ax.grid(linewidth = 0.5)
# - tighten margins: fig.tight_layout()
# - add an overall figure title: ax.suptitle(some_title)
# - save: ax.savefig(some_path, format = 'png')
# - show: plt.show()
# - close: plt.close('all)

# helper function to format xticklabels
def formatxticklabels(ax: mpl.axes.Axes
    ,horizontalalignment: str = 'right'
    ,rotationmode: str = 'anchor'
    ,xticklabelrotation: int = 30
    ,xticklabelfontsize: int = 10):
    for ticklabel in ax.get_xticklabels():
        ticklabel.set_horizontalalignment(horizontalalignment)
        ticklabel.set_rotation_mode(rotationmode)
        ticklabel.set_rotation(xticklabelrotation)
        ticklabel.set_fontsize(xticklabelfontsize)

if __name__ == '__main__':
    sp.call('cls', shell = True)

    with open('.\\data\\pima.pkl','rb') as fl:
        df = pk.load(fl)