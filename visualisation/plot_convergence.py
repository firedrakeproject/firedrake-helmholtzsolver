import sys
import numpy as np
from matplotlib import pyplot as plt
from convergence_history import *

def plot_history(histories,filename):
    '''Plot a set of convergence histories.

    :arg histories: List of convergence histories, i.e. instances of
        :class:`ConvergenceHistory`.
    :arg filename: File to write output to
    '''
    colors = ['blue','red','green','black','magenta']
    plt.clf()
    ax = plt.gca()
    ax.set_xlim(-0.5,20.5)
    ax.set_ylim(1E-6/1.2,1.2)
    ax.set_yscale('log')
    ax.set_xlabel('iteration')
    ax.set_ylabel('relative residual $||r||_2/||r_0||_2$')
    plots = []
    labels = []
    for (hist,color) in zip(histories,colors):
        plots.append(plt.plot(hist.its,hist.rnorm_rel,
                              linewidth=2,
                              color=color,
                              markersize=6,
                              markerfacecolor='white',
                              markeredgecolor=color,
                              markeredgewidth=2,
                              marker='o')[0])
        labels.append(hist.label)
    plt.legend(plots,labels,'upper right')
    plt.savefig(filename,bbox_inches='tight')

#########################################################
# M A I N
#########################################################
if (__name__ == '__main__'):
    if ((len(sys.argv) <= 2) or (len(sys.argv) % 2)):
        print 'Usage: python '+sys.argv[0]+' <datafilename_1> <label_1> ... <datafilename_n> <label_n> <outputfilename>'
        sys.exit(1)
    n_files = (len(sys.argv)-2)/2
    datafilenames = sys.argv[1:2*n_files+1:2]
    labels = sys.argv[2:2*n_files+1:2]
    outputfilename = sys.argv[-1]
    histories = [ConvergenceHistory(x,y) for (x,y) in zip(datafilenames,
                                                          labels)]
    plot_history(histories,outputfilename)
