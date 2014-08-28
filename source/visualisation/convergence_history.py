import re
import sys
import numpy as np

class ConvergenceHistory(object):
    '''Class representing a convergence history.

    The history is stored internally in three numpy arrays
    its, rnorm (residual norm) and rnorm_rel (residual norm divided by
    initial residual).

    :arg filename: input filename
    '''
    def __init__(self,filename,label=None):
        self.filename=filename
        if (label):
            self.label = label
        else:
            self.label = filename
        self.its = []
        self.rnorm = []
        self.rnorm_rel = []
        self._readfile()

    def _readfile(self):
        '''Read convergence history from file.
        '''
        regex_int = '[0-9]+'
        regex_float = '[0-9]+\.[0-9]+[eE][\+\-][0-9]+'
        datafile = open(self.filename,'r')
        for line in datafile:
            m = re.match(' *KSP *mixed *('+regex_int+') *: *('+regex_float+') *('+regex_float+').*',line)
            if m:
                self.its.append(int(m.group(1)))
                self.rnorm.append(float(m.group(2)))
                self.rnorm_rel.append(float(m.group(3)))
        self.its = np.array(self.its)
        self.rnorm = np.array(self.rnorm)
        self.rnorm_rel = np.array(self.rnorm_rel)
        datafile.close()
