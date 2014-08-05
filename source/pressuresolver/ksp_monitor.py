from mpi_utils import Logger
import numpy as np
from mpi4py import MPI

class KSPMonitor(object):
    '''KSP Monitor writing to stdout.

    :arg label: Name of solver
    :arg verbose: verbosity. 0=print nothing, 1=only print summary of results, 2=print everything in detail
    '''
    def __init__(self,label='',verbose=2):
        self.label = label
        self.verbose = verbose
        self.initial_residual = 1.0
        self.logger = Logger()
        self.iterations = []
        self.resnorm = []

    '''Call logger. 

    This method is called by the KSP class and should write the output.
    
    :arg ksp: The calling ksp instance
    :arg its: The current iterations
    :arg rnorm: The current residual norm
    '''
    def __call__(self,ksp,its,rnorm):
        if (its==0):
            self.rnorm0 = rnorm
            if (self.verbose >= 1):
                s = '  KSP '+('%20s' % self.label)
                s += '    iter             rnrm   rnrm/rnrm_0       rho'
                self.logger.write(s)
        if (self.verbose>=2):
            s = '  KSP '+('%20s' % self.label)
            s += ('  %6d' % its)+' : '
            s += ('  %10.6e' % rnorm)
            s += ('  %10.6e' % (rnorm/self.rnorm0))
            if (its > 0):
                s += ('  %8.4f' % (rnorm/self.rnorm_old))
            else:
                s += '      ----'
        if (self.verbose >= 1):
            self.logger.write(s) 
        self.iterations.append(its)
        self.resnorm.append(rnorm)
        self.rnorm_old = rnorm

    def save_convergence_history(self,filename):
        '''Save the convergence history to a file.

        :arg filename: name of file to write to
        '''
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if (rank == 0):
            file = open(filename,'w')
            print >> file, '# 1: iteration k'
            print >> file, '# 2: residual norm (rnorm_k)'
            print >> file, '# 3: relative residual norm (rnorm_k/rnorm_0)'
            print >> file, '# 4: convergence factor (rnorm_k/rnorm_{k-1})'
            print >> file, ''
            for (its,rnorm) in zip(self.iterations,self.resnorm):
                s = ('  %6d' % its)
                s += ('  %10.6e' % rnorm)
                s += ('  %10.6e' % (rnorm/self.rnorm0))
                if (its > 0):
                    s += ('  %8.4f' % (rnorm/rnorm_old))
                else:
                    s += '      ----'
                rnorm_old = rnorm
                print >> file, s
            file.close()
            
        
