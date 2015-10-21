from firedrake import *
import numpy as np
import gaussian_expression
'''Package with auxilliary methods for logging
'''
__all__ = ['ksp_monitor',
           'logger',
           'gaussian_expression']

_random_seed=318500177
gaussian_expression.init_random(_random_seed,False)

