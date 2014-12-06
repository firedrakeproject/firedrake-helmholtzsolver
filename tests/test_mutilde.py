from firedrake import *
from pressuresolver.mu_tilde import *
import numpy as np
import pytest
from fixtures import *
from ffc import log
log.set_level(log.ERROR)
op2.init(log_level="WARNING")

def test_mutilde_omegazero(W3,W2,Wb):
    '''Check that applying :math:`\\tilde{M}_u` to a field gives the same result
    as :math:`M_u` if :math:`\omega_c=0`.

    :arg W3: L2 pressure space
    :arg W2: Hdiv velocity space
    :arg Wb: buoyancy space
    '''
    omega_N = 0.0
    mutilde = Mutilde(W3,W2,Wb,omega_N)
    u = Function(W2)
    u.project(Expression(('x[0]+2.*x[1]','x[0]*(x[0]+1.)')))
    v = mutilde.apply(u)
    w = assemble(dot(TestFunction(W2),u)*dx)
    assert np.allclose(v.dat.data - w.dat.data, 0.0)

def test_mutilde_inverse(W3,W2,Wb):
    '''Check that applying :math:`\\tilde{M}_u` to a function and then solving for the
    same operator does not change the function.

    :arg W3: L2 pressure space
    :arg W2: Hdiv velocity space
    :arg Wb: buoyancy space
    '''
    omega_N = 0.8
    mutilde = Mutilde(W3,W2,Wb,omega_N)
    u = Function(W2)
    u.project(Expression(('x[0]+2.*x[1]','x[0]*(x[0]+1.)')))
    v = mutilde.apply(u)
    w = mutilde.divide(v)
    assert np.allclose(u.dat.data - w.dat.data, 0.0) 

##############################################################
# M A I N
##############################################################
if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
