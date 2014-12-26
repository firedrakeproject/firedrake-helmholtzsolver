from firedrake import *
from pressuresolver.mu_tilde import *
import numpy as np
import pytest
from fixtures import *

def test_mutilde_omegazero(W2,Wb,velocity_expression):
    '''Check that applying :math:`\\tilde{M}_u` to a field gives the same result
    as :math:`M_u` if :math:`\omega_c=0`.

    :arg W2: Hdiv velocity space
    :arg Wb: buoyancy space
    :arg velocity_expression: Analytical expression for velocity function
    '''
    omega_N = 0.0
    mutilde = Mutilde(W2,Wb,omega_N)
    u = Function(W2)
    u.project(velocity_expression)
    v = mutilde.apply(u)
    w = assemble(dot(TestFunction(W2),u)*dx)
    assert np.allclose(v.dat.data - w.dat.data, 0.0)

def test_mutilde_apply(W2,Wb,velocity_expression):
    '''Check that applying :math:`\\tilde{M}_u` to a function returns the correct
    result.

    :arg W2: Hdiv velocity space
    :arg Wb: buoyancy space
    :arg velocity_expression: Analytical expression for velocity function
    '''
    omega_N = 0.8
    mutilde = Mutilde(W2,Wb,omega_N)
    zhat = VerticalNormal(W2.mesh()).zhat

    u = Function(W2)
    u_test = TestFunction(W2)
    b_test = TestFunction(Wb)
    b_trial = TrialFunction(Wb)
    Mb = assemble(b_test*b_trial*dx)
    u.project(velocity_expression)
    v = mutilde.apply(u)

    Mbinv_QT_u = Function(Wb)
    QT_u = assemble(dot(zhat*b_test,u)*dx)
    solve(Mb,Mbinv_QT_u,QT_u,solver_parameters={'ksp_type':'cg',
                                                'pc_type':'jacobi',
                                                'ksp_rtol':1E-9})
    Q_MBinv_QT_u = dot(u_test,zhat*Mbinv_QT_u)*dx
    Mu_u = dot(u_test,u)*dx
    w = assemble(Mu_u+omega_N**2*Q_MBinv_QT_u)

    assert np.allclose(v.dat.data - w.dat.data, 0.0) 

def test_mutilde_inverse(W2,Wb,velocity_expression):
    '''Check that applying :math:`\\tilde{M}_u` to a function and then solving for the
    same operator does not change the function.

    :arg W2: Hdiv velocity space
    :arg Wb: buoyancy space
    :arg velocity_expression: Analytical expression for velocity function
    '''
    omega_N = 0.8
    mutilde = Mutilde(W2,Wb,omega_N)
    u = Function(W2)
    w = Function(W2)
    u.project(velocity_expression)
    v = mutilde.apply(u)
    mutilde.divide(v,w)
    assert np.allclose(u.dat.data - w.dat.data, 0.0) 

##############################################################
# M A I N
##############################################################
if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
