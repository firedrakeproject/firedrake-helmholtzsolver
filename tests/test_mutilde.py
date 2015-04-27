from firedrake import *
from pressuresolver.mu_tilde import *
from mixedoperators import *
import numpy as np
import pytest
from fixtures import *

@pytest.fixture(params=[False,True])
def mutilde(request,W2,W3):
    '''Return a multilde object
    
    :arg request: Use lumping?
    :arg W2: Hdiv velocity space
    :arg W3: L2 pressure space
    '''
    dt = 10.
    N = 0.01
    c = 300.
    mixed_operator = MixedOperator(W2,W3,dt,c,N)
    return Mutilde(mixed_operator,
                   lumped=request.param,
                   tolerance_u=1.E-12)

def test_mutilde_omegazero(W2,W3,mutilde,velocity_expression):
    '''Check that applying :math:`\\tilde{M}_u` to a field gives the same result
    as :math:`M_u` if :math:`\omega_c=0`.

    :arg W2: Hdiv velocity space
    :arg W3: L3 pressure space 
    :arg velocity_expression: Analytical expression for velocity function
    '''
    dt = 2.0
    N = 0.0
    c = 300.
    mixed_operator = MixedOperator(W2,W3,dt,c,N)
    mutilde = Mutilde(mixed_operator,
                      lumped=False,
                      tolerance_u=1.E-10)
    u = Function(W2)
    u.project(velocity_expression)
    v = mutilde.apply(u)
    w = assemble(dot(TestFunction(W2),u)*W3.mesh()._dx,bcs=mutilde._bcs)

    mu = 1./max(v.dat.data)
    assert np.allclose(mu*(v.dat.data,w.dat.data),0.0)

def test_mutilde_apply(W2,Wb,mutilde,velocity_expression):
    '''Check that applying :math:`\\tilde{M}_u` to a function returns the correct
    result.

    :arg W2: Hdiv velocity space
    :arg Wb: buoyancy space
    :arg mutilde: Matrix :math:`\\tilde{M}_u` as returned by the fixture
    :arg velocity_expression: Analytical expression for velocity function
    '''
    if (mutilde._lumped):
        print r'Skipping test for lumped \tilde{M}_i...'
        return
    zhat = VerticalNormal(W2.mesh()).zhat
    omega_N2 = mutilde._mixed_operator._omega_N2
    u = Function(W2)
    u_test = TestFunction(W2)
    b_test = TestFunction(Wb)
    b_trial = TrialFunction(Wb)
    Mb = assemble(b_test*b_trial*dx)
    u.project(velocity_expression)
    v = mutilde.apply(u)
    for bc in mutilde._bcs:
        bc.apply(u)
    Mbinv_QT_u = Function(Wb)
    QT_u = assemble(dot(zhat*b_test,u)*dx)
    solve(Mb,Mbinv_QT_u,QT_u,solver_parameters={'ksp_type':'cg',
                                                'pc_type':'jacobi',
                                                'ksp_rtol':1E-9})
    Q_MBinv_QT_u = dot(u_test,zhat*Mbinv_QT_u)*dx
    Mu_u = dot(u_test,u)*dx
    w = assemble(Mu_u+omega_N2*Q_MBinv_QT_u)
    for bc in mutilde._bcs:
        bc.apply(w)

    mu = 1./max(v.dat.data)
    assert np.allclose(mu*(v.dat.data - w.dat.data), 0.0) 

def test_mutilde_inverse(W2,Wb,mutilde,velocity_expression):
    '''Check that applying :math:`\\tilde{M}_u` to a function and then solving for the
    same operator does not change the function.

    :arg W2: Hdiv velocity space
    :arg Wb: buoyancy space
    :arg mutilde: Matrix :math:`\\tilde{M}_u` as returned by the fixture
    :arg velocity_expression: Analytical expression for velocity function
    '''
    u = Function(W2)
    w = Function(W2)
    u.project(velocity_expression)
    v = mutilde.apply(u)
    mutilde.divide(v,w,tolerance=1.E-12,preonly=False)
    print 'Maximum difference = ', np.max(u.dat.data - w.dat.data)
    mu = 1./max(u.dat.data)
    assert np.allclose(mu*(u.dat.data - w.dat.data), 0.0) 

##############################################################
# M A I N
##############################################################
if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
