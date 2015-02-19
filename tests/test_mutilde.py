from firedrake import *
from pressuresolver.mu_tilde import *
import numpy as np
import pytest
from fixtures import *

@pytest.fixture(params=[False,True])
def mutilde(request,W2,Wb):
    '''Return a multilde object
    
    :arg request: Use lumping?
    :arg W2: Hdiv velocity space
    :arg Wb: buoyancy space
    '''
    omega_N = 2.0
    return Mutilde(W2,Wb,omega_N,
                   tolerance_u=1.E-10,
                   lumped=request.param)

@pytest.fixture
def bcs(W2):
    '''Dirichlet boundary conditions on top and bottom of the domain.

    :arg W2: Hdiv velocity space
    '''
    return [DirichletBC(W2, 0.0, "bottom"),
            DirichletBC(W2, 0.0, "top")]


def test_mutilde_omegazero(W2,Wb,velocity_expression,bcs):
    '''Check that applying :math:`\\tilde{M}_u` to a field gives the same result
    as :math:`M_u` if :math:`\omega_c=0`.

    :arg W2: Hdiv velocity space
    :arg Wb: buoyancy space
    :arg velocity_expression: Analytical expression for velocity function
    :arg bcs: Boundary conditions on W2
    '''
    omega_N = 0.0
    mutilde = Mutilde(W2,Wb,omega_N)
    u = Function(W2)
    u.project(velocity_expression)
    for bc in bcs:
        bc.apply(u)
    v = mutilde.apply(u)
    w = assemble(dot(TestFunction(W2),u)*dx)
    for bc in bcs:
        bc.apply(w)
    assert np.allclose(v.dat.data - w.dat.data, 0.0)

def test_mutilde_apply(W2,Wb,mutilde,velocity_expression,bcs):
    '''Check that applying :math:`\\tilde{M}_u` to a function returns the correct
    result.

    :arg W2: Hdiv velocity space
    :arg Wb: buoyancy space
    :arg mutilde: Matrix :math:`\\tilde{M}_u` as returned by the fixture
    :arg velocity_expression: Analytical expression for velocity function
    :arg bcs: Boundary conditions on W2
    '''
    zhat = VerticalNormal(W2.mesh()).zhat
    omega_N = mutilde._omega_N
    u = Function(W2)
    u_test = TestFunction(W2)
    b_test = TestFunction(Wb)
    b_trial = TrialFunction(Wb)
    Mb = assemble(b_test*b_trial*dx)
    u.project(velocity_expression)
    v = mutilde.apply(u)
    for bc in bcs:
        bc.apply(u)
    Mbinv_QT_u = Function(Wb)
    QT_u = assemble(dot(zhat*b_test,u)*dx)
    solve(Mb,Mbinv_QT_u,QT_u,solver_parameters={'ksp_type':'cg',
                                                'pc_type':'jacobi',
                                                'ksp_rtol':1E-9})
    Q_MBinv_QT_u = dot(u_test,zhat*Mbinv_QT_u)*dx
    Mu_u = dot(u_test,u)*dx
    w = assemble(Mu_u+omega_N**2*Q_MBinv_QT_u)
    for bc in bcs:
        bc.apply(w)

    atol=1.E-5
    print 'Maximum difference = ', np.max(v.dat.data - w.dat.data)
    assert np.allclose(v.dat.data - w.dat.data, 0.0, atol=atol) 

def test_mutilde_inverse(W2,Wb,mutilde,velocity_expression):
    '''Check that applying :math:`\\tilde{M}_u` to a function and then solving for the
    same operator does not change the function.

    :arg W2: Hdiv velocity space
    :arg Wb: buoyancy space
    :arg mutilde: Matrix :math:`\\tilde{M}_u` as returned by the fixture
    :arg velocity_expression: Analytical expression for velocity function
    '''
    if (mutilde._lumped):
        print 'Skipping tested for lumped \\tilde{M}_u...'
    else:
        u = Function(W2)
        w = Function(W2)
        u.project(velocity_expression)
        v = mutilde.apply(u)
        mutilde.divide(v,w)
        print 'Maximum difference = ', np.max(u.dat.data - w.dat.data)
        assert np.allclose(u.dat.data - w.dat.data, 0.0) 

##############################################################
# M A I N
##############################################################
if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
