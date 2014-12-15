from firedrake import *
from pressuresolver.operators import *
from pressuresolver.smoothers import *
import numpy as np
import pytest
from fixtures import *

def test_smoother(W3,W2_horiz,W2_vert,pressure_expression):
    '''Test smoother.

    Check that smoother reduces residual norm

    :arg W3: Pressure space
    :arg W2_horiz: Horizontal velocity component
    :arg W2_vert: Vertical velocity component
    :arg pressure_expression: analytical expression for RHS
    '''

    mesh = W3.mesh()
    ncells = mesh.cell_set.size

    print 'Number of cells on finest grid = '+str(ncells)
    dx = 2./math.sqrt(3.)*math.sqrt(4.*math.pi/(ncells))
   
    omega_c = 8.*0.5*dx
    omega_N = 0.5

    op = Operator_Hhat(W3,W2_horiz,W2_vert,omega_c,omega_N)

    jac = Jacobi(op)

    b = Function(W3).project(pressure_expression)
    phi = Function(W3)
    phi.assign(0)

    res_norm = norm(op.residual(b,phi))
    res_norm0 = res_norm

    res_reduction = []

    print 'k    ||r_k||/||r_0||   ||r_k||/||r_{k-1||}'
    for k in range(10):
        res_norm_old = res_norm
        jac.smooth(b,phi)
        res_norm = norm(op.residual(b,phi))
        rho = res_norm/res_norm_old
        print ('%3d' % k)+'      '+('%6.3f' % (res_norm/res_norm0))+\
              '      '+('%6.3f' % (rho))
        res_reduction.append((rho<1) and (rho>0))
    assert np.all(res_reduction)

##############################################################
# M A I N
##############################################################
if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
