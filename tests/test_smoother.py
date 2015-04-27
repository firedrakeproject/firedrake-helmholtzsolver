from firedrake import *
from pressuresolver.operators import *
from pressuresolver.smoothers import *
import numpy as np
import pytest
from fixtures import *

def test_smoother_lowest_order(R_earth,
                               W3_hierarchy,
                               W2_horiz_hierarchy,
                               W2_vert_hierarchy,
                               pressure_expression):
    '''Test smoother.

    Check that smoother reduces residual norm. Note that we pass the entire
    function space hierarchies, but only use the function spaces on the finest grid.

    :arg R_earth: Earth radius
    :arg W3_hierarchy: Pressure space hierarchy
    :arg W2_horiz_hierarchy: Horizontal velocity component hierarchy
    :arg W2_vert_hierarchy: Vertical velocity component hierarchy
    :arg pressure_expression: analytical expression for RHS
    '''

    W3 = W3_hierarchy[-1]
    W2_horiz = W2_horiz_hierarchy[-1]
    W2_vert = W2_vert_hierarchy[-1]

    mesh = W3.mesh()
    ncells = mesh.cell_set.size

    print 'Number of cells on finest grid = '+str(ncells)
    dx = 2./math.sqrt(3.)*math.sqrt(4.*math.pi/(ncells))*R_earth

    nu_cfl = 2.0
    c = 300.
    N = 0.01
    dt = nu_cfl/c*dx

    omega_c = 0.5*c*dt
    omega_N = 0.5*N*dt

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

def test_smoother(R_earth,W3,W2_horiz,W2_vert,pressure_expression):
    '''Test smoother.

    Check that smoother reduces residual norm

    :arg R_earth: Earth radius
    :arg W3: Pressure space
    :arg W2_horiz: Horizontal velocity component
    :arg W2_vert: Vertical velocity component
    :arg pressure_expression: analytical expression for RHS
    '''

    mesh = W3.mesh()
    ncells = mesh.cell_set.size

    print 'Number of cells on finest grid = '+str(ncells)
    dx = 2./math.sqrt(3.)*math.sqrt(4.*math.pi/(ncells))*R_earth
   
    nu_cfl = 2.0
    c = 300.
    N = 0.01
    dt = nu_cfl/c*dx

    omega_c = 0.5*c*dt
    omega_N = 0.5*N*dt

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
