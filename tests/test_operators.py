from firedrake import *
from pressuresolver.operators import *
from pressuresolver.mu_tilde import *
import numpy as np
import pytest
from mixedoperators import *
from fixtures import *
import numpy as np

def test_spectral_radius(R_earth,
                         W3_coarse,
                         W2_coarse,
                         W2_horiz_coarse,
                         W2_vert_coarse,
                         Wb_coarse,
                         velocity_expression):
    '''Calculate condition number of :math:`\hat{H}^{-1}H` and check that
    it is smaller than 10.

    The choice of upper bound is a bit random, so maybe relax it to less than 10.

    :arg R_earth: Earth radius
    :arg W3_coarse: Pressure space
    :arg W2_coarse: Velocity space
    :arg W2_horiz_coarse: Horizontal component of velocity space
    :arg W2_vert_coarse: Vertical component of velocity space
    :arg Wb_coarse: buoyancy space
    :arg velocity_expression: Expression for velocity to project
    '''

    mesh = W3_coarse.mesh()
    ncells = mesh.cell_set.size
    print 'Number of cells on finest grid = '+str(ncells)
    if (mesh.geometric_dimension == 3):
        dx = 2./math.sqrt(3.)*math.sqrt(4.*math.pi/(ncells))*R_earth
    else:
        dx = 2.*math.pi/float(ncells)*R_earth
    N = 0.01
    c = 300.
    nu_cfl = 2.0
    dt = nu_cfl/c*dx
    omega_c = 0.5*c*dt
    omega_N = 0.5*N*dt
    mixed_operator = MixedOperator(W2_coarse,W3_coarse,dt,c,N)
    mutilde = Mutilde(mixed_operator,
                      lumped=False,
                      tolerance_u=1.E-12)
 
    op = Operator_H(W3_coarse,W2_coarse,mutilde,omega_c)
    op_hat = Operator_Hhat(W3_coarse,W2_horiz_coarse,W2_vert_coarse,omega_c,omega_N)

    phi = Function(W3_coarse)

    ndof_pressure = len(phi.dat.data)
    mat = np.zeros((ndof_pressure,ndof_pressure))
    mat_hat = np.zeros((ndof_pressure,ndof_pressure))
    for i in range(ndof_pressure):
        phi.assign(0.0)
        phi.dat.data[i] = 1.0
        psi = op.apply(phi)
        mat[i,:] = psi.dat.data[:]
        psi = op_hat.apply(phi)
        mat_hat[i,:] = psi.dat.data[:]
    m = np.dot(np.linalg.inv(mat_hat),mat)
    kappa = np.linalg.cond(m)
    print 'cond(H)           = '+('%8.4f' % np.linalg.cond(mat))
    print 'cond(Hhat)        = '+('%8.4f' % np.linalg.cond(mat_hat))
    print '--- \hat{H}^{-1}*H ---'
    print 'cond(Hhat^{-1}*H) = '+('%8.4f' % kappa)
    assert (kappa < 10.)

def test_spectral_radius_Hhat(R_earth,
                              W3_coarse,
                              W2_coarse,
                              W2_horiz_coarse,
                              W2_vert_coarse,
                              Wb_coarse,
                              velocity_expression):
    '''Calculate condition number of :math:`\hat{H}_z^{-1}hat{H}` and this is significantly
    smaller than the condition number of the matrix :math:`\hat{H}`

    :arg R_earth: Earth radius
    :arg W3_coarse: Pressure space
    :arg W2_coarse: Velocity space
    :arg W2_horiz_coarse: Horizontal component of velocity space
    :arg W2_vert_coarse: Vertical component of velocity space
    :arg Wb_coarse: buoyancy space
    :arg velocity_expression: Expression for velocity to project
    '''

    mesh = W3_coarse.mesh()
    ncells = mesh.cell_set.size
    print 'Number of cells on finest grid = '+str(ncells)
    if (mesh.geometric_dimension == 3):
        dx = 2./math.sqrt(3.)*math.sqrt(4.*math.pi/(ncells))*R_earth
    else:
        dx = 2.*math.pi/float(ncells)*R_earth
    N = 0.01
    c = 300.
    nu_cfl = 2.0
    dt = nu_cfl/c*dx
    omega_c = 0.5*c*dt
    omega_N = 0.5*N*dt
 
    op_hat = Operator_Hhat(W3_coarse,W2_horiz_coarse,W2_vert_coarse,omega_c,omega_N)
    op_hat_z = op_hat.vertical_diagonal()

    phi = Function(W3_coarse)

    ndof_pressure = len(phi.dat.data)
    mat = np.zeros((ndof_pressure,ndof_pressure))
    mat_prec = np.zeros((ndof_pressure,ndof_pressure))
    for i in range(ndof_pressure):
        phi.assign(0.0)
        phi.dat.data[i] = 1.0
        psi = op_hat.apply(phi)
        mat[i,:] = psi.dat.data[:]
        op_hat_z.solve(psi)
        mat_prec[i,:] = psi.dat.data[:]
    kappa = np.linalg.cond(mat)
    kappa_prec = np.linalg.cond(mat_prec)
    print 'cond(Hhat)             = '+('%8.4f' % kappa)
    print 'cond(Hhat_z^{-1}*Hhat) = '+('%8.4f' % kappa_prec)
    print 'ratio                  = ',kappa_prec/kappa
    assert (kappa_prec/kappa < 1.E-4)
 
def test_operator_Hhat(W3_coarse,
                       W2_horiz_coarse,
                       W2_vert_coarse):
    '''Test operator :math:`\hat{H}`

    Apply the operator to a field and compare the result to that obtained by
    extracting the vertical diagonal operator and applying this to the same field.
    The diagonal entries of the two results should be identical.

    :arg W3_coarse: Pressure space
    :arg W2_horiz_coarse: Horizontal component of velocity space
    :arg W2_vert_coarse: Vertical component of velocity space
    '''
    omega_c = 0.8
    omega_N = 0.9
    op = Operator_Hhat(W3_coarse,W2_horiz_coarse,W2_vert_coarse,omega_c,omega_N)

    op_vert = op.vertical_diagonal() 

    phi = Function(W3_coarse)
    ndof_pressure = len(phi.dat.data)
    mat = np.zeros(ndof_pressure)
    for i in range(ndof_pressure):
        phi.assign(0.0)
        phi.dat.data[i] = 1.0
        psi = op.apply(phi)
        
        mat[i] = psi.dat.data[i]

    mat_vert = np.zeros(ndof_pressure)
    for i in range(ndof_pressure):
        phi.assign(0.0)
        phi.dat.data[i] = 1.0
        op_vert.ax(phi)
        mat_vert[i] = phi.dat.data[i]

    assert np.allclose(mat - mat_vert, 0.0,atol=1.E-5)

##############################################################
# M A I N
##############################################################
if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
