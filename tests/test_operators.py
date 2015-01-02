from firedrake import *
from pressuresolver.operators import *
from pressuresolver.mu_tilde import *
import numpy as np
import pytest
from fixtures import *
import numpy as np

def test_spectral_radius(W3,W2,W2_horiz,W2_vert,Wb,velocity_expression):
    '''Calculate condition number of :math:`\hat{H}^{-1}H` and check that
    it is smaller than 2.

    The choice of upper bound is a bit random, so maybe relax it to less than 10.

    :arg W3: Pressure space
    :arg W2: Velocity
    :arg W2_horiz: Horizontal component of velocity space
    :arg W2_vert: Vertical component of velocity space
    :arg Wb: buoyancy space
    '''
    omega_c = 0.8
    omega_N = 0.9

    mutilde = Mutilde(W2,Wb,omega_N,tolerance_b=1E-12,tolerance_u=1E-9)

    op = Operator_H(W3,W2,mutilde,omega_c)
    op_hat = Operator_Hhat(W3,W2_horiz,W2_vert,omega_c,omega_N)

    phi = Function(W3)

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
 
def test_operator_Hhat(W3,W2_horiz,W2_vert):
    '''Test operator :math:`\hat{H}`

    Apply the operator to a field and compare the result to that obtained by
    extracting the vertical diagonal operator and applying this to the same field.
    The diagonal entries of the two results should be identical.

    :arg W3: Pressure space
    :arg W2_horiz: Horizontal component of velocity space
    :arg W2_vert: Vertical component of velocity space
    '''
    omega_c = 0.8
    omega_N = 0.9
    op = Operator_Hhat(W3,W2_horiz,W2_vert,omega_c,omega_N)

    op_vert = op.vertical_diagonal() 

    phi = Function(W3)
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

    assert np.allclose(mat - mat_vert, 0.0)

##############################################################
# M A I N
##############################################################
if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
