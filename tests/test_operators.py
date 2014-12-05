from firedrake import *
from pressuresolver.operators3d import *
import numpy as np
import pytest
from fixtures import *

 
def test_operator_Hhat(W3,W2_horiz,W2_vert):
    '''Test operator :math:`\hat{H}`


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
