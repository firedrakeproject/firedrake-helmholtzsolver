from firedrake import *
from pressuresolver.lumpedmass import *
import numpy as np
import pytest
from fixtures import *
op2.init(log_level="WARNING")
from ffc import log
log.set_level(log.ERROR)

def test_lumped_mass_diagonal(W2_coarse):
    '''Check that we really picked out the diagonal elements of the mass matrix.

    Calculate the lumped mass matrix and the full matrix representation of
    the full velocity mass matrix. Compare the entries of the lumped mass matrix, 
    which are represented by a velocity field, to the diagonal entries of the full
    velocity mass matrix.

    :arg W2_coarse: Velocity space on coarse grid
    '''
    
    u = TestFunction(W2_coarse)
    v = TrialFunction(W2_coarse)
    ufl_form = dot(u,v)*dx
    full_mass_matrix = assemble(ufl_form).M.values
    lumped_mass = LumpedMass(ufl_form) 
    diff = [x-y[i] for i,(x,y) in enumerate(zip(lumped_mass._data.dat.data,full_mass_matrix))]
    assert np.allclose(diff, 0.0)

def test_lumped_mass_inverse(W2_coarse,velocity_expression):
    '''Test the lumped mass matrix inverse.

    Multiply and divide a field by the lumped mass matrix and check that the result
    is the origonal field.

    :arg W2_coarse: Velocity space on coarse grid
    :arg velocity_expression: Analytical expression for velocity
    '''
    u = TestFunction(W2_coarse)
    v = TrialFunction(W2_coarse)
    ufl_form = dot(u,v)*dx

    mass = LumpedMass(ufl_form)
    
    u = Function(W2_coarse).project(velocity_expression)
    mass.multiply(u)
    mass.divide(u)

    u_0 = Function(W2_coarse).project(velocity_expression)
    mu = 1./np.max(u_0.dat.data)
    assert np.allclose(mu*(u.dat.data-u_0.dat.data), 0.0)


##############################################################
# M A I N
##############################################################
if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
