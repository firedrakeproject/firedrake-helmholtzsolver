from firedrake import *
from pressuresolver.lumpedmass import *
import numpy as np
import pytest
from fixtures import *
op2.init(log_level="WARNING")
from ffc import log
log.set_level(log.ERROR)
    
def test_full_mass(W2,velocity_expression):
    '''Test the full mass matrix.

    Calculate the full mass matrix and compare it's action to the hand assembled version.

    :arg W2: Velocity space
    :arg velocity_expression: Analytical expression for velocity
    '''
    full_mass = FullMass(W2)
    
    v = Function(W2).project(velocity_expression)
    full_mass.multiply(v)

    u = Function(W2).project(velocity_expression)
    w = TestFunction(W2)
    v_ufl = assemble(dot(w,u)*dx)
    assert np.allclose(v.dat.data-v_ufl.dat.data, 0.0)

def test_full_mass_inverse(W2,velocity_expression):
    '''Test the full mass matrix inverse.

    Multiply and divide a field by the full mass matrix and check that the result
    is the origonal field.

    :arg W2: Velocity space
    :arg velocity_expression: Analytical expression for velocity
    '''
    full_mass = FullMass(W2)
    
    u = Function(W2).project(velocity_expression)
    full_mass.multiply(u)
    full_mass.divide(u)

    u_0 = Function(W2).project(velocity_expression)
    assert np.allclose(u.dat.data-u_0.dat.data, 0.0)

def test_lumped_mass_diagonal(W2_horiz):
    '''Check that we really picked out the diagonal elements of the mass matrix.

    Calculate the lumped mass matrix and the full matrix representation of
    the full velocity mass matrix. Compare the entries of the lumped mass matrix, 
    which are represented by a velocity field, to the diagonal entries of the full
    velocity mass matrix.

    :arg W2_vert: Velocity space
    '''
    lumped_mass = LumpedMass(W2_horiz) 
    
    u = TestFunction(W2_horiz)
    v = TrialFunction(W2_horiz)
    full_mass_matrix = assemble(dot(u,v)*dx).M.values
    diff = [x-y[i] for i,(x,y) in enumerate(zip(lumped_mass._data.dat.data,full_mass_matrix))]
    assert np.allclose(diff, 0.0)

def test_lumped_mass_inverse(W2,velocity_expression):
    '''Test the lumped mass matrix inverse.

    Multiply and divide a field by the lumped mass matrix and check that the result
    is the origonal field.

    :arg W2: Velocity space
    :arg velocity_expression: Analytical expression for velocity
    '''
    full_mass = LumpedMass(W2)
    
    u = Function(W2).project(velocity_expression)
    full_mass.multiply(u)
    full_mass.divide(u)

    u_0 = Function(W2).project(velocity_expression)
    assert np.allclose(u.dat.data-u_0.dat.data, 0.0)


##############################################################
# M A I N
##############################################################
if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
