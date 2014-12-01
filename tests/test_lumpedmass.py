from firedrake import *
from pressuresolver.lumpedmass_diagonal import *
import numpy as np
import pytest


@pytest.fixture
def mesh():
    '''Create 1+1 dimensional mesh by extruding a circle.'''
    D = 0.1
    nlayers=4
    ncells=3

    host_mesh = CircleManifoldMesh(ncells)
    mesh = ExtrudedMesh(host_mesh,
                        layers=nlayers,
                        extrusion_type='radial',
                        layer_height=D/nlayers)

    return mesh

@pytest.fixture
def finite_elements():
    '''Create finite elements of horizontal and vertical function spaces.

    :math:`U_1` = horizontal H1 space
    :math:`U_2` = horizontal L2 space
    :math:`V_0` = vertical H1 space
    :math:`V_1` = vertical L2 space
    '''
    # Finite elements

    # Horizontal elements
    U1 = FiniteElement('CG',interval,2)
    U2 = FiniteElement('DG',interval,1)

    # Vertical elements
    V0 = FiniteElement('CG',interval,2)
    V1 = FiniteElement('DG',interval,1)

    return U1, U2, V0, V1

@pytest.fixture
def W2_horiz(finite_elements,mesh):
    '''HDiv space for horizontal velocity component.
            
    Build vertical horizontal space :math:`W_2^{h} = Hdiv(U_1\otimes V_1)`
    
    :arg finite_elements: Horizontal and vertical finite element
    :arg mesh: Underlying extruded mesh
    '''

    U1, U2, V0, V1 = finite_elements

    # Three dimensional elements
    W2_elt = HDiv(OuterProductElement(U1,V1))

    W2_horiz = FunctionSpace(mesh,W2_elt)
    
    return W2_horiz
    
def test_diagonal(W2_horiz):
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

##############################################################
# M A I N
##############################################################
if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
