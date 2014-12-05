from firedrake import *
from pressuresolver.lumpedmass_diagonal import *
import numpy as np
import pytest
from fixtures import *
    
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
