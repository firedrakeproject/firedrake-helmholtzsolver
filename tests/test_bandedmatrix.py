from firedrake import *
from bandedmatrix import *
import numpy as np
import pytest


@pytest.fixture
def mesh():
    D = 0.1
    nlayers=2
    ncells=3

    host_mesh = CircleManifoldMesh(ncells)
    mesh = ExtrudedMesh(host_mesh,
                        layers=nlayers,
                        extrusion_type='radial',
                        layer_height=D/nlayers)

    return mesh

@pytest.fixture
def finite_elements():
    # Finite elements

    # Horizontal elements
    U1 = FiniteElement('CG',interval,2)
    U2 = FiniteElement('DG',interval,1)

    # Vertical elements
    V0 = FiniteElement('CG',interval,2)
    V1 = FiniteElement('DG',interval,1)

    return U1, U2, V0, V1

@pytest.fixture
def W2_vert(finite_elements,mesh):

    U1, U2, V0, V1 = finite_elements

    # Three dimensional elements
    W2_elt = HDiv(OuterProductElement(U2,V0))

    W2_vert = FunctionSpace(mesh,W2_elt)
    
    return W2_vert

@pytest.fixture
def W3(finite_elements,mesh):

    U1, U2, V0, V1 = finite_elements

    # Three dimensional elements
    W3_elt = OuterProductElement(U2,V1)

    W3 = FunctionSpace(mesh,W3_elt)
    return W3
    
def test_mass_action(W3):
    u = Function(W3)
    v = Function(W3)

    u.interpolate(Expression('x[0]*x[1] + 10*x[1]'))
    v.assign(0)

    mat = BandedMatrix(v.function_space(),u.function_space())
    phi = TestFunction(W3)
    psi = TrialFunction(W3)
    form = phi*psi*dx
    mat.assemble_ufl_form(form)

    mat.axpy(u, v)

    f = assemble(action(form, u))
    assert np.allclose(norm(assemble(f - v)), 0.0)

def test_derivative_action(W2_vert,W3):
    u = Function(W2_vert)
    v = Function(W3)

    u.project(Expression(('x[0]*x[1] + 10*x[1]', 'x[1] - x[0] / 10')))
    v.assign(0)

    mat = BandedMatrix(v.function_space(),u.function_space())
    phi = TestFunction(W3)
    w = TrialFunction(W2_vert)
    form = phi*div(w)*dx
    mat.assemble_ufl_form(form)
    mat.axpy(u, v)
  
    f = assemble(action(form, u))
    assert np.allclose(norm(assemble(f - v)), 0.0)

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
