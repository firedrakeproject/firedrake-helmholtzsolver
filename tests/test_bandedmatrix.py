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

def test_matmul(W2_vert,W3):

    u = Function(W2_vert)
    v = Function(W3)
    v_tmp = Function(W3)

    u.project(Expression(('x[0]*x[1] + 10*x[1]', 'x[1] - x[0] / 10')))
    v.assign(0)

    mat_A = BandedMatrix(v.function_space(),v.function_space())
    mat_B = BandedMatrix(v.function_space(),u.function_space())

    phi = TestFunction(W3)
    psi = TrialFunction(W3)
    w = TrialFunction(W2_vert)

    form_A = phi*psi*dx
    form_B = phi*div(w)*dx
    mat_A.assemble_ufl_form(form_A)
    mat_B.assemble_ufl_form(form_B)
    mat_C = mat_A.matmul(mat_B)

    mat_C.axpy(u, v)

    f_u = assemble(action(form_B, u))
    f_v = assemble(action(form_A, f_u))

    assert np.allclose(norm(assemble(f_v - v)), 0.0)

def test_matadd(W2_vert,W3):

    u = Function(W3)
    v = Function(W3)
    v_tmp = Function(W3)

    u.project(Expression('x[0]*x[1] + 10*x[1]'))
    v.assign(0)

    mat_M = BandedMatrix(W3,W3)
    mat_D = BandedMatrix(W3,W2_vert)
    mat_DT = BandedMatrix(W2_vert,W3)

    phi = TestFunction(W3)
    psi = TrialFunction(W3)
    w = TrialFunction(W2_vert)
    w2 = TestFunction(W2_vert)

    form_M = phi*psi*dx
    form_D = phi*div(w)*dx
    form_DT = div(w2)*psi*dx

    omega = 2.0

    mat_M.assemble_ufl_form(form_M)
    mat_D.assemble_ufl_form(form_D)
    mat_DT.assemble_ufl_form(form_DT)

    # Calculate H = M + omega*D*DT
    mat_H = mat_M.matadd(mat_D.matmul(mat_DT),omega=omega)

    mat_H.axpy(u, v)

    f = assemble(action(form_M, u)) \
      + omega*assemble(action(form_D,assemble(action(form_DT, u))))

    assert np.allclose(norm(assemble(f - v)), 0.0)

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
