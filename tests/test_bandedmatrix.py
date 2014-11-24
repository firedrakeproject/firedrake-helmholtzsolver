from firedrake import *
from bandedmatrix import *
import numpy as np
import pytest


@pytest.fixture
def mesh():
    '''Create 1+1 dimensional mesh by extruding a circle.'''
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
def W2_vert(finite_elements,mesh):
    '''HDiv space for vertical velocity component.
            
    Build vertical velocity space :math:`W_2^{v} = Hdiv(U_2\otimes V_0)`
    
    :arg finite_elements: Horizontal and vertical finite element
    :arg mesh: Underlying extruded mesh
    '''

    U1, U2, V0, V1 = finite_elements

    # Three dimensional elements
    W2_elt = HDiv(OuterProductElement(U2,V0))

    W2_vert = FunctionSpace(mesh,W2_elt)
    
    return W2_vert

@pytest.fixture
def W3(finite_elements,mesh):
    '''L2 pressure space.
            
    Build pressure space :math:`W_3 = Hdiv(U_2\otimes V_1)`

    :arg finite_elements: Horizontal and vertical finite element
    :arg mesh: Underlying extruded mesh
    '''

    U1, U2, V0, V1 = finite_elements

    # Three dimensional elements
    W3_elt = OuterProductElement(U2,V1)

    W3 = FunctionSpace(mesh,W3_elt)
    return W3
    
def test_mass_action(W3):
    '''Test mass matrix action.

    Calculate the action of the :math:`W_3` mass matrix on a :math:`W_3` field both using
    the banded matrix class and UFL; compare the results.

    :arg W3: L2 pressure function space
    '''
    u = Function(W3)
    v = Function(W3)

    u.interpolate(Expression('x[0]*x[1] + 10*x[1]'))
    v.assign(0)

    mat = BandedMatrix(W3,W3)
    phi_test = TestFunction(W3)
    phi_trial = TrialFunction(W3)
    form = phi_test*phi_trial*dx
    mat.assemble_ufl_form(form)

    mat.axpy(u, v)

    v_ufl = assemble(action(form, u))
    assert np.allclose(norm(assemble(v_ufl - v)), 0.0)

def test_derivative_action(W2_vert,W3):
    '''Test weak derivative action.

    Calculate the action of the :math:`W_2 \\rightarrow W_3` weak derivative both using
    the banded matrix class and UFL; compare the results.

    :arg W2_vert: L2 pressure function space
    :arg W3: HDiv space for vertical velocity component
    '''
    u = Function(W2_vert)
    v = Function(W3)

    u.project(Expression(('x[0]*x[1] + 10*x[1]', 'x[1] - x[0] / 10')))
    v.assign(0)

    mat = BandedMatrix(W3,W2_vert)
    phi_test = TestFunction(W3)
    w_trial = TrialFunction(W2_vert)
    form = phi_test*div(w_trial)*dx
    mat.assemble_ufl_form(form)
    mat.axpy(u, v)
  
    v_ufl = assemble(action(form, u))
    assert np.allclose(norm(assemble(v_ufl - v)), 0.0)

def test_matmul(W2_vert,W3):
    '''Test matrix multiplication.
    
    Assemble banded matrices for the :math:`W_3` mass matrix and the 
    :math:`W_2\\rightarrow W_3` weak derivative and multiply these banded matrices to
    obtain a new banded matrix. Apply this matrix to a :math:`W_2` field and compare to
    the result of doing the same operation in UFL.

    :arg W2_vert: L2 pressure function space
    :arg W3: HDiv space for vertical velocity component
    '''
    u = Function(W2_vert)
    v = Function(W3)
    v_tmp = Function(W3)

    u.project(Expression(('x[0]*x[1] + 10*x[1]', 'x[1] - x[0] / 10')))
    v.assign(0)

    mat_M = BandedMatrix(W3,W3)
    mat_D = BandedMatrix(W3,W2_vert)

    phi_test = TestFunction(W3)
    phi_trial = TrialFunction(W3)
    w_trial = TrialFunction(W2_vert)

    form_M = phi_test*phi_trial*dx
    form_D = phi_test*div(w_trial)*dx
    mat_M.assemble_ufl_form(form_M)
    mat_D.assemble_ufl_form(form_D)

    mat_MD = mat_M.matmul(mat_D)

    mat_MD.axpy(u, v)

    v_ufl = assemble(action(form_M,assemble(action(form_D, u))))

    assert np.allclose(norm(assemble(v_ufl - v)), 0.0)

@pytest.fixture
def omega():
    return 2.0

@pytest.fixture
def form_M(W3):
    '''UFL form for pressure mass matrix.'''
    phi = TestFunction(W3)
    psi = TrialFunction(W3)
    return phi*psi*dx

@pytest.fixture
def form_D(W2_vert,W3):
    '''UFL form for weak derivative matrix.'''
    phi = TestFunction(W3)
    w = TrialFunction(W2_vert)
    return phi*div(w)*dx

@pytest.fixture
def form_DT(W2_vert,W3):
    '''UFL form for transpose of weak derivative matrix.'''
    w = TestFunction(W2_vert)
    phi = TrialFunction(W3)
    return div(w)*phi*dx

@pytest.fixture
def helmholtz_matrix(W2_vert,W3,form_M,form_D,form_DT,omega):
    '''Build matrix for Helmholtz operator.

    Calculate 

    :math::
        
        H = M_{p} + \omega DD^T
    
    :arg W2_vert: HDiv space for vertical velocity component
    :arg W3: L2 space for pressure
    :arg form_M: UFL form for pressure mass matrix
    :arg form_D: UFL form for weak derivative
    :arg form_DT: UFL form for transpose of weak derivative
    :arg omega: value of parameter omega
    '''
    mat_M = BandedMatrix(W3,W3)
    mat_D = BandedMatrix(W3,W2_vert)
    mat_DT = BandedMatrix(W2_vert,W3)

    mat_M.assemble_ufl_form(form_M)
    mat_D.assemble_ufl_form(form_D)
    mat_DT.assemble_ufl_form(form_DT)

    # Calculate H = M + omega*D*DT
    mat_H = mat_M.matadd(mat_D.matmul(mat_DT),omega=omega)

    return mat_H

def test_matadd(helmholtz_matrix, W2_vert, W3, form_M, form_D, form_DT, omega):
    '''Test matrix addition.
    
    Assemble banded matrices for the :math:`W_3` mass matrix :math:`M` and the 
    :math:`W_2\\rightarrow W_3` weak derivative :math:`D`. Calculate the sum 
    :math:`H = M+omega D D^T`, apply it to a field in :math:`W3` and compare to
    the result of doing the same operation in UFL.

    :arg W2_vert: L2 pressure function space
    :arg W3: HDiv space for vertical velocity component
    :arg form_M: UFL form for pressure mass matrix
    :arg form_D: UFL form for weak derivative
    :arg form_DT: UFL form for transpose of weak derivative
    :arg omega: value of parameter omega
    '''

    u = Function(W3)
    v = Function(W3)
    v_tmp = Function(W3)

    u.project(Expression('x[0]*x[1] + 10*x[1]'))
    v.assign(0)

    mat_H = helmholtz_matrix

    mat_H.axpy(u, v)

    v_ufl = assemble(action(form_M, u)) \
          + omega*assemble(action(form_D,assemble(action(form_DT, u))))

    assert np.allclose(norm(assemble(v_ufl - v)), 0.0)

def test_mass_solve(W3,form_M):
    '''Test LU solver with mass matrix.

    Invert the :math:`W_3` mass matrix on a :math:`W_3` for a given field and
    check that the result is correct by multiplying back by the UFL form.

    :arg W3: L2 pressure function space
    :arg form_M: pressure mass matrix
    '''
    u = Function(W3)
    v = Function(W3)

    u.interpolate(Expression('x[0]*x[1] + 10*x[1]'))

    mat = BandedMatrix(W3,W3)
    mat.assemble_ufl_form(form_M)

    v.assign(u)
    mat.lu_decompose()
    mat.lu_solve(v)

    v_ufl = assemble(action(form_M, v))

    assert np.allclose(norm(assemble(v_ufl - u)), 0.0)

def test_helmholtz_solve(helmholtz_matrix, W2_vert, W3, form_M, form_D, form_DT, omega):
    '''Test LU solver with the Helmholtz matrix.

    Invert the helmholtz matrix on a :math:`W_3` for a given field and
    check that the result is correct by multiplying back by the UFL form.

    :arg W2_vert: L2 pressure function space
    :arg W3: HDiv space for vertical velocity component
    :arg form_M: UFL form for pressure mass matrix
    :arg form_D: UFL form for weak derivative
    :arg form_DT: UFL form for transpose of weak derivative
    :arg omega: value of parameter omega
    '''
    u = Function(W3)
    v = Function(W3)

    u.interpolate(Expression('x[0]*x[1] + 10*x[1]'))

    mat = helmholtz_matrix

    v.assign(u)
    mat.lu_decompose()
    mat.lu_solve(v)

    v_ufl = assemble(action(form_M, v)) \
          + omega*assemble(action(form_D,assemble(action(form_DT, v))))

    assert np.allclose(norm(assemble(v_ufl - u)), 0.0)

##############################################################
# M A I N
##############################################################
if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
