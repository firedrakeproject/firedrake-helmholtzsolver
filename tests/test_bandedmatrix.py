from firedrake import *
from bandedmatrix import *
import numpy as np
import pytest
from fixtures import *
    
def allclose(u,v):
    '''Compare two functions, and normalise them first

        :arg u: First function
        :arg v: Second function
    '''
    mu = 1./np.max(u.dat.data)
    return np.allclose(mu*(u.dat.data-v.dat.data),0.0)

def test_mass_action_inplace(W3,pressure_expression):
    '''Test mass matrix action in place.

    Calculate the action of the :math:`W_3` mass matrix on a :math:`W_3` field both using
    the banded matrix class and UFL; compare the results.

    :arg W3: L2 pressure function space
    :arg pressure_expression: Analytical expression for pressure
    '''
    u = Function(W3)
    v = Function(W3)

    u.interpolate(pressure_expression)
    v.assign(u)

    mat = BandedMatrix(W3,W3)
    phi_test = TestFunction(W3)
    phi_trial = TrialFunction(W3)
    form = phi_test*phi_trial*dx
    mat.assemble_ufl_form(form)

    mat.ax(v)

    v_ufl = assemble(action(form, u))
    assert allclose(v_ufl,v)

def test_mass_action(W3,pressure_expression):
    '''Test mass matrix action.

    Calculate the action of the :math:`W_3` mass matrix on a :math:`W_3` field both using
    the banded matrix class and UFL; compare the results.

    :arg W3: L2 pressure function space
    :arg pressure_expression: Analytical expression for pressure
    '''
    u = Function(W3)
    v = Function(W3)

    u.interpolate(pressure_expression)
    v.assign(0)

    mat = BandedMatrix(W3,W3)
    phi_test = TestFunction(W3)
    phi_trial = TrialFunction(W3)
    form = phi_test*phi_trial*dx
    mat.assemble_ufl_form(form)

    mat.axpy(u, v)

    v_ufl = assemble(action(form, u))
    assert allclose(v_ufl,v)

def test_derivative_action(W2_vert,W3,velocity_expression):
    '''Test weak derivative action.

    Calculate the action of the :math:`W_2 \\rightarrow W_3` weak derivative both using
    the banded matrix class and UFL; compare the results.

    :arg W2_vert: L2 pressure function space
    :arg W3: HDiv space for vertical velocity component
    :arg velocity_expression: Analytical expression for velocity
    '''
    u = Function(W2_vert)
    v = Function(W3)

    u.project(velocity_expression)
    v.assign(0)

    mat = BandedMatrix(W3,W2_vert)
    phi_test = TestFunction(W3)
    w_trial = TrialFunction(W2_vert)
    form = phi_test*div(w_trial)*dx
    mat.assemble_ufl_form(form)
    mat.axpy(u, v)
  
    v_ufl = assemble(action(form, u))
    assert allclose(v_ufl,v)

def test_matmul(W2_vert,W3,velocity_expression):
    '''Test matrix multiplication :math:`C=AB`.
    
    Assemble banded matrices for the :math:`W_3` mass matrix and the 
    :math:`W_2\\rightarrow W_3` weak derivative and multiply these banded matrices to
    obtain a new banded matrix. Apply this matrix to a :math:`W_2` field and compare to
    the result of doing the same operation in UFL.

    :arg W2_vert: L2 pressure function space
    :arg W3: HDiv space for vertical velocity component
    :arg velocity_expression: Analytical expression for velocity
    '''
    u = Function(W2_vert)
    v = Function(W3)

    u.project(velocity_expression)
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
    assert allclose(v_ufl,v)

def test_transpose_matmul(W2_vert,W3,pressure_expression):
    '''Test matrix multiplication :math:`C=A^TB`.
    
    Assemble banded matrices for the :math:`W_3` mass matrix and the 
    transpose of the :math:`W_2\\rightarrow W_3` weak derivative and multiply these banded 
    matrices to obtain a new banded matrix.
    Apply this matrix to a :math:`W_3` field and compare to the result of doing the same
    operation in UFL.

    :arg W2_vert: L2 pressure function space
    :arg W3: HDiv space for vertical velocity component
    :arg pressure_expression: Analytical expression for pressue
    '''
    u = Function(W3)
    v = Function(W2_vert)

    u.project(pressure_expression)
    v.assign(0)

    mat_M = BandedMatrix(W3,W3)
    mat_D = BandedMatrix(W3,W2_vert)

    phi_test = TestFunction(W3)
    phi_trial = TrialFunction(W3)
    w_trial = TrialFunction(W2_vert)
    w_test = TestFunction(W2_vert)

    form_M = phi_test*phi_trial*dx
    form_D = phi_test*div(w_trial)*dx
    form_DT = div(w_test)*phi_trial*dx
    mat_M.assemble_ufl_form(form_M)
    mat_D.assemble_ufl_form(form_D)

    mat_DT_M = mat_D.transpose_matmul(mat_M)

    mat_DT_M.axpy(u, v)

    v_ufl = assemble(action(form_DT,assemble(action(form_M, u))))

    assert allclose(v_ufl,v)

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

def test_transpose(helmholtz_matrix, W2_vert, W3, form_D, form_DT,
                   pressure_expression):
    '''Test matrix transpose.

    Assemble derivative matrices :math:`D` and :math:`D^T` and check that they
    are the transposes of each other.

    :arg W2_vert: L2 pressure function space
    :arg W3: HDiv space for vertical velocity component
    :arg form_D: UFL form for weak derivative
    :arg form_DT: UFL form for transpose of weak derivative
    :arg pressure_expression: Analytical expression for pressure
    '''

    mat_D = BandedMatrix(W3,W2_vert)
    mat_DT = BandedMatrix(W2_vert,W3)

    mat_D.assemble_ufl_form(form_D)
    mat_DT.assemble_ufl_form(form_DT)

    mat_Dtranspose = mat_D.transpose()

    u = Function(W3)
    v1 = Function(W2_vert)
    v2 = Function(W2_vert)
    u.project(pressure_expression)
    mat_DT.axpy(u,v1)
    mat_Dtranspose.axpy(u,v2)

    assert allclose(v1,v2)

def test_matadd(helmholtz_matrix, W2_vert, W3, form_M, form_D, form_DT, omega,
                pressure_expression):
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
    :arg pressure_expression: Analytical expression for pressure
    '''

    u = Function(W3)
    v = Function(W3)

    u.project(pressure_expression)
    v.assign(0)

    mat_H = helmholtz_matrix

    mat_H.axpy(u, v)

    v_ufl = assemble(action(form_M, u))
    v_ufl += omega*assemble(action(form_D,assemble(action(form_DT, u))))

    assert allclose(v_ufl,v)

def test_mass_solve(W3,form_M,pressure_expression):
    '''Test LU solver with mass matrix.

    Invert the :math:`W_3` mass matrix on a :math:`W_3` for a given field and
    check that the result is correct by multiplying back by the UFL form.

    :arg W3: L2 pressure function space
    :arg form_M: pressure mass matrix
    :arg pressure_expression: Analytical expression for pressure
    '''
    u = Function(W3)
    v = Function(W3)

    u.interpolate(pressure_expression)

    mat = BandedMatrix(W3,W3)
    mat.assemble_ufl_form(form_M)

    v.assign(u)
    mat.solve(v)

    v_ufl = assemble(action(form_M, v))

    assert allclose(v_ufl,u)

def test_helmholtz_solve(helmholtz_matrix, W2_vert, W3, form_M, form_D, form_DT, omega,
                         pressure_expression):
    '''Test LU solver with the Helmholtz matrix.

    Invert the helmholtz matrix on a :math:`W_3` for a given field and
    check that the result is correct by multiplying back by the UFL form.

    :arg W2_vert: L2 pressure function space
    :arg W3: HDiv space for vertical velocity component
    :arg form_M: UFL form for pressure mass matrix
    :arg form_D: UFL form for weak derivative
    :arg form_DT: UFL form for transpose of weak derivative
    :arg omega: value of parameter omega
    :arg pressure_expression: Analytical expression for pressure
    '''
    u = Function(W3)
    v = Function(W3)

    u.interpolate(pressure_expression)

    mat = helmholtz_matrix

    v.assign(u)
    mat.solve(v)

    v_ufl = assemble(action(form_M, v))
    v_ufl += omega*assemble(action(form_D,assemble(action(form_DT, v))))

    assert allclose(v_ufl,u)

def test_helmholtz_solve_lowest_order(W3_hierarchy,
                                      W2_vert_hierarchy,
                                      omega,
                                      pressure_expression):
    '''Test tridiagonal solver with the lowet order Helmholtz matrix.

    Invert the helmholtz matrix on a :math:`W_3` for a given field and
    check that the result is correct by multiplying back by the UFL form.

    :arg W3_hierarchy: Hierarchy of lowest order W3 spaces
    :arg W2_vert: Hierarchy of lowest order W2 function spaces
    :arg omega: value of parameter omega
    :arg pressure_expression: Analytical expression for pressure
    '''
    W3 = W3_hierarchy[-1]
    W2_vert = W2_vert_hierarchy[-1]

    mat_M = BandedMatrix(W3,W3)
    mat_D = BandedMatrix(W3,W2_vert)
    mat_DT = BandedMatrix(W2_vert,W3)

    form_M = TestFunction(W3)*TrialFunction(W3)*dx
    form_D = TestFunction(W3)*div(TrialFunction(W2_vert))*dx
    form_DT = div(TestFunction(W2_vert))*TrialFunction(W3)*dx

    mat_M.assemble_ufl_form(form_M)
    mat_D.assemble_ufl_form(form_D)
    mat_DT.assemble_ufl_form(form_DT)

    # Calculate H = M + omega*D*DT
    mat_H = mat_M.matadd(mat_D.matmul(mat_DT),omega=omega)

    u = Function(W3)
    v = Function(W3)
    u.interpolate(pressure_expression)

    v.assign(u)
    mat_H.solve(v)

    v_ufl = assemble(action(form_M, v))
    v_ufl += omega*assemble(action(form_D,assemble(action(form_DT, v))))

    assert allclose(v_ufl,u)


def test_spai(W2_vert):
    '''Test sparse approximate inverse of velocity mass matrix.

    Calculate the sparse approximate inverse of the vertical velocity mass matrix
    and check that it is reasonably close to the exact inverse.

    :arg W2_vert: L2 pressure function space
    '''

    w = TestFunction(W2_vert)
    u = TrialFunction(W2_vert)
    form_Mu = dot(w,u)*dx

    # Create unit matrix
    mat_unit = BandedMatrix(W2_vert,W2_vert)
    for icol in range(len(mat_unit._data.data)):
        for i in range(mat_unit._n_row):
            mat_unit._data.data[icol][mat_unit.bandwidth*i+mat_unit.gamma_m] = 1.0

    mat_A = BandedMatrix(W2_vert,W2_vert)
    mat_A.assemble_ufl_form(form_Mu)

    mat_M = mat_A.spai(mat_A._n_row)

    mat_AM = (mat_A.matmul(mat_M)).matadd(mat_unit,omega=-1.0)

    nrm = np.linalg.norm(mat_AM._data.data)

    assert np.allclose(mat_AM._data.data, 0.0)

def test_diagonal(W2_vert):
    '''Check correct extraction of diagonal

    Construct the W2_v mass matrix and extract the diagonal. Check that
    this really matches the diagonal of the matrix.

    :arg W2_vert: vertical HDiv space
    '''
    u = TestFunction(W2_vert)
    v = TrialFunction(W2_vert) 
    ufl_form = dot(u,v)*dx
    mat = BandedMatrix(W2_vert,W2_vert)
    mat.assemble_ufl_form(ufl_form)
    mat_diag = mat.diagonal()
    diff = [(np.diag(m_dense) - m) for (m_dense,m) \
             in zip(mat.dense().dat.data,mat_diag._data.data)]
    assert np.allclose(diff, 0.0)

def test_inv_diagonal(W2_vert):
    '''Check correct extraction of inverse diagonal

    Construct the W2_v mass matrix and extract the diagonal and inverse
    diagonal. Check that the product of the two is the identity matrix

    :arg W2_vert: vertical HDiv space
    '''
    u = TestFunction(W2_vert)
    v = TrialFunction(W2_vert) 
    ufl_form = dot(u,v)*dx
    mat = BandedMatrix(W2_vert,W2_vert)
    mat.assemble_ufl_form(ufl_form)
    mat_diag = mat.diagonal()
    mat_inv_diag = mat.inv_diagonal()
    diff = [np.multiply(m_inv ,m) for (m_inv,m) \
             in zip(mat_inv_diag._data.data,mat_diag._data.data)]
    assert np.allclose(diff, 1.0)
    
def test_boundary_conditions(W2_vert_coarse,velocity_expression):
    '''Check that boundary conditions are applied correctly.

    Create a banded matrix with boundary conditions on the columns space.
    Apply it to a field and check that
    
        * It leaves the values on the boundary nodes unchanges
        * On the other nodes the result this the same as if the
          matrix without BCs was applied to a field which has the
          boundary nodes set to zero first.

    :arg W2_vert_coarse: vertical HDiv space
    :arg velocity_expression: Expression to project for velocity
    '''
    u = TestFunction(W2_vert_coarse)
    v = TrialFunction(W2_vert_coarse) 
    ufl_form = dot(u,v)*dx
    mat = BandedMatrix(W2_vert_coarse,W2_vert_coarse)
    mat.assemble_ufl_form(ufl_form)
    bcs = [DirichletBC(W2_vert_coarse, 0.0, "bottom"),
           DirichletBC(W2_vert_coarse, 0.0, "top")]
    boundary_nodes = np.concatenate([bc.nodes for bc in bcs])
    interior_nodes = [i for i in range(W2_vert_coarse.dof_count) if i not in boundary_nodes]
    # Set boundary dofs to zero and apply original matrix
    u = Function(W2_vert_coarse)
    u.project(velocity_expression)
    for bc in bcs:
        bc.apply(u)
    mat.ax(u)
    # Apply boundary conditions to matrix, and apply this new matrix
    mat.apply_vertical_bcs()
    v = Function(W2_vert_coarse)
    v.project(velocity_expression)
    mat.ax(v)
    w = Function(W2_vert_coarse)
    w.project(velocity_expression)
    # 1. Check that applying the modified matrix leaves the values on the boundary nodes
    #    unchanged
    assert np.allclose(w.dat.data[boundary_nodes] - v.dat.data[boundary_nodes], 0.0)
    # 2. Check that applying the modified matrix gives the same values on the interior
    #    nodes if the boundary nodes have been zeroed out before applying the original,
    #    full matrix.
    assert np.allclose(u.dat.data[interior_nodes] - v.dat.data[interior_nodes], 0.0)

##############################################################
# M A I N
##############################################################
if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
