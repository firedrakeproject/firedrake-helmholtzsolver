from firedrake import *
from bandedmatrix import *
import numpy as np
import pytest
from fixtures import *

def test_W3_mass_action(W3,pressure_expression):
    '''Test DG mass matrix action in place.

    Calculate the action of the :math:`W_3` mass matrix on a :math:`W_3` field both using
    the locally assembled matrix class and UFL; compare the results.

    :arg W3: L2 pressure function space
    :arg pressure_expression: Analytical expression for pressure
    '''
    u = Function(W3)

    u.interpolate(pressure_expression)

    mat = LocallyAssembledMatrix(W3,W3)
    phi_test = TestFunction(W3)
    phi_trial = TrialFunction(W3)
    form = phi_test*phi_trial*dx
    mat.assemble_ufl_form(form)
    v = mat.ax(u)

    v_ufl = assemble(action(form, u))
    assert np.allclose(norm(assemble(v_ufl - v)), 0.0)

def test_W2_mass_action(W2,velocity_expression):
    '''Test HDiv mass matrix action in place.

    Calculate the action of the :math:`W_2` mass matrix on a :math:`W_2` field both using
    the locally assembled matrix class and UFL; compare the results.

    :arg W2: HDiv function space
    :arg velocity_expression: Analytical expression for velocity
    '''
    u = Function(W2)

    u.project(velocity_expression)

    u_test = TestFunction(W2)
    u_trial = TrialFunction(W2)
    form = dot(u_test,u_trial)*dx
    mat = LocallyAssembledMatrix(W2,W2,form)

    v = mat.ax(u)

    v_ufl = assemble(action(form, u))
    assert np.allclose(norm(assemble(v_ufl - v)), 0.0)

def test_matmul(W2,W3,velocity_expression):
    '''Matrix matrix product.

        Calculate :math:`v=D^T Du` where :math:`D` is the weak derivative by assembling
        both the matrices for :math:`D` and :math:`D^T` and comparing this to the
        expression obtained by evaluating the expression in UFL.
    
    :arg W3: L2 discontinuous function space
    :arg W2: HDiv function space
    :arg velocity_expression: Analytical expression for velocity
    '''
    u = Function(W2)
    phi = Function(W3)
    u.project(velocity_expression)
    
    u_test = TestFunction(W2)
    u_trial = TrialFunction(W2)
    phi_test = TestFunction(W3)
    phi_trial = TrialFunction(W3)

    form_D = phi_test*div(u_trial)*dx
    form_DT = div(u_test)*phi_trial*dx
    
    mat_D = LocallyAssembledMatrix(W3,W2,form_D)
    mat_DT = LocallyAssembledMatrix(W2,W3,form_DT)
    mat_DT_D = mat_DT.matmul(mat_D)
    
    v = mat_DT_D.ax(u)

    phi = assemble(action(form_D,u))
    v_ufl = assemble(action(form_DT,phi))

    assert np.allclose(norm(assemble(v_ufl - v)), 0.0)

def test_matadd(W2,W3,velocity_expression):
    '''Matrix matrix addition.

        Calculate :math:`v=(M_u + \omega^2 D^T D)u` where :math:`D` is the weak derivative
        and :math:`M_u` is the velocity mass matrix.
        This is achieved by assembling the matrices :math:`M_u`, :math:`D` and :math:`D^T`
        and comparing this to the expression obtained by evaluating the expression in UFL.
    
    :arg W3: L2 discontinuous function space
    :arg W2: HDiv function space
    :arg velocity_expression: Analytical expression for velocity
    '''
    u = Function(W2)
    phi = Function(W3)
    u.project(velocity_expression)
    omega = 2.0
    
    u_test = TestFunction(W2)
    u_trial = TrialFunction(W2)
    phi_test = TestFunction(W3)
    phi_trial = TrialFunction(W3)

    form_Mu = dot(u_test,u_trial)*dx
    form_D = phi_test*div(u_trial)*dx
    form_DT = div(u_test)*phi_trial*dx
    
    mat_Mu = LocallyAssembledMatrix(W2,W2,form_Mu)
    mat_D = LocallyAssembledMatrix(W3,W2,form_D)
    mat_DT = LocallyAssembledMatrix(W2,W3,form_DT)
    mat_sum = mat_Mu.matadd(mat_DT.matmul(mat_D),omega)
    
    v = mat_sum.ax(u)

    phi = assemble(action(form_D,u))
    v_ufl = assemble(action(form_Mu,u))+omega*assemble(action(form_DT,phi))

    assert np.allclose(norm(assemble(v_ufl - v)), 0.0)

def test_inverse(W3,pressure_expression):
    '''Test DG mass matrix action in place.

    Calculate the action of the :math:`W_3` mass matrix on a :math:`W_3` field both using
    the locally assembled matrix class and UFL; compare the results.

    :arg W3: L2 pressure function space
    :arg pressure_expression: Analytical expression for pressure
    '''
    b = Function(W3)
    u_ufl = Function(W3)

    b.interpolate(pressure_expression)

    mat = LocallyAssembledMatrix(W3,W3)
    phi_test = TestFunction(W3)
    phi_trial = TrialFunction(W3)
    form = phi_test*phi_trial*dx

    mat.assemble_ufl_form(form)
    mat_inv = mat.inverse()
    solve(assemble(form), u_ufl, b)
    u = mat_inv.ax(b)

    assert np.allclose(norm(assemble(u_ufl - u)), 0.0)


