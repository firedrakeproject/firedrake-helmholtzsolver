from firedrake import *
from pressuresolver.operators import *
from pressuresolver.hierarchy import *
from pressuresolver.mu_tilde import *
from pressuresolver.smoothers import *
from pressuresolver.preconditioners import *
from pressuresolver.solvers import *
from pressuresolver.ksp_monitor import *
import numpy as np
import pytest
from fixtures import *

op2.init(log_level="WARNING")
from ffc import log
log.set_level(log.ERROR)

def test_pressuresolve(W3,
                       W2,
                       W2_horiz,
                       W2_vert,
                       Wb,
                       W3_hierarchy,
                       W2_hierarchy,
                       W2_horiz_hierarchy,
                       W2_vert_hierarchy,
                       Wb_hierarchy,
                       pressure_expression):
    '''Test pressure solver at next-to-lowest order. This tests
    the hp-multigrid preconditioner.

    :arg W3: (Higher order) pressure space
    :arg W2: (Higher order) velocity space
    :arg W2_horiz: Horizontal component of (higher order) velocity space
    :arg W2_vert: Vertical component of (higher order) velocity space
    :arg Wb: (Higher order) buoyancy space
    :arg W3_hierarchy: Pressure space hierarchy
    :arg W2_hierarchy: Velocity space hierarchy
    :arg W2_horiz_hierarchy: Horizontal velocity component hierarchy
    :arg W2_vert_hierarchy: Vertical velocity component hierarchy
    :arg Wb_hierarchy: Buoyancy space hierarchy
    :arg pressure_expression: analytical expression for RHS
    '''

    mesh = W3.mesh()
    ncells = mesh.cell_set.size

    print 'Number of cells on finest grid = '+str(ncells)
    dx = 2./math.sqrt(3.)*math.sqrt(4.*math.pi/(ncells))
   
    omega_c = 8.*0.5*dx
    omega_N = 0.5

    # Construct h-multigrid preconditioner
    operator_Hhat_hierarchy = HierarchyContainer(Operator_Hhat,
      zip(W3_hierarchy,
          W2_horiz_hierarchy,
          W2_vert_hierarchy),
      omega_c,
      omega_N)

    presmoother_hierarchy = HierarchyContainer(Jacobi,
      zip(operator_Hhat_hierarchy))

    postsmoother_hierarchy = HierarchyContainer(Jacobi,
      zip(operator_Hhat_hierarchy))

    coarsegrid_solver = postsmoother_hierarchy[0]

    hmultigrid = hMultigrid(W3_hierarchy,
      operator_Hhat_hierarchy,
      presmoother_hierarchy,
      postsmoother_hierarchy,
      coarsegrid_solver)

    # Higher order operator \hat{H} and smoothers

    operator_Hhat = Operator_Hhat(W3,W2_horiz,W2_vert,
                                  omega_c,omega_N)

    presmoother = Jacobi(operator_Hhat)
    postsmoother = Jacobi(operator_Hhat)

    # hp-multigrid preconditioner
    preconditioner = hpMultigrid(hmultigrid,
                                 operator_Hhat,
                                 presmoother,
                                 postsmoother)

    # Higher order operator H

    mutilde = Mutilde(W2,Wb,omega_N)
    operator_H = Operator_H(W3,W2,mutilde,omega_c)

    ksp_monitor = KSPMonitor()
    ksp_type = 'cg'

    solver = PETScSolver(operator_H,
      preconditioner,
      ksp_type,
      ksp_monitor,
      tolerance=1.E-5,
      maxiter=30)

    b = Function(W3).project(pressure_expression)
    phi = Function(W3)

    solver.solve(b,phi)

    assert (ksp_monitor.its < 20)

def test_pressuresolve_lowestorder(W3_hierarchy,
                                   W2_hierarchy,
                                   W2_horiz_hierarchy,
                                   W2_vert_hierarchy,
                                   Wb_hierarchy,
                                   pressure_expression):
    '''Test pressure solver at lowest order. This mainly tests the
    h-multigrid preconditioner.

    :arg W3_hierarchy: Pressure space hierarchy
    :arg W2_hierarchy: Velocity space hierarchy
    :arg W2_horiz_hierarchy: Horizontal velocity component hierarchy
    :arg W2_vert_hierarchy: Vertical velocity component hierarchy
    :arg Wb_hierarchy: Buoyancy space hierarchy
    :arg pressure_expression: analytical expression for RHS
    '''

    W3 = W3_hierarchy[-1]
    W2 = W2_hierarchy[-1]
    Wb = Wb_hierarchy[-1]

    mesh = W3.mesh()
    ncells = mesh.cell_set.size

    print 'Number of cells on finest grid = '+str(ncells)
    dx = 2./math.sqrt(3.)*math.sqrt(4.*math.pi/(ncells))
   
    omega_c = 8.*0.5*dx
    omega_N = 0.5

    mutilde = Mutilde(W2,Wb,omega_N)

    operator_H = Operator_H(W3,W2,mutilde,omega_c)

    operator_Hhat_hierarchy = HierarchyContainer(Operator_Hhat,
      zip(W3_hierarchy,
          W2_horiz_hierarchy,
          W2_vert_hierarchy),
      omega_c,
      omega_N)

    presmoother_hierarchy = HierarchyContainer(Jacobi,
      zip(operator_Hhat_hierarchy))

    postsmoother_hierarchy = HierarchyContainer(Jacobi,
      zip(operator_Hhat_hierarchy))

    coarsegrid_solver = postsmoother_hierarchy[0]

    preconditioner = hMultigrid(W3_hierarchy,
      operator_Hhat_hierarchy,
      presmoother_hierarchy,
      postsmoother_hierarchy,
      coarsegrid_solver)

    ksp_monitor = KSPMonitor()
    ksp_type = 'cg'

    solver = PETScSolver(operator_H,
      preconditioner,
      ksp_type,
      ksp_monitor,
      tolerance=1.E-5,
      maxiter=10)

    b = Function(W3).project(pressure_expression)
    phi = Function(W3)

    solver.solve(b,phi)

    assert (ksp_monitor.its < 20)


##############################################################
# M A I N
##############################################################
if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
