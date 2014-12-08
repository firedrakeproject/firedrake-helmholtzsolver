from firedrake import *
from pressuresolver.operators3d import *
from pressuresolver.hierarchy import *
from pressuresolver.mu_tilde import *
from pressuresolver.smoothers3d import *
from pressuresolver.preconditioners3d import *
from pressuresolver.solvers3d import *
from pressuresolver.ksp_monitor import *
import numpy as np
import pytest
from fixtures import *

op2.init(log_level="WARNING")
from ffc import log
log.set_level(log.ERROR)


def test_pressuresolve_lowest_order(W3_hierarchy,
                                    W2_hierarchy,
                                    W2_horiz_hierarchy,
                                    W2_vert_hierarchy,
                                    Wb_hierarchy,
                                    pressure_expression):
    '''Test pressure solver`

    :arg W3_hierarchy: Pressure space hierarchy
    :arg W2_hierarchy: Velocity space hierarchy
    :arg W2_horiz_hierarchy: Horizontal velocity component hierarchy
    :arg W2_vert_hierarchy: Vertical velocity component hierarchy
    :arg Wb_hierarchy: Buoyancy space hierarchy
    :arg pressure_expression: analytical expression for RHS
    '''

    if (W3_hierarchy == None):
        return

    W3 = W3_hierarchy[-1]

    omega_N = 0.5
    omega_c = 0.5

    mutilde_hierarchy = HierarchyContainer(Mutilde,
      zip(W3_hierarchy,
          W2_hierarchy,
          Wb_hierarchy),
      omega_N)

    operator_H_hierarchy = HierarchyContainer(Operator_H,
      zip(W3_hierarchy,
          W2_hierarchy,
          mutilde_hierarchy),
      omega_c)

    operator_H = operator_H_hierarchy[-1]

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
      ksp_monitor)

    b = Function(W3).project(pressure_expression)
    phi = Function(W3)

    solver.solve(b,phi)

    assert True

##############################################################
# M A I N
##############################################################
if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
