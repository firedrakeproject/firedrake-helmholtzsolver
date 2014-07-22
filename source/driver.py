import sys
import os
import math
from firedrake import * 
op2.init(log_level="WARNING")
from ffc import log
log.set_level(log.ERROR)
import helmholtz
import pressuresolver
from pressuresolver import operators, smoothers, solvers, preconditioners, lumpedmass
import profile_wrapper

##########################################################
# M A I N
##########################################################
if (__name__ == '__main__'):
    # Parameters
    ref_count_coarse = 0
    nlevel = 4
    outputDir = 'output'
    solver_name = 'PETSc'
    preconditioner_name = 'Multigrid' 
    tolerance_outer = 1.E-6
    tolerance_inner = 1.E-5
    maxiter_inner=1
    maxiter_outer=20
    mu_relax = 0.95
    use_maximal_eigenvalue=False
    higher_order=True
        
    # Create mesh
    coarse_mesh = UnitIcosahedralSphereMesh(refinement_level=ref_count_coarse)
    global_normal = Expression(("x[0]","x[1]","x[2]"))

    mesh_hierarchy = MeshHierarchy(coarse_mesh,nlevel)

    for level_mesh in mesh_hierarchy:
        global_normal = Expression(("x[0]","x[1]","x[2]"))
        level_mesh.init_cell_orientations(global_normal)


    fine_level = len(mesh_hierarchy)-1
    mesh = mesh_hierarchy[fine_level]

    ncells = mesh.num_cells()
    print 'Number of cells on finest grid = '+str(ncells)
    dx = 2./math.sqrt(3.)*math.sqrt(4.*math.pi/(ncells))

    omega = 8.*0.5*dx

    if (higher_order):
        V_pressure = FunctionSpace(mesh,'DG',1)
        V_velocity = FunctionSpace(mesh,'BDFM',2)
        lumped_mass = lumpedmass.LumpedMassBDFM1(V_velocity)
    else:
        V_pressure = FunctionSpace(mesh,'DG',0)
        V_velocity = FunctionSpace(mesh,'RT',1)
        lumped_mass = lumpedmass.LumpedMassRT1(V_velocity)

    lumped_mass.test_kinetic_energy()

    # Construct preconditioner
    if (preconditioner_name == 'Jacobi'):
        operator = operators.Operator(V_pressure,V_velocity,omega)
        if (higher_order):
            preconditioner = smoothers.Jacobi_HigherOrder(operator)
        else:
            preconditioner = smoothers.Jacobi_LowestOrder(operator,
              use_maximal_eigenvalue=use_maximal_eigenvalue)
    elif (preconditioner_name == 'Multigrid'):
        V_pressure_hierarchy = FunctionSpaceHierarchy(mesh_hierarchy,'DG',0)
        V_velocity_hierarchy = FunctionSpaceHierarchy(mesh_hierarchy,'RT',1)
        operator_hierarchy = operators.OperatorHierarchy(V_pressure_hierarchy,
                                                         V_velocity_hierarchy,
                                                         omega)
        presmoother_hierarchy = \
          smoothers.SmootherHierarchy(smoothers.Jacobi_LowestOrder,
            operator_hierarchy,n_smooth=2,
            mu_relax=mu_relax,
            use_maximal_eigenvalue=use_maximal_eigenvalue)
        postsmoother_hierarchy = \
          smoothers.SmootherHierarchy(smoothers.Jacobi_LowestOrder,
            operator_hierarchy,n_smooth=2,
            mu_relax=mu_relax,
            use_maximal_eigenvalue=use_maximal_eigenvalue)
        coarsegrid_solver = smoothers.Jacobi_LowestOrder(operator_hierarchy[0])
        coarsegrid_solver.n_smooth = 1
        hmultigrid = preconditioners.hMultigrid(operator_hierarchy,
                                                presmoother_hierarchy,
                                                postsmoother_hierarchy,
                                                coarsegrid_solver)
        if (higher_order):
            operator = operators.Operator(V_pressure,V_velocity,omega)
            higherorder_presmoother = smoothers.Jacobi_HigherOrder(operator,
              mu_relax=mu_relax,
              n_smooth=2)
            higherorder_postsmoother = higherorder_presmoother
            hpmultigrid = preconditioners.hpMultigrid(hmultigrid,
                                                      operator,
                                                      higherorder_presmoother,
                                                      higherorder_postsmoother)
            preconditioner = hpmultigrid
        else:
            operator = operator_hierarchy[fine_level]
            preconditioner = hmultigrid
    else:
        print 'Unknown preconditioner: \''+prec_name+'\'.'
        sys.exit(-1)

    # Construct solver
    if (solver_name == 'Loop'):
        pressure_solver = solvers.LoopSolver(operator,
                                             preconditioner,
                                             tolerance=tolerance_inner,
                                             maxiter=maxiter_inner,
                                             verbose=2)
    elif (solver_name == 'CG'):
        pressure_solver = solvers.CGSolver(operator,
                                           preconditioner,
                                           tolerance=tolerance_inner,
                                           maxiter=maxiter_inner,
                                           verbose=2)
    elif (solver_name == 'PETSc'):
        pressure_solver = solvers.PETScSolver(operator,
                                              preconditioner,
                                              tolerance=tolerance_inner,
                                              maxiter=maxiter_inner,
                                              verbose=2)
    else:
        print 'Unknown solver: \''+solver_name+'\'.'
        sys.exit(-1)
        

    helmholtz_solver = helmholtz.PETScSolver(V_pressure,
                                             V_velocity,
                                             pressure_solver,
                                             omega,
                                             tolerance=tolerance_outer,
                                             maxiter=maxiter_outer)

    r_phi = Function(V_pressure).project(Expression('exp(-0.5*(x[0]*x[0]+x[1]*x[1])/(0.25*0.25))'))
    r_u = Function(V_velocity)
    r_u.assign(0.0)
    # Solve
    phi, w = helmholtz_solver.solve(r_phi,r_u)

    # Write output to disk
    DFile_w = File(os.path.join(outputDir,'velocity.pvd'))
    DFile_w << w
    DFile_phi = File(os.path.join(outputDir,'pressure.pvd'))
    DFile_phi << phi

