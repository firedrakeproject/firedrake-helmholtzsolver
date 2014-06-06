import os
from firedrake import *
op2.init(log_level="WARNING")
from ffc import log
log.set_level(log.ERROR)
import helmholtz
import pressuresolver
from pressuresolver import operators, smoothers, solvers, preconditioners

##########################################################
# M A I N
##########################################################

# Parameters
ref_count_coarse = 2
nlevel = 4
omega = 0.5**(ref_count_coarse+nlevel)
spherical = True
outputDir = 'output'
ignore_mass_lumping = False
solver_name = 'Loop'
preconditioner_name = 'Multigrid' 
tolerance_outer = 1.E-6
tolerance_inner = 1.E-6
maxiter=10
        
# Create mesh
if (spherical):
    coarse_mesh = UnitIcosahedralSphereMesh(refinement_level=ref_count_coarse)
    global_normal = Expression(("x[0]","x[1]","x[2]"))
else:
    n = 2**ref_count_coarse
    coarse_mesh = UnitSquareMesh(n,n)

mesh_hierarchy = MeshHierarchy(coarse_mesh,nlevel)

if (spherical):
    for level_mesh in mesh_hierarchy:
        global_normal = Expression(("x[0]","x[1]","x[2]"))
        level_mesh.init_cell_orientations(global_normal)


fine_level = len(mesh_hierarchy)-1
mesh = mesh_hierarchy[fine_level]

# Construct preconditioner
if (preconditioner_name == 'Jacobi'):
    V_pressure = FunctionSpace(mesh,'DG',0)
    V_velocity = FunctionSpace(mesh,'RT',1)
    operator = pressuresolver.operators.Operator(V_pressure,V_velocity,
                                                 omega,
                                                 ignore_mass_lumping=ignore_mass_lumping)
    preconditioner = pressuresolver.smoothers.Jacobi(operator)
elif (preconditioner_name == 'Multigrid'):
    V_pressure_hierarchy = FunctionSpaceHierarchy(mesh_hierarchy,'DG',0)
    V_velocity_hierarchy = FunctionSpaceHierarchy(mesh_hierarchy,'RT',1)
    operator_hierarchy = pressuresolver.operators.OperatorHierarchy(V_pressure_hierarchy,
                                                                    V_velocity_hierarchy,
                                                                    omega,
                                                                    ignore_mass_lumping=ignore_mass_lumping)
    operator = operator_hierarchy[fine_level]

    presmoother_hierarchy = pressuresolver.smoothers.JacobiHierarchy(operator_hierarchy)
    postsmoother_hierarchy = presmoother_hierarchy
    coarsegrid_solver = pressuresolver.smoothers.Jacobi(operator_hierarchy[0])
    coarsegrid_solver.n_smooth = 4
    preconditioner = pressuresolver.preconditioners.Multigrid(operator_hierarchy,
                                                              presmoother_hierarchy,
                                                              postsmoother_hierarchy,
                                                              coarsegrid_solver,
                                                              tolerance=tolerance_inner)
    V_pressure = V_pressure_hierarchy[fine_level]
    V_velocity = V_velocity_hierarchy[fine_level]
else:
    print 'Unknown preconditioner: \''+prec_name+'\'.'
    sys.exit(-1)

# Construct solver
if (solver_name == 'Loop'):
    pressure_solver = pressuresolver.solvers.LoopSolver(operator,
                                                        preconditioner,
                                                        tolerance=tolerance_inner,
                                                        verbose=2)
elif (solver_name == 'CG'):
    pressure_solver = pressuresolver.solvers.ConjugateGradient(operator,
                                                               preconditioner,
                                                               tolerance=tolerance_inner,
                                                               verbose=2)
else:
    print 'Unknown solver: \''+solver_name+'\'.'
    sys.exit(-1)
    

helmholtz_solver = helmholtz.Solver(V_pressure,V_velocity,pressure_solver,omega,
                                    tolerance=tolerance_outer,
                                    maxiter=maxiter)

r_phi = Function(V_pressure).project(Expression('exp(-0.5*(x[0]*x[0]+x[1]*x[1])/(0.25*0.25))'))
r_u = Function(V_velocity)
r_u.assign(0.0)
# Solve
w, phi = helmholtz_solver.solve(r_phi,r_u)

# Write output to disk
DFile_w = File(os.path.join(outputDir,'velocity.pvd'))
DFile_w << w
DFile_phi = File(os.path.join(outputDir,'pressure.pvd'))
DFile_phi << phi

