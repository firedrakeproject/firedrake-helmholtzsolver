import os
from firedrake import *
op2.init(log_level="WARNING")
from ffc import log
log.set_level(log.ERROR)
import helmholtz
import pressuresolver
from pressuresolver import operators, smoothers, solvers

##########################################################
# M A I N
##########################################################

# Parameters
ref_count_coarse = 2
nlevel = 3
omega = 0.5**(ref_count_coarse+nlevel)
spherical = True
outputDir = 'output'
ignore_mass_lumping = False
tolerance_outer = 1.E-6
tolerance_inner = 1.E-6
maxiter=10
        
# Create mesh
if (spherical):
    mesh = UnitIcosahedralSphereMesh(refinement_level=ref_count_coarse)
    global_normal = Expression(("x[0]","x[1]","x[2]"))
else:
    n = 2**ref_count_coarse
    mesh = UnitSquareMesh(n,n)

mesh_hierarchy = MeshHierarchy(mesh,nlevel)
if (spherical):
    for level_mesh in mesh_hierarchy:
        global_normal = Expression(("x[0]","x[1]","x[2]"))
        level_mesh.init_cell_orientations(global_normal)


finelevel = -1
fine_mesh = mesh_hierarchy[finelevel]
V_pressure_hierarchy = FunctionSpaceHierarchy(mesh_hierarchy,'DG',0)
V_velocity_hierarchy = FunctionSpaceHierarchy(mesh_hierarchy,'RT',1)

# Set up function spaces
V_pressure = V_pressure_hierarchy[finelevel]
V_velocity = V_velocity_hierarchy[finelevel]

operator_hierarchy = pressuresolver.operators.OperatorHierarchy(V_pressure_hierarchy,
                                                       V_velocity_hierarchy,
                                                       omega,
                                                       ignore_mass_lumping=ignore_mass_lumping)
operator = operator_hierarchy[finelevel]
preconditioner_hierarchy = pressuresolver.smoothers.JacobiHierarchy(operator_hierarchy)
preconditioner = preconditioner_hierarchy[finelevel]

pressure_solver = pressuresolver.solvers.ConjugateGradient(operator,preconditioner,
                                                           tolerance=tolerance_inner,
                                                           verbose=1)
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

