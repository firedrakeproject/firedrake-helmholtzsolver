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
ref_count = 4
omega = 0.5**ref_count
spherical = True
outputDir = 'output'
        
# Create mesh
n = 2**ref_count
if (spherical):
    mesh = UnitIcosahedralSphereMesh(refinement_level=ref_count)
    global_normal = Expression(("x[0]","x[1]","x[2]"))
    mesh.init_cell_orientations(global_normal)
else:
    mesh = UnitSquareMesh(n,n)
    
# Set up function spaces
V_pressure = FunctionSpace(mesh,'DG',0,name='P0')
V_velocity = FunctionSpace(mesh,'RT',1,name='RT1')

operator = pressuresolver.operators.Operator(V_pressure,V_velocity,omega)
preconditioner = pressuresolver.smoothers.Jacobi(operator)
pressure_solver = pressuresolver.solvers.ConjugateGradient(operator,preconditioner,tolerance=1.E-3)
helmholtz_solver = helmholtz.Solver(V_pressure,V_velocity,pressure_solver,omega,tolerance=1.E-3)
r_phi = Function(V_pressure).interpolate(Expression('(x[0]-0.4)*(x[0]-0.4)+(x[1]-0.3)*(x[1]-0.3)<0.2*0.2?1.0:0.0'))
r_u = Function(V_velocity)

# Solve
v = helmholtz_solver.solve(r_phi,r_u)
w, phi = v.split() 

# Write output to disk
DFile_w = File(os.path.join(outputDir,'velocity.pvd'))
DFile_w << w
DFile_phi = File(os.path.join(outputDir,'pressure.pvd'))
DFile_phi << phi

