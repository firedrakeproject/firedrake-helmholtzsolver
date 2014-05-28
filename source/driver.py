import os
from firedrake import *
import helmholtz
import pressuresolver
from pressuresolver import operators, smoothers

##########################################################
# M A I N
##########################################################

# Parameters
omega = 0.1
ref_count = 3
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
pressure_solver = pressuresolver.smoothers.Jacobi(operator)
helmholtz_solver = helmholtz.Solver(V_pressure,V_velocity,pressure_solver,omega)
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

