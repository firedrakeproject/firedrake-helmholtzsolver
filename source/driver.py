import os
from firedrake import *
from helmholtzsolver import *

##########################################################
# M A I N
##########################################################

# Parameters
omega = 1.0
ref_count = 5
spherical = False
outputDir = 'output'

solver = HelmholtzSolver(omega,ref_count,spherical)
V_pressure = solver.V_pressure
V_velocity = solver.V_velocity
r_phi = Function(V_pressure).interpolate(Expression('sin(pi*x[0])*sin(pi*x[1])'))
r_u = Function(V_velocity).project(Expression(('0.0','0.0')))

# Solve
v = solver.solve(r_phi,r_u)
w, phi = v.split() 

# Write output to disk
DFile_w = File(os.path.join(outputDir,'velocity.pvd'))
DFile_w << w
DFile_phi = File(os.path.join(outputDir,'pressure.pvd'))
DFile_phi << phi

