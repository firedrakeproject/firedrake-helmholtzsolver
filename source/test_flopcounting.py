from firedrake import *
from flop_counter import *
from pyop2 import performance_summary

# Construct mesh and function space
mesh = UnitSquareMesh(8,8)

#########################################################
### Testcase I: primal Helmholtz operator
#########################################################

# Construct function spaces, function and testfunction
V = FunctionSpace(mesh, "CG", 1)

# Construct function and test-function
u = Function(V)
u_test = TestFunction(V)

ufl_form_helmholtz = (dot(grad(u_test), grad(u)) + u_test*u) * dx
flop_counter = FlopCounter1Form(ufl_form_helmholtz,'helmholtz_2d')
print "=== 2d Helmholtz ==="
print "number of FLOPs = ", flop_counter.flops
print

#########################################################
### Testcase II: gradient operator on 2d mesh
#########################################################

# Construct function spaces, function and testfunction
V2 = FunctionSpace(mesh, "RT", 1)
V3 = FunctionSpace(mesh, "DG", 0)

p = Function(V3)
v_test = TestFunction(V2)

# UFL form
ufl_form_2dgrad = div(v_test)*p*dx

flop_counter = FlopCounter1Form(ufl_form_2dgrad,'grad_2d')
print "=== 2d gradient ==="
print "number of FLOPs = ", flop_counter.flops
print

#########################################################
### Testcase III: gradient operator on extruded 2+1d mesh
#########################################################

# extruded mesh
nlayers=16
extruded_mesh = ExtrudedMesh(mesh,nlayers)

# Lowest order horizontal elements
U1 = FiniteElement('RT',triangle,1)
U2 = FiniteElement('DG',triangle,0)
# Lowest order vertical elements
V0 = FiniteElement('CG',interval,1)
V1 = FiniteElement('DG',interval,0)
# Lowest order product elements
W2_elt = HDiv(OuterProductElement(U1,V1)) + HDiv(OuterProductElement(U2,V0))
W3_elt = OuterProductElement(U2,V1)

W2 = FunctionSpace(extruded_mesh,W2_elt)
W3 = FunctionSpace(extruded_mesh,W3_elt)

p = Function(W3)
v_test = TestFunction(W2)

# UFL form
ufl_form_3dgrad = div(v_test)*p*dx

flop_counter = FlopCounter1Form(ufl_form_3dgrad,'grad_3d')
print "=== (2+1)d gradient ==="
print "number of FLOPs = ", flop_counter.flops
print
performance_summary()
