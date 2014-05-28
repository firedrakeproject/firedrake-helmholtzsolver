from lumpedmass import *

##########################################################
# Schur complement operator class
##########################################################
class Operator(object):

##########################################################
# Constructor
##########################################################
    def __init__(self,V_pressure,V_velocity,omega):
        self.omega = omega
        self.V_velocity = V_velocity
        self.V_pressure = V_pressure
        self.w = TestFunction(self.V_velocity)
        self.psi = TestFunction(self.V_pressure)
        self.lumped_mass = LumpedMass(self.V_velocity)

##########################################################
# Apply matrix
#
#  y = A.x with
#
# A := M_phi + omega^2 *B^T * (M_u^{lumped})^{-1} * B 
#
# where B is the FE matrix of the gradient and
# B^T is the FE matrix of the divergence operator
#
##########################################################
    def apply(self,phi):
        # Calculate action of B
        B_phi = assemble(div(self.w)*phi*dx)
        # divide by lumped velocity mass matrix
        self.lumped_mass.divide(B_phi)
        # Calculate action of B^T
        BT_B_phi = assemble(self.psi*div(B_phi)*dx)
        # Calculate action of pressure mass matrix
        M_phi = assemble(self.psi*phi*dx)
        return M_phi + self.omega**2*BT_B_phi

##########################################################
#
# Calculate residual 
#
# b - A.phi for the Schur complement matrix
#
##########################################################
    def residual(self,b,phi):
        return assemble(b - self.apply(phi))

##########################################################
# Abstract inverse operator base class
##########################################################
class InverseOperator(object):

##########################################################
# Constructor
##########################################################
    def __init__(self,operator):
        self.operator = operator
        self.V_velocity = operator.V_velocity
        self.V_pressure = operator.V_pressure
        self.omega = operator.omega

##########################################################
# Solve
##########################################################
    def solve(self,b,phi):
        raise NotImplementedError

##########################################################
# Solve approximately
##########################################################
    def solveApprox(self,b,phi):
        raise NotImplementedError

