from lumpedmass import *

##########################################################
# Schur complement operator class
##########################################################
class Operator(object):

##########################################################
# Constructor
##########################################################
    def __init__(self,V_pressure,V_velocity,omega,
                 ignore_mass_lumping=False):
        self.omega = omega
        self.ignore_mass_lumping = ignore_mass_lumping
        self.V_velocity = V_velocity
        self.V_pressure = V_pressure
        self.w = TestFunction(self.V_velocity)
        self.psi = TestFunction(self.V_pressure)
        self.dx = self.V_pressure.mesh()._dx
        self.lumped_mass = LumpedMass(self.V_velocity,self.ignore_mass_lumping)

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
        B_phi = assemble(div(self.w)*phi*self.dx)
        # divide by lumped velocity mass matrix
        self.lumped_mass.divide(B_phi)
        # Calculate action of B^T
        BT_B_phi = assemble(self.psi*div(B_phi)*self.dx)
        # Calculate action of pressure mass matrix
        M_phi = assemble(self.psi*phi*self.dx)
        return assemble(M_phi + self.omega**2*BT_B_phi)

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
# Operator hierarchy
##########################################################
class OperatorHierarchy(object):

##########################################################
# Constructor
##########################################################
    def __init__(self,V_pressure_hierarchy,V_velocity_hierarchy,omega,
                 ignore_mass_lumping=False):
        self.ignore_mass_lumping = ignore_mass_lumping
        self.omega = omega
        self.V_pressure_hierarchy = V_pressure_hierarchy
        self.V_velocity_hierarchy = V_velocity_hierarchy
        self._hierarchy = [Operator(V_pressure,V_velocity,
                           self.omega,self.ignore_mass_lumping)
                           for (V_pressure,V_velocity) in zip(self.V_pressure_hierarchy,
                                                              self.V_velocity_hierarchy)]

##########################################################
# Get item
##########################################################
    def __getitem__(self,index):
        return self._hierarchy[index]

##########################################################
# Number of levels
##########################################################
    def __len__(self):
        return len(self._hierarchy)

