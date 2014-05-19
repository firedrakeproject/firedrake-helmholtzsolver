from firedrake import *

##########################################################
#
# Solve Helmholtz-like system in mixed formulation
#
#   \phi + \omega \div(u)      = r_\phi
#      u + \omega \grad(\phi) = r_u
#
#  in domain \Omega
#
# with boundary condition \phi = 0 on \partial\Omega
#
# The function spaces are: \phi, r_\phi \in V_2 (DG) and u, r_u in V_1
#
##########################################################

class HelmholtzSolver:

  ##########################################################
  # Constructor
  ##########################################################
  def __init__(self,omega,ref_count,spherical=False):
    self.omega = omega
    self.ref_count = ref_count
    self.spherical = spherical
    # Create mesh
    n = 2**self.ref_count
    if (self.spherical):
      self.mesh = UnitIcosahedralSphereMesh(refinement_level=ref_count)
    else:
      self.mesh = UnitSquareMesh(n,n)
    # Set up function spaces
    self.V_pressure = FunctionSpace(self.mesh,'DG',0,name='P0')
    self.V_velocity = FunctionSpace(self.mesh,'RT',1,name='RT1')
    self.V_mixed = self.V_velocity*self.V_pressure
    self.u, self.phi = TrialFunctions(self.V_mixed)
    self.w, self.psi = TestFunctions(self.V_mixed)
    # Define bilinear form 
    self.a_outer = (  self.psi*self.phi \
                    + self.omega*self.psi*div(self.u) \
                    + inner(self.w,self.u) \
                    - self.omega*div(self.w)*self.phi)*dx
  
  ##########################################################
  # Solve for a particular RHS
  ##########################################################
  def solve(self,r_phi,r_u):
    L = (self.psi*r_phi + inner(self.w,r_u))*dx
    v_mixed = Function(self.V_mixed)
    solve(self.a_outer == L,v_mixed,
          solver_parameters={'ksp_type':'cg',
                             'pc_type':'fieldsplit',
                             'pc_fieldsplit_type':'schur',
                             'pc_fieldsplit_schur_fact_type':'FULL',
                             'fieldsplit_P0_ksp_type':'cg',
                             'fieldsplit_RT1_ksp_type':'cg'})
    return v_mixed
