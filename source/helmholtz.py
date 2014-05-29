from firedrake import *

##########################################################
#
# Solve Helmholtz-like system in mixed formulation
#
#   \phi + \omega \div(u)      = r_\phi
#      u + \omega \grad(\phi)  = r_u
#
#  in domain \Omega
#
##########################################################

class Solver:

##########################################################
# Constructor
##########################################################
    def __init__(self,V_pressure,V_velocity,pressure_solver,omega,
                 maxiter=100,
                 tolerance=1.E-6,
                 verbose=2):
        self.omega = omega
        self.maxiter = maxiter
        self.tolerance = tolerance
        self.V_pressure = V_pressure
        self.V_velocity = V_velocity
        self.verbose = verbose
        self.pressure_solver = pressure_solver
        # Set up test- and trial function spaces
        self.u = TrialFunction(self.V_velocity)
        self.phi = TrialFunction(self.V_pressure)
        self.w = TestFunction(self.V_velocity)
        self.psi = TestFunction(self.V_pressure)
        # Extract lumped mass
        self.lumped_mass = pressure_solver.operator.lumped_mass

##########################################################
# Solve for a particular RHS
##########################################################
    def solve(self,r_phi,r_u):
        if (self.verbose > 0):
            print ' === Helmholtz solver ==='
        # Fields for solution
        u = Function(self.V_velocity)
        phi = Function(self.V_pressure)
        u.assign(0.0)
        phi.assign(0.0)
        # Pressure correction
        d_phi = Function(self.V_pressure)
        # Calculate initial residual
        res_norm_0 = sqrt(assemble((dot(r_u,r_u)+r_phi*r_phi)*dx))
        if (self.verbose > 0):
            print ' initial outer residual : '+('%e' % res_norm_0)
        # Residual correction
        dR_phi = r_phi
        dR_u = r_u
        for i in range(1,self.maxiter+1):
            # Construct RHS for linear (pressure) solve
            F_pressure = assemble(self.psi*(dR_phi - self.omega*div(dR_u))*dx)
            # Solve for pressure correction
            d_phi.assign(0.0)
            self.pressure_solver.solve(F_pressure,d_phi)
            # Update solution with correction
            # phi -> phi + d_phi
            phi += d_phi
            # u -> u + omega*(M_u^{lumped})^{-1}*grad(d_phi)
            grad_dphi = assemble(div(self.w)*d_phi*dx)
            self.lumped_mass.divide(grad_dphi)
            u += dR_u + self.omega * grad_dphi
            # Calculate current residual
            r_phi_cur, r_u_cur = self.residual(phi,u)
            # Update residual correction
            dR_phi = r_phi - r_phi_cur
            dR_u   = r_u - r_u_cur
            # Check for convergence and print out residual norm
            res_norm = sqrt(assemble((dot(dR_u,dR_u)+dR_phi*dR_phi)*dx))
            if (self.verbose > 1):
                print ' i = '+('%4d' % i) +  \
                      ' : '+('%8.4e' % res_norm) + \
                      ' [ '+('%8.4e' % (res_norm/res_norm_0))+' ] '
            if (res_norm/res_norm_0 < self.tolerance):
                break
        if (self.verbose > 0):
            if (res_norm/res_norm_0 < self.tolerance):
                print ' Outer loop converged after '+str(i)+' iterations.'
            else:
                print ' Outer loop failed to converge after '+str(self.maxiter)+' iterations.'
        return u, phi    

##########################################################
# Solver using firedrake's built-in PETSc solvers
##########################################################
    def solve_petsc(self,r_phi,r_u):
        V_mixed = self.V_velocity*self.V_pressure
        psi_mixed, w_mixed = TestFunctions(V_mixed)
        phi_mixed, u_mixed = TrialFunctions(V_mixed)
        # Solve using PETSc solvers
        v_mixed = Function(V_mixed)
        # Define bilinear form 
        a_outer = (  psi_mixed*phi_mixed \
                   + self.omega*psi_mixed*div(u_mixed) \
                   + inner(w_mixed,u_mixed) \
                   - self.omega*div(w_mixed)*phi_mixed)*dx
        L = (psi_mixed*r_phi + inner(w_mixed,r_u))*dx
        solve(a_outer == L,v_mixed,
              solver_parameters={'ksp_type':'cg',
                                 'pc_type':'fieldsplit',
                                 'pc_fieldsplit_type':'schur',
                                 'pc_fieldsplit_schur_fact_type':'FULL',
                                 'fieldsplit_P0_ksp_type':'cg',
                                 'fieldsplit_RT1_ksp_type':'cg'})
        return v_mixed.split()

##########################################################
# Calculate residual for full equation
##########################################################
    def residual(self,phi,u):
        a_phi_rhs = ( inner(self.psi,phi) \
                    + self.omega*self.psi*div(u) )*dx
        a_u_rhs = ( inner(self.w,u) \
                  - self.omega*div(self.w)*phi )*dx
        a_phi_mass = self.phi*self.psi*dx
        a_u_mass = inner(self.w,self.u)*dx
        r_phi = Function(self.V_pressure)
        r_u = Function(self.V_velocity)
        solve(a_phi_mass == a_phi_rhs,r_phi,solver_parameters={'ksp_type':'cg'})
        solve(a_u_mass == a_u_rhs,r_u,solver_parameters={'ksp_type':'cg'})
        return (r_phi,r_u)
