from firedrake import *
'''Solve Helmholtz system in mixed formulation.

This module contains the :class:`.Solver` for solving a Helmholtz system 
using finite elements.
'''

class Solver:
    '''Solver for the Helmholtz system

        .. math::
  
            \phi + \omega (\\nabla\cdot\phi^*\\vec{u}) = r_\phi

            \\vec{u} - \omega \\nabla{\phi} = \\vec{r}_u
        in the mixed finite element formulation.

        :arg V_pressure: Function space for pressure field :math:`\phi`
        :arg V_velocity: Function space for velocity field :math:`\\vec{u}`
        :arg pressure_solver: Solver for Schur complement pressure system, e.g. :class:`.LoopSolver` and :class:`.ConjugateGradient`.
        :arg omega: Positive real number
        :arg maxiter: Maximal number of iterations for outer iteration
        :arg tolerance: Tolerance for outer iteration
        :arg verbose: Verbosity level (0=no output, 1=minimal output, 2=show convergence rates)
    '''
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

    def solve(self,r_phi,r_u):
        '''Solve Helmholtz system using nested iteration.

        Solve the mixed Helmholtz system for right hand sides :math:`r_\phi` and :math:`r_u`.
        The full velocity mass matrix is used in the outer iteration and the pressure correction
        system is solved with the specified :class:`pressure_solver` in an inner iteration.
        See `Notes in LaTeX <./FEMmultigrid.pdf>`_ for more details of the algorithm.

            :arg r_phi: right hand side for pressure equation, function in :math:`DG` space.
            :arg r_u: right hand side for velocity equation, function in :math:`H(div)` space.
        '''
        if (self.verbose > 0):
            print ' === Helmholtz solver ==='
        # Fields for solution
        u = Function(self.V_velocity)
        du = Function(self.V_velocity)
        phi = Function(self.V_pressure)
        F_pressure = Function(self.V_pressure)
        u.assign(0.0)
        phi.assign(0.0)
        # Pressure correction
        d_phi = Function(self.V_pressure)
        # Residual correction
        Mr_phi = assemble(self.psi*r_phi*dx)
        Mr_u = assemble(dot(self.w,r_u)*dx)
        dMr_phi = Function(self.V_pressure)
        dMr_u = Function(self.V_velocity)
        dMinvMr_u = Function(self.V_velocity)
        dMr_phi.assign(Mr_phi)
        dMr_u.assign(Mr_u)
        # Calculate initial residual
        res_norm = self.residual_norm(dMr_phi,dMr_u) 
        res_norm_0 = res_norm 
        if (self.verbose > 0):
            print ' initial outer residual : '+('%e' % res_norm_0)
        for i in range(1,self.maxiter+1):
            # Construct RHS for linear (pressure) solve
            dMinvMr_u.assign(dMr_u)
            self.lumped_mass.divide(dMinvMr_u)
            F_pressure.assign(dMr_phi - self.omega*assemble(self.psi*div(dMinvMr_u)*dx))
            # Solve for pressure correction
            d_phi.assign(0.0)
            self.pressure_solver.solve(F_pressure,d_phi)
            # Update solution with correction
            # phi -> phi + d_phi
            phi += d_phi
            # u -> u + (M_u^{lumped})^{-1}*(R_u + omega*grad(d_phi))
            grad_dphi = assemble(div(self.w)*d_phi*dx)
            du.assign(dMr_u + self.omega * grad_dphi)
            self.lumped_mass.divide(du)
            u += du
            # Calculate current residual
            Mr_phi_cur = assemble((self.psi*phi + self.omega*self.psi*div(u))*dx)
            Mr_u_cur = assemble(( inner(self.w,u) - self.omega*div(self.w)*phi)*dx)
            # Update residual correction
            dMr_phi.assign(Mr_phi - Mr_phi_cur)
            dMr_u.assign(Mr_u - Mr_u_cur)
            # Check for convergence and print out residual norm
            res_norm_old = res_norm
            res_norm = self.residual_norm(dMr_phi,dMr_u) 
            if (self.verbose > 1):
                print ' i = '+('%4d' % i) +  \
                      ' : '+('%8.4e' % res_norm) + \
                      ' [ '+('%8.4e' % (res_norm/res_norm_0))+' ' + \
                      ' rho = '+('%6.3f' % (res_norm/res_norm_old))+' ] '
            if (res_norm/res_norm_0 < self.tolerance):
                break
        if (self.verbose > 0):
            if (res_norm/res_norm_0 < self.tolerance):
                print ' Outer loop converged after '+str(i)+' iterations.'
            else:
                print ' Outer loop failed to converge after '+str(self.maxiter)+' iterations.'
        return u, phi    

    def residual_norm(self,Mr_phi,Mr_u):
        ''' Calculate outer residual norm.
    
        Calculates an approximation to norm of the full residual in the outer iteration:

        .. math::

            norm = ||\\tilde{r}_\phi||_{L_2} + ||\\tilde{r}_u||_{L_2}

        where :math:`\\tilde{r}_\phi = M_\phi^{-1} (M_\phi r_{\phi})` and 
        :math:`\\tilde{r}_u = \left(M_u^*\\right)^{-1} (M_ur_u)`. The multiplication with the
        inverse mass matrices is necessary because the outer iteration calculates the residuals
        :math:`M_{\phi} r_{\phi}` and :math:`M_ur_{u}`

        :arg Mr_phi: Residual multiplied by pressure mass matrix :math:`M_{\phi}r_{\phi}` 
        :arg Mr_u: Residual multiplied by velocity mass matrix :math:`M_ur_{\phi}` 
        '''

        # Rescale by (lumped) mass matrices
        # Calculate r_u = (M_u^{lumped})^{-1}*Mr_u
        r_u = Function(self.V_velocity)
        r_u.assign(Mr_u)
        self.lumped_mass.divide(r_u)
        # Calculate r_phi = (M_{phi})^{-1}*Mr_phi
        a_phi_mass = assemble(self.psi*self.phi*dx)
        r_phi = Function(self.V_pressure)
        solve(a_phi_mass, r_phi, Mr_phi, solver_parameters={'ksp_type':'cg'})
        return sqrt(assemble((r_phi*r_phi+dot(r_u,r_u))*dx))

    def solve_petsc(self,r_phi,r_u):
        '''Solve Helmholtz system using PETSc solver.

        Solve the mixed Helmholtz system for right hand sides :math:`r_\phi` and :math:`r_u`
        by using the PETSc solvers with suitable Schur complement preconditioner.

            :arg r_phi: right hand side for pressure equation, function in :math:`DG` space.
            :arg r_u: right hand side for velocity equation, function in :math:`H(div)` space.
        '''
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
