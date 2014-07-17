from firedrake import *
'''Solve Helmholtz system in mixed formulation.

This module contains the :class:`.Solver` for solving a Helmholtz system 
using finite elements.
'''

class Solver:
    '''Solver for the Helmholtz system

        .. math::
  
            \phi + \omega (\\nabla\cdot\phi^*\\vec{u}) = r_\phi

            \\vec{u} + \omega \\nabla{\phi} = \\vec{r}_u

        in the mixed finite element formulation.

        :arg V_pressure: Function space for pressure field :math:`\phi`
        :arg V_velocity: Function space for velocity field :math:`\\vec{u}`
        :arg pressure_solver: Solver for Schur complement pressure system,
            e.g. :class:`.LoopSolver` and :class:`.CGSolver`.
        :arg omega: Positive real number
        :arg maxiter: Maximal number of iterations for outer iteration
        :arg tolerance: Tolerance for outer iteration
        :arg verbose: Verbosity level (0=no output, 1=minimal output,
            2=show convergence rates)
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

        Solve the mixed Helmholtz system for right hand sides :math:`r_\phi`
        and :math:`r_u`. The full velocity mass matrix is used in the outer
        iteration and the pressure correction system is solved with the 
        specified :class:`pressure_solver` in an inner iteration, i.e. a
        preconditioned Richardson iteration for the mixed system is used, and
        the velocity mass matrix is replaced by a lumped version in the
        preconditioner.

        See `Notes in LaTeX <./FEMmultigrid.pdf>`_ for more details of the
        algorithm.

        :arg r_phi: right hand side for pressure equation, function in 
            :math:`DG` space.
        :arg r_u: right hand side for velocity equation, function in 
            :math:`H(div)` space.
        '''
        if (self.verbose > 0):
            print ' === Helmholtz solver ==='
        # Fields for solution
        u = Function(self.V_velocity)
        du = Function(self.V_velocity)
        phi = Function(self.V_pressure)
        self.F_pressure = Function(self.V_pressure)
        self.dMinvMr_u = Function(self.V_velocity)
        u.assign(0.0)
        phi.assign(0.0)
        # Pressure correction
        d_phi = Function(self.V_pressure)
        # Residual correction
        Mr_phi = assemble(self.psi*r_phi*dx)
        Mr_u = assemble(dot(self.w,r_u)*dx)
        dMr_phi = Function(self.V_pressure)
        dMr_u = Function(self.V_velocity)
        dMr_phi.assign(Mr_phi)
        dMr_u.assign(Mr_u)
        # Calculate initial residual
        res_norm = self.residual_norm(dMr_phi,dMr_u) 
        res_norm_0 = res_norm 
        DFile = File('output/Fpressure.pvd')
        DFile << dMr_phi
        if (self.verbose > 0):
            print ' initial outer residual : '+('%e' % res_norm_0)
        for i in range(1,self.maxiter+1):
            # Calculate correction to pressure and velocity by
            # calling Schur-complement preconditioner
            self.schur_complement_preconditioner(dMr_phi,dMr_u,d_phi,du)
            # Add correction phi -> phi + d_phi, u -> u + du
            phi.assign(phi + d_phi)
            u.assign(u + du)
            # Calculate current residual
            Mr_phi_cur = assemble((self.psi*phi \
                                 + self.omega*self.psi*div(u))*dx)
            Mr_u_cur = assemble(( inner(self.w,u) \
                                 - self.omega*div(self.w)*phi)*dx)
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

    def schur_complement_preconditioner(self,R_phi,R_u,phi,u):
        '''Schur complement proconditioner for mixed system.

        Use the Schur complement in pressure space to precondition
        the mixed system. More specifically, calculate:

        .. math::

            F = R_{\phi} - \omega B (M_u^*)^{-1} R_{u}

            \phi = A^{-1}

            u = (M_u^*)^{-1} ( R_u + \omega B^T \phi)

        This is equivalent to using the following Schur complement
        decomposition of the mixed system with lumped velocity mass matrix

        .. math::

            \\begin{pmatrix}
                M_\phi & \omega B \\\\
                -\omega B^T & M_u^* 
            \\end{pmatrix}^{-1}
            = 
            \\begin{pmatrix}
                1 & 0 \\\\
                (M_u^*)^{-1}\omega B^T & 1
            \\end{pmatrix}
            \\begin{pmatrix}
                H^{-1} & 0 \\\\
                0 & (M_u^*)^{-1}
            \\end{pmatrix}
            \\begin{pmatrix}
                1 & -\omega B (M_u^*)^{-1} \\\\
                0 &  1
            \\end{pmatrix}

        where :math:`H = M_{\phi} + \omega^2 B (M_u^*)^{-1} B^T` is the
        Helmholtz operator in pressure space.
       
        :arg R_phi: = :math:`R_{\phi}` RHS in pressure space
        :arg R_u: = :math:`R_u` RHS in velocity space
        :arg phi: = :math:`\phi` Resulting pressure correction
        :arg u: Resulting velocity correction
        '''
        # Construct RHS for linear (pressure) solve
        self.dMinvMr_u.assign(R_u)
        self.lumped_mass.divide(self.dMinvMr_u)
        self.F_pressure.assign(R_phi - \
                               self.omega*assemble(self.psi*div(self.dMinvMr_u)*dx))
        # Solve for pressure correction
        phi.assign(0.0)
        self.pressure_solver.solve(self.F_pressure,phi)
        # Calculate for corresponding velocity
        # u = (M_u^{lumped})^{-1}*(R_u + omega*grad(phi))
        grad_dphi = assemble(div(self.w)*phi*dx)
        u.assign(R_u + self.omega * grad_dphi)
        self.lumped_mass.divide(u)


    def residual_norm(self,Mr_phi,Mr_u):
        ''' Calculate outer residual norm.
    
        Calculates the norm of the full residual in the outer iteration:

        .. math::

            norm = ||\\tilde{r}_\phi||_{L_2} + ||\\tilde{r}_u||_{L_2}

        where :math:`\\tilde{r}_\phi = M_\phi^{-1} (M_\phi r_{\phi})` and 
        :math:`\\tilde{r}_u = \left(M_u\\right)^{-1} (M_ur_u)`. The
        multiplication with the inverse mass matrices is necessary because the
        outer iteration calculates the residuals
        :math:`M_{\phi} r_{\phi}` and :math:`M_ur_{u}`

        :arg Mr_phi: Residual multiplied by pressure mass matrix
            :math:`M_{\phi}r_{\phi}` 
        :arg Mr_u: Residual multiplied by velocity mass matrix
            :math:`M_ur_{\phi}` 
        '''

        # Rescale by mass matrices

        # (1) Calculate r_u = (M_u^{-1})*Mr_u
        a_u_mass = assemble(dot(self.w,self.u)*dx)
        r_u = Function(self.V_velocity)
        solve(a_u_mass, r_u, Mr_u, solver_parameters={'ksp_type':'cg'})

        # (2) Calculate r_phi = (M_{phi})^{-1}*Mr_phi
        a_phi_mass = assemble(self.psi*self.phi*dx)
        r_phi = Function(self.V_pressure)
        solve(a_phi_mass, r_phi, Mr_phi, solver_parameters={'ksp_type':'cg'})
        return sqrt(assemble((r_phi*r_phi+dot(r_u,r_u))*dx))

    def solve_petsc(self,r_phi,r_u):
        '''Solve Helmholtz system using PETSc solver.

        Solve the mixed Helmholtz system for right hand sides :math:`r_\phi`
        and :math:`r_u` by using the PETSc solvers with suitable Schur
        complement preconditioner.

        :arg r_phi: right hand side for pressure equation, function in
            :math:`DG` space.
        :arg r_u: right hand side for velocity equation, function in
            :math:`H(div)` space.
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
