from firedrake import *
import sys, petsc4py
import numpy as np

petsc4py.init(sys.argv)

from petsc4py import PETSc

'''Solve Helmholtz system in mixed formulation.

This module contains the :class:`.Solver` for solving a Helmholtz system 
using finite elements.
'''

class PETScSolver:
    '''Solver for the Helmholtz system

        .. math::
  
            \phi + \omega (\\nabla\cdot\phi^*\\vec{u}) = r_\phi

            \\vec{u} + \omega \\nabla{\phi} = \\vec{r}_u

        in the mixed finite element formulation.

        :arg V_pressure: Function space for pressure field :math:`\phi`
        :arg V_velocity: Function space for velocity field :math:`\\vec{u}`
        :arg pressure_solver: Solver for Schur complement pressure system,
            e.g. :class:`.LoopSolver` and :class:`.CGSolver`.
        :arg lumped_mass: Explicitly specify lumped mass for preconsitioner. If none is set, use the lumped mass from the pressuresolver object.
        :arg omega: Positive real number
        :arg maxiter: Maximal number of iterations for outer iteration
        :arg tolerance: Tolerance for outer iteration
    '''
    def __init__(self,V_pressure,V_velocity,pressure_solver,omega,
                 lumped_mass=None,
                 maxiter=100,
                 tolerance=1.E-6):
        self.omega = omega
        self.maxiter = maxiter
        self.tolerance = tolerance
        self.V_pressure = V_pressure
        self.V_velocity = V_velocity

        self.ndof_phi = self.V_pressure.dof_dset.size
        self.ndof_u = self.V_velocity.dof_dset.size
        self.ndof = self.ndof_phi+self.ndof_u 
        self.u = PETSc.Vec()
        self.u.create()
        self.u.setSizes((self.ndof, None))
        self.u.setFromOptions()
        self.rhs = self.u.duplicate()

        op = PETSc.Mat().create()
        op.setSizes(((self.ndof, None), (self.ndof, None)))
        op.setType(op.Type.PYTHON)
        op.setPythonContext(MixedOperator(self.V_pressure,
                                          self.V_velocity,
                                          self.omega))
        op.setUp()

        self.ksp = PETSc.KSP()
        self.ksp.create()
        self.ksp.setOptionsPrefix('mixed_')
        self.ksp.setOperators(op)
        self.ksp.setTolerances(rtol=self.tolerance,max_it=self.maxiter)
        self.ksp.setFromOptions()
        pc = self.ksp.getPC()
        pc.setType(pc.Type.PYTHON)
        pc.setPythonContext(MixedPreconditioner(pressure_solver,
                                                self.V_pressure,
                                                self.V_velocity,
                                                lumped_mass))

        # Set up test- and trial function spaces
        self.v = Function(self.V_velocity)
        self.phi = Function(self.V_pressure)
        self.w = TestFunction(self.V_velocity)
        self.psi = TestFunction(self.V_pressure)

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
        # Fields for solution
        self.phi.assign(0.0)
        self.v.assign(0.0)

        # RHS
        Mr_phi = assemble(self.psi*r_phi*dx)
        Mr_u = assemble(dot(self.w,r_u)*dx)

        with Mr_phi.dat.vec_ro as v:
            self.rhs.array[:self.ndof_phi] = v.array[:]
        with Mr_u.dat.vec_ro as v:
            self.rhs.array[self.ndof_phi:] = v.array[:]
        self.ksp.solve(self.rhs,self.u)
        with self.phi.dat.vec as v:
            v.array[:] = self.u.array[:self.ndof_phi] 
        with self.v.dat.vec as v:
            v.array[:] = self.u.array[self.ndof_phi:]

        return self.phi, self.v

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

class MixedOperator(object):
    '''Matrix free operator for mixed Helmholtz system

    :arg V_pressure: Function space for pressure
    :arg V_velocity: Function space for velocity
    '''
    def __init__(self,V_pressure,V_velocity,omega):
        self.V_pressure = V_pressure
        self.V_velocity = V_velocity
        self.omega = omega
        self.psi = TestFunction(self.V_pressure)
        self.w = TestFunction(self.V_velocity)
        self.ndof_phi = self.V_pressure.dof_dset.size
        self.phi_tmp = Function(self.V_pressure)
        self.u_tmp = Function(self.V_velocity)
        self.Mr_phi_tmp = Function(self.V_pressure)
        self.Mr_u_tmp = Function(self.V_velocity)

    def apply(self,phi,u,Mr_phi,Mr_u):
        assemble((self.psi*phi + self.omega*self.psi*div(u))*dx,
                 tensor=Mr_phi)
        assemble((inner(self.w,u) - self.omega*div(self.w)*phi)*dx,
                  tensor=Mr_u)

    def mult(self,mat,x,y):
        '''PETSc interface for operator application

        PETSc interface wrapper for the :func:`apply` method.
        :arg x: PETSc vector representing the field to be multiplied.
        :arg y: PETSc vector representing the result.
        '''
        with self.phi_tmp.dat.vec as v:
            v.array[:] = x.array[:self.ndof_phi]
        with self.u_tmp.dat.vec as v:
            v.array[:] = x.array[self.ndof_phi:]
        self.apply(self.phi_tmp,self.u_tmp,
                   self.Mr_phi_tmp,self.Mr_u_tmp)
        with self.Mr_phi_tmp.dat.vec_ro as v:
            y.array[:self.ndof_phi] = v.array[:]
        with self.Mr_u_tmp.dat.vec_ro as v:
            y.array[self.ndof_phi:] = v.array[:]

class MixedPreconditioner(object):
    '''Schur complement preconditioner for the mixed Helmholtz system

    Use the following Schur complement decomposition of the mixed system with
    lumped velocity mass matrix

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
    
    :arg pressure_solver: Solver in pressure space
    :arg V_pressure: Function space for pressure
    :arg V_velocity: Function space for velocity
    :arg lumped_mass: Explicitly specify lumped mass matrix. If not specified, use the one from the Helmholtz operator.
    '''
    def __init__(self,pressure_solver,
                 V_pressure,
                 V_velocity,
                 lumped_mass=None):
        self.pressure_solver = pressure_solver
        if (lumped_mass==None):
            self.lumped_mass = self.pressure_solver.operator.lumped_mass
        else:
            self.lumped_mass = lumped_mass
        self.V_pressure = V_pressure
        self.V_velocity = V_velocity
        self.F_pressure = Function(self.V_pressure)
        self.dMinvMr_u = Function(self.V_velocity)
        self.psi = TestFunction(self.V_pressure)
        self.w = TestFunction(self.V_velocity)
        self.omega = self.pressure_solver.operator.omega
        self.phi_tmp = Function(self.V_pressure)
        self.u_tmp = Function(self.V_velocity)
        self.P_phi_tmp = Function(self.V_pressure)
        self.P_u_tmp = Function(self.V_velocity)
        self.ndof_phi = self.V_pressure.dof_dset.size
        
    def solve(self,R_phi,R_u,phi,u):
        '''Schur complement proconditioner for mixed system.

        Use the Schur complement in pressure space to precondition
        the mixed system. More specifically, calculate:

        .. math::

            F = R_{\phi} - \omega B (M_u^*)^{-1} R_{u}

            \phi = A^{-1}

            u = (M_u^*)^{-1} ( R_u + \omega B^T \phi)
 
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

    def apply(self,pc,x,y):
        '''PETSc interface for preconditioner solve.

        PETSc interface wrapper for the :func:`solve` method.

        :arg x: PETSc vector representing the right hand side in pressure
            space
        :arg y: PETSc vector representing the solution pressure space.
        '''
        with self.phi_tmp.dat.vec as v:
            v.array[:] = x.array[:self.ndof_phi]
        with self.u_tmp.dat.vec as v:
            v.array[:] = x.array[self.ndof_phi:]
        self.solve(self.phi_tmp,self.u_tmp,
                   self.P_phi_tmp,self.P_u_tmp)
        with self.P_phi_tmp.dat.vec_ro as v:
            y.array[:self.ndof_phi] = v.array[:]
        with self.P_u_tmp.dat.vec_ro as v:
            y.array[self.ndof_phi:] = v.array[:]
        
