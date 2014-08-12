from firedrake import *
import sys, petsc4py
import numpy as np
from pressuresolver.mpi_utils import Logger
import xml.etree.cElementTree as ET

petsc4py.init(sys.argv)

from petsc4py import PETSc

'''Solve Helmholtz system in mixed formulation.

This module contains the :class:`.Solver` for solving a Helmholtz system 
using finite elements.
'''

class PETScSolver(object):
    '''Solver for the Helmholtz system

        .. math::
  
            \phi + \omega (\\nabla\cdot\phi^*\\vec{u}) = r_\phi

            -\\vec{u} - \omega \\nabla{\phi} = \\vec{r}_u

        in the mixed finite element formulation.

        :arg V_pressure: Function space for pressure field :math:`\phi`
        :arg V_velocity: Function space for velocity field :math:`\\vec{u}`
        :arg pressure_solver: Solver for Schur complement pressure system,
            e.g. :class:`.LoopSolver` and :class:`.CGSolver`.
        :arg velocity_mass_matrix: Explicitly specify mass matrix to be used 
            in the RHS construction and backsubstitution in the Schur
            complement preconditioner. If none is set, use the lumped mass
            from the pressuresolver object.
        :arg schur_diagonal_only: Only use the diagonal part in the 
            Schur complement preconditioner, see :class:`MixedPreconditioner`.
        :arg ksp_monitor: KSP monitor instance, see e.g. :class:`KSP_Monitor`
        :arg omega: Positive real number
        :arg maxiter: Maximal number of iterations for outer iteration
        :arg tolerance: Tolerance for outer iteration
    '''
    def __init__(self,V_pressure,V_velocity,pressure_solver,omega,
                 velocity_mass_matrix=None,
                 schur_diagonal_only=False,
                 ksp_monitor=None,
                 maxiter=100,
                 tolerance=1.E-6):
        self.logger = Logger()
        self.omega = omega
        self.maxiter = maxiter
        self.tolerance = tolerance
        self.V_pressure = V_pressure
        self.V_velocity = V_velocity
        self.pressure_solver = pressure_solver        
        self.schur_diagonal_only = schur_diagonal_only
        self.velocity_mass_matrix = velocity_mass_matrix

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
        self.ksp_monitor = ksp_monitor
        self.ksp.setMonitor(self.ksp_monitor)
        self.logger.write('  Mixed KSP type = '+str(self.ksp.getType()))
        pc = self.ksp.getPC()
        pc.setType(pc.Type.PYTHON)
        pc.setPythonContext(MixedPreconditioner(self.pressure_solver,
                                                self.V_pressure,
                                                self.V_velocity,
                                                self.velocity_mass_matrix,
                                                self.schur_diagonal_only))

        # Set up test- and trial function spaces
        self.v = Function(self.V_velocity)
        self.phi = Function(self.V_pressure)
        self.w = TestFunction(self.V_velocity)
        self.psi = TestFunction(self.V_pressure)

    def add_to_xml(self,parent,function):
        '''Add to existing xml tree.

        :arg parent: Parent node to be added to
        :arg function: Function of object
        '''
        e = ET.SubElement(parent,function)
        e.set("type",type(self).__name__)
        v_str = self.V_velocity.ufl_element()._short_name
        v_str += str(self.V_velocity.ufl_element().degree())
        e.set("velocity_space",v_str)
        v_str = self.V_pressure.ufl_element()._short_name
        v_str += str(self.V_pressure.ufl_element().degree())
        e.set("pressure_space",v_str)
        self.pressure_solver.add_to_xml(e,"pressure_solver")
        e.set("ksp_type",str(self.ksp.getType()))
        e.set("omega",('%e' % self.omega))
        e.set("maxiter",str(self.maxiter))
        e.set("tolerance",str(self.tolerance))
        e.set("schur_diagonal_only",str(self.schur_diagonal_only))
        self.velocity_mass_matrix.add_to_xml(e,"velocity_mass_matrix")
        

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
        with self.ksp_monitor:
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
                                 'fieldsplit_RT0_ksp_type':'cg'})
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
        assemble((self.omega*div(self.w)*phi - inner(self.w,u))*dx,
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
           \omega B^T & -M_u^* 
        \\end{pmatrix}^{-1}
        = 
        \\begin{pmatrix}
           1 & 0 \\\\
           (M_u^*)^{-1}\omega B^T & 1
        \\end{pmatrix}
        \\begin{pmatrix}
           H^{-1} & 0 \\\\
           0 & -(M_u^*)^{-1}
        \\end{pmatrix}
        \\begin{pmatrix}
            1 & \omega B (M_u^*)^{-1} \\\\
            0 &  1
        \\end{pmatrix}

    where :math:`H = M_{\phi} + \omega^2 B (M_u^*)^{-1} B^T` is the
    Helmholtz operator in pressure space. Depending on the value of the
    parameter diagonal_only, only the central, block-diagonal matrix is used
    and the back/forward substitution with the upper/lower triagular matrix
    is ignored.
    
    :arg pressure_solver: Solver in pressure space
    :arg V_pressure: Function space for pressure
    :arg V_velocity: Function space for velocity
    :arg velocity_mass_matrix: Explicitly specify velocity mass matrix.
        If not specified, use the one from the Helmholtz operator.
    :arg diagonal_only: Only use diagonal matrix, ignore forward/backward
        substitution with triagular matrices
    '''
    def __init__(self,pressure_solver,
                 V_pressure,
                 V_velocity,
                 velocity_mass_matrix=None,
                 diagonal_only=False):
        self.pressure_solver = pressure_solver
        if (velocity_mass_matrix==None):
            self.velocity_mass_matrix \
                = self.pressure_solver.operator.velocity_mass_matrix
        else:
            self.velocity_mass_matrix \
                = velocity_mass_matrix
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
        self.diagonal_only = diagonal_only
        
    def solve(self,R_phi,R_u,phi,u):
        '''Schur complement proconditioner for mixed system.

        Use the Schur complement in pressure space to precondition
        the mixed system. More specifically, calculate:

        .. math::

            F = R_{\phi} + \omega B (M_u^*)^{-1} R_{u}

            \phi = H^{-1} F

            u = (M_u^*)^{-1} (-R_u + \omega B^T \phi)

        If the diagonal_only parameter has been set, calculate instead:

        .. math::

            \phi = H^{-1} R_{\phi}

            u = -(M_u^*)^{-1} R_u

        :arg R_phi: = :math:`R_{\phi}` RHS in pressure space
        :arg R_u: = :math:`R_u` RHS in velocity space
        :arg phi: = :math:`\phi` Resulting pressure correction
        :arg u: Resulting velocity correction
        '''
        # Construct RHS for linear (pressure) solve
        if (self.diagonal_only):
            # Solve for pressure correction
            phi.assign(0.0)
            self.pressure_solver.solve(R_phi,phi)
            # Solve for velocity correction
            # u = -(M_u^{lumped})^{-1}*R_u
            u.assign(-R_u)
            self.velocity_mass_matrix.divide(u)
        else:
            self.dMinvMr_u.assign(R_u)
            self.velocity_mass_matrix.divide(self.dMinvMr_u)
            self.F_pressure.assign(R_phi + \
                                   self.omega * \
                                     assemble(self.psi*div(self.dMinvMr_u)*dx))
            # Solve for pressure correction
            phi.assign(0.0)
            self.pressure_solver.solve(self.F_pressure,phi)
            # Calculate for corresponding velocity
            # u = (M_u^{lumped})^{-1}*(-R_u + omega*grad(phi))
            grad_dphi = assemble(div(self.w)*phi*dx)
            u.assign(-R_u + self.omega * grad_dphi)
            self.velocity_mass_matrix.divide(u)

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
        
