from firedrake import *
import os, sys, petsc4py
import numpy as np
from mixedarray import *
from mixedoperators import *
from mixedpreconditioners import *
from auxilliary.logger import *
from pressuresolver.vertical_normal import *
from pressuresolver.mu_tilde import *
import xml.etree.cElementTree as ET

petsc4py.init(sys.argv)

from petsc4py import PETSc

'''Solve Linear gravity wave system in mixed formulation.

This module contains the :class:`.PETScSolver` for solving the linear gravity wave system

.. math::

    \\vec{u}  - \Delta t/2 grad p - \Delta t/2\hat{z} b = \\vec{r}_u

    \\Delta t/2 c^2 div\\vec{u} + p = r_p

    \\Delta t/2 N^2 \zhat\dot\\vec{u} + b = r_b

using mixed finite elements.
'''

class Solver(object):
    '''Iterative solver for the mixed formulation of the linear gravity wave system.

        This class uses the iterative PETSc solvers to solve the linear
        equation encountered for the gravity wave system defined as

        .. math::

            \\vec{u}  - \\frac{\Delta t}{2} grad p 
                      - \\frac{\Delta t}{2}\hat{\\vec{z}} b = \\vec{r}_u

            \\frac{\Delta t}{2}c^2 div\\vec{u} + p = r_p

            \\frac{\Delta t}{2} N^2 \hat{\\vec{z}}\cdot\\vec{u} + b = r_b

        The mixed finite element formulation results in
    
        .. math::

            M_u \\vec{U} - \\frac{\Delta t}{2} D^T \\vec{P} 
                         - \\frac{\Delta t}{2} Q \\vec{B} = \\vec{R}_u

            \\frac{\Delta t}{2} c^2 D \\vec{U} + M_p \\vec{P} = \\vec{R}_p

            \\frac{\Delta t}{2} N^2 Q^T \\vec{U} + M_b\\vec{B} = \\vec{R}_b

        where :math:`\\vec{U}`, :math:`\\vec{P}` and :math:`\\vec{B}` are the
        dof-vectors for velocity, pressure and buoyancy.

        :arg W2: HDiv Function space for velocity field :math:`\\vec{u}`
        :arg W3: L2 Function space for pressure field :math:`p`
        :arg Wb: Function space for buoyancy field :math:`b`
        :arg ksp_type: String describing the PETSc KSP solver (e.g. ``gmres``)
        :arg pressure_solver: Solver for Schur complement pressure system.
            This is an instance of :class:`.IterativeSolver`
        :arg dt: Positive real number, time step size :math:`\Delta t`
        :arg c: Positive real number, speed of sound waves
        :arg N: Positive real number, buoyancy frequency
        :arg schur_diagonal_only: Only use the diagonal part in the 
            Schur complement preconditioner, see :class:`MixedPreconditioner`.
        :arg ksp_monitor: KSP monitor instance, see e.g. :class:`KSPMonitor`
        :arg maxiter: Maximal number of iterations for outer iteration
        :arg tolerance: Tolerance for outer iteration
    '''
    def __init__(self,
                 W2,W3,Wb,
                 pressure_solver,
                 dt,c,N,
                 ksp_type='gmres',
                 schur_diagonal_only=False,
                 ksp_monitor=None,
                 maxiter=100,
                 tolerance=1.E-6,
                 matrixfree_prec=True):
        self._ksp_type = ksp_type
        self._logger = Logger()
        self._W3 = W3
        self._W2 = W2
        self._Wb = Wb
        self._dt = dt
        self._c = c
        self._N = N
        self._maxiter = maxiter
        self._tolerance = tolerance
        self._pressure_solver = pressure_solver
        self._schur_diagonal_only = schur_diagonal_only
        self._mixedarray = MixedArray(self._W2,self._W3,self._Wb)
        self._ndof = self._mixedarray.ndof
        self._x = PETSc.Vec()
        self._x.create()
        self._x.setSizes((self._ndof, None))
        self._x.setFromOptions()
        self._y = self._x.duplicate()

        op = PETSc.Mat().create()
        op.setSizes(((self._ndof, None), (self._ndof, None)))
        op.setType(op.Type.PYTHON)
        op.setPythonContext(MixedOperator(self._W2,self._W3,self._Wb,
                                          self._dt,self._c,self._N))
        op.setUp()

        self._ksp = PETSc.KSP()
        self._ksp.create()
        self._ksp.setOptionsPrefix('mixed_')
        self._ksp.setOperators(op)
        self._ksp.setTolerances(rtol=self._tolerance,max_it=self._maxiter)
        self._ksp.setType(self._ksp_type)
        self._ksp_monitor = ksp_monitor
        self._ksp.setMonitor(self._ksp_monitor)
        self._logger.write('  Mixed KSP type = '+str(self._ksp.getType()))
        pc = self._ksp.getPC()
        pc.setType(pc.Type.PYTHON)
        pc.setPythonContext(MixedPreconditioner(self._W2,self._W3,self._Wb,
                                                self._dt,self._N,self._c,
                                                self._pressure_solver,
                                                self._schur_diagonal_only,
                                                matrixfree_prec=matrixfree_prec))

        # Set up test- and trial function spaces
        self._u = Function(self._W2)
        self._p = Function(self._W3)
        self._b = Function(self._Wb)
        self._utest = TestFunction(self._W2)
        self._ptest = TestFunction(self._W3)
        self._btest = TestFunction(self._Wb)
        self._dx = self._W3._mesh._dx

    def add_to_xml(self,parent,function):
        '''Add to existing xml tree.

        :arg parent: Parent node to be added to
        :arg function: Function of object
        '''
        e = ET.SubElement(parent,function)
        e.set("type",type(self).__name__)
        v_str = self._W2.ufl_element().shortstr()
        e.set("velocity_space",v_str)
        v_str = self._W3.ufl_element().shortstr()
        e.set("pressure_space",v_str)
        v_str = self._Wb.ufl_element().shortstr()
        e.set("buoyancy_space",v_str)
        self._pressure_solver.add_to_xml(e,"pressure_solver")
        e.set("ksp_type",str(self._ksp.getType()))
        e.set("dt",('%e' % self._dt))
        e.set("c",('%e' % self._c))
        e.set("N",('%e' % self._N))
        e.set("maxiter",str(self._maxiter))
        e.set("tolerance",str(self._tolerance))
        e.set("schur_diagonal_only",str(self._schur_diagonal_only))
        

    def solve(self,r_u,r_p,r_b):
        '''Solve Gravity system using nested iteration and return result.

        Solve the mixed linear system for right hand sides :math:`r_u`,
        :math:`r_p` and :math:`r_b`. The full velocity mass matrix is used in the outer
        iteration and the pressure correction system is solved with the 
        specified :class:`pressure_solver` in an inner iteration.

        Return the solution :math:`u`, :math:`p`, :math:`b`.

        See `Notes in LaTeX <./GravityWaves.pdf>`_ for more details of the
        algorithm.

        :arg r_u: right hand side for velocity equation
        :arg r_p: right hand side for pressure equation
        :arg r_b: right hand side for buoyancy equation
        '''
        # Fields for solution
        self._u.assign(0.0)
        self._p.assign(0.0)
        self._b.assign(0.0)

        # Copy data in
        with r_u.dat.vec_ro as u, \
             r_p.dat.vec_ro as p, \
             r_b.dat.vec_ro as b:
            self._mixedarray.combine(self._y,u,p,b)
        # PETSc ksp solve
        with self._ksp_monitor:
            self._ksp.solve(self._y,self._x)
        # Copy data out
        with self._u.dat.vec as u, \
             self._p.dat.vec as p, \
             self._b.dat.vec as b:
            self._mixedarray.split(self._x,u,p,b)
        return self._u, self._p, self._b

class MixedPreconditioner(object):
    '''Schur complement preconditioner for the mixed gravity wave system.

        Use the following algorithm to precondition the mixed gravity wave system:

        1. Calculate

        .. math::

            M_u\\tilde{\\vec{R}}_u = M_u\\vec{R}_u+\\frac{\Delta t}{2}QM_b^{-1}(M_b\\vec{R}_b)

            M_p\\tilde{\\vec{R}}_p = M_p\\vec{R}_p
                - \\frac{\Delta t}{2}D\\tilde{M}_u^{-1}(M_u\\tilde{\\vec{R}}_u)
        
        2. Solve :math:`H\\vec{P}=(M_p\\tilde{\\vec{R}}_p)` for :math:`\\vec{P}` using
            the specified pressure solver

        3. Calculate
        
        .. math::
            \\vec{U} = \\tilde{M}_u^{-1}((M_u\\tilde{\\vec{R}}_u)
                     + \\frac{\Delta t}{2}D^T \\vec{P})

            \\vec{B} = M_b^{-1}((M_b\\vec{R}_b)-\\frac{\Delta t}{2}N^2 Q^T \\vec{U})

        Here :math:`\\tilde{M_u} = M_u + \omega_N^2 Q M_b^{-1} Q^T` is the
        modified velocity mass matrix (see :class:`.Mutilde`) and
        :math:`H = M_{p} + \omega_c^2 D (\\tilde{M}_u)^{-1} D^T` is the
        Helmholtz operator in pressure space. Depending on the value of the
        parameter diagonal_only, only the central, block-diagonal matrix is used
        and in backward/forward substitution (steps 1. and 3. above) the terms which are 
        formally of order :math:`\Delta t` are ignored.
    
        :arg W2: Hdiv function space for velocity
        :arg W3: L2 function space for velocity
        :arg Wb: Function space for buoyancy
        :arg dt: Time step size
        :arg N: Buoyancy frequency
        :arg c: Speed of sound waves
        :arg pressure_solver: Solver in pressure space
        :arg diagonal_only: Only use diagonal matrix, ignore forward/backward
            substitution with triagular matrices
        :arg use_petsc: Use PETSc to solve mixed system in :math:`(u,p)` space
    '''
    def __init__(self,
                 W2,W3,Wb,
                 dt,N,c,
                 pressure_solver,
                 diagonal_only=False,
                 matrixfree_prec=True,
                 tolerance_b=1.E-12,maxiter_b=1000,
                 tolerance_u=1.E-12,maxiter_u=1000):
        self._pressure_solver = pressure_solver
        self._W2 = W2
        self._W3 = W3
        self._Wb = Wb
        self._omega_N = 0.5*dt*N
        self._dt_half = Constant(0.5*dt)
        self._dt_half_N2 = Constant(0.5*dt*N**2)
        self._dt_half_c2 = Constant(0.5*dt*c**2)
        self._diagonal_only = diagonal_only
        self._mesh = self._W3._mesh
        self._zhat = VerticalNormal(self._mesh)
        self._dx = self._mesh._dx
        self._matrixfree_prec = matrixfree_prec
        if (not self._matrixfree_prec):
            self._Wmixed = self._W2 * self._W3
            self._mutest, self._mptest = TestFunctions(self._Wmixed)
            self._mutrial, self._mptrial = TrialFunctions(self._Wmixed)
            self._bcs = [DirichletBC(self._Wmixed.sub(0), 0.0, "bottom"),
                         DirichletBC(self._Wmixed.sub(0), 0.0, "top")]
            self._sparams={'pc_type': 'fieldsplit',
                           'pc_fieldsplit_type': 'schur',
                           'ksp_type': 'gmres',
                           'ksp_max_it': 30,
                           'pc_fieldsplit_schur_fact_type': 'FULL',
                           'pc_fieldsplit_schur_precondition': 'selfp',
                           'fieldsplit_0_ksp_type': 'preonly',
                           'fieldsplit_0_pc_type': 'bjacobi',
                           'fieldsplit_0_sub_pc_type': 'ilu',
                           'fieldsplit_1_ksp_type': 'preonly',
                           'fieldsplit_1_pc_type': 'gamg',
                           'fieldsplit_1_mg_levels_ksp_type': 'chebyshev',
                           'fieldsplit_1_mg_levels_ksp_chebyshev_estimate_eigenvalues': True,
                           'fieldsplit_1_mg_levels_ksp_chebyshev_estimate_eigenvalues_random': True,
                           'fieldsplit_1_mg_levels_ksp_max_it': 1,
                           'fieldsplit_1_mg_levels_pc_type': 'bjacobi',
                           'fieldsplit_1_mg_levels_sub_pc_type': 'ilu',
                           'ksp_monitor': True}

            self._a = (  self._mptest*self._mptrial \
                       + self._dt_half_c2*self._mptest*div(self._mutrial) \
                       - self._dt_half*div(self._mutest)*self._mptrial \
                       + (dot(self._mutest,self._mutrial) + self._omega_N**2 \
                            * dot(self._mutest,self._zhat.zhat) \
                            * dot(self._mutrial,self._zhat.zhat)) \
                      )*self._dx
            self._vmixed = Function(self._Wmixed)
        else:
            self._bcs = [DirichletBC(self._W2, 0.0, "bottom"),
                         DirichletBC(self._W2, 0.0, "top")]
        self._utest = TestFunction(self._W2)
        self._ptest = TestFunction(self._W3)
        self._btest = TestFunction(self._Wb)
        # Buoyancy mass matrix
        self._Mb = assemble(self._btest*TrialFunction(self._Wb)*self._dx)
        self._solver_param_b = {'ksp_type':'cg',
                                'ksp_rtol':tolerance_b,
                                'ksp_max_it':maxiter_b,
                                'ksp_monitor':False,
                                'pc_type':'jacobi'}
        self._mutilde = Mutilde(self._W2,self._Wb,self._omega_N,
                                tolerance_b=tolerance_b,maxiter_b=maxiter_b,
                                tolerance_u=tolerance_u,maxiter_u=maxiter_u)
        # Temporary functions
        self._rtilde_u = Function(self._W2)
        self._rtilde_p = Function(self._W3)
        self._u = Function(self._W2)
        self._p = Function(self._W3)
        self._b = Function(self._Wb)
        self._tmp_u = Function(self._W2)
        self._tmp_b = Function(self._Wb)
        self._Pu = Function(self._W2)
        self._Pp = Function(self._W3)
        self._Pb = Function(self._Wb)
        self._mixedarray = MixedArray(self._W2,self._W3,self._Wb)
        
    def solve(self,r_u,r_p,r_b,u,p,b):
        '''Preconditioner solve.

        Given the RHS r_u, r_p and r_b, calculate the fields 
        u,p and b.

        :arg r_u: RHS in velocity space
        :arg r_p: RHS in pressure space
        :arg r_b: RHS in buoyancy space
        :arg u: Solution for velocity
        :arg p: Solution for pressure
        :arg b: Solution for buoyancy
        '''
       
        if (self._diagonal_only):
            assert (self._matrixfree_prec)
            # Pressure solve
            p.assign(0.0)
            self._pressure_solver.solve(r_p,p)
            # Velocity solve
            self._mutilde.divide(r_u,u)
            # Buoyancy solve
            solve(self._Mb,b,r_b,solver_parameters=self._solver_param_b)
        else:
            # Modified RHS for velocity 
            solve(self._Mb,self._tmp_b,r_b,solver_parameters=self._solver_param_b)
            assemble(self._dt_half * dot(self._utest,self._zhat.zhat) \
                                   * self._tmp_b * self._dx,
                     tensor=self._rtilde_u)
            self._rtilde_u += r_u
            if (self._matrixfree_prec):
                # Modified RHS for pressure
                self._mutilde.divide(self._rtilde_u,self._tmp_u)
                assemble(- self._dt_half_c2 * self._ptest * div(self._tmp_u) * self._dx,
                         tensor=self._rtilde_p)
                self._rtilde_p += r_p
                # Pressure solve
                p.assign(0.0)
                self._pressure_solver.solve(self._rtilde_p,p)
                # Backsubstitution for velocity 
                assemble(self._dt_half * div(self._utest) * p*self._dx,
                         tensor=self._tmp_u)
                self._tmp_u += self._rtilde_u
                self._mutilde.divide(self._tmp_u,u)
            else:
                m_p = assemble(TestFunction(self._W3)*TrialFunction(self._W3)*self._dx)
                m_u = assemble(dot(TestFunction(self._W2),TrialFunction(self._W2))*self._dx)
                r_u.assign(self._rtilde_u)
                solve(m_p, self._rtilde_p, r_p)
                solve(m_u, self._rtilde_u, r_u)
                L = (  self._mptest*self._rtilde_p \
                     + dot(self._mutest,self._rtilde_u))*self._dx
                solve(self._a == L,self._vmixed,
                      solver_parameters=self._sparams,
                      bcs=self._bcs)
                u.assign(self._vmixed.sub(0))
                p.assign(self._vmixed.sub(1))
            # Backsubstitution for buoyancy
            assemble(- self._dt_half_N2 * self._btest*dot(self._zhat.zhat,u)*self._dx,
                     tensor=self._tmp_b)
            self._tmp_b += r_b
            solve(self._Mb,b,self._tmp_b,solver_parameters=self._solver_param_b)

    def apply(self,pc,x,y):
        '''PETSc interface for preconditioner solve.

        PETSc interface wrapper for the :func:`solve` method.

        :arg x: PETSc vector representing the mixed right hand side
        :arg y: PETSc vector representing the mixed solution vector
        '''
        
        with self._u.dat.vec as u, \
             self._p.dat.vec as p, \
             self._b.dat.vec as b:
            self._mixedarray.split(x,u,p,b)
        self.solve(self._u,self._p,self._b,
                   self._Pu,self._Pp,self._Pb)
        with self._Pu.dat.vec_ro as u, \
             self._Pp.dat.vec_ro as p, \
             self._Pb.dat.vec_ro as b:
            self._mixedarray.combine(y,u,p,b)
        
