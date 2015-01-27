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
                 orography=False,
                 schur_diagonal_only=False,
                 ksp_monitor=None,
                 maxiter=100,
                 tolerance=1.E-6,
                 matrixfree=True):
        self._ksp_type = ksp_type
        self._logger = Logger()
        self._Wb = Wb
        self._dt = dt
        self._c = c
        self._N = N
        self._dt_half = Constant(0.5*dt)
        self._dt_half_N2 = Constant(0.5*dt*N**2)
        self._dt_half_c2 = Constant(0.5*dt*c**2)
        self._omega_N2 = Constant((0.5*dt*N)**2)
        self._maxiter = maxiter
        self._tolerance = tolerance
        self._pressure_solver = pressure_solver
        self._schur_diagonal_only = schur_diagonal_only
        self._matrixfree=matrixfree
        self._orography = orography
        self._ksp_monitor = ksp_monitor
        if (self._orography):
            # Solve UPB system
            self._W3 = W3
            self._W2 = W2
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
            op.setPythonContext(MixedOperatorUPB(self._W2,self._W3,self._Wb,
                                                 self._dt,self._c,self._N))
            op.setUp()

            self._ksp = PETSc.KSP()
            self._ksp.create()
            self._ksp.setOptionsPrefix('mixed_')
            self._ksp.setOperators(op)
            self._ksp.setTolerances(rtol=self._tolerance,max_it=self._maxiter)
            self._ksp.setType(self._ksp_type)
            self._ksp.setMonitor(self._ksp_monitor)
            self._logger.write('  Mixed KSP type = '+str(self._ksp.getType()))
            pc = self._ksp.getPC()
            pc.setType(pc.Type.PYTHON)
            pc.setPythonContext(MixedPreconditionerUPB(self._W2,self._W3,self._Wb,
                                                       self._dt,self._N,self._c,
                                                       self._pressure_solver,
                                                       self._schur_diagonal_only,
                                                       matrixfree_prec=self._matrixfree))
        else:
            # Solve UP system
            if (self._matrixfree):
                self._W2 = W2
                self._W3 = W3
                self._mixedarray = MixedArray(self._W2,self._W3)
                self._ndof = self._mixedarray.ndof
                self._x = PETSc.Vec()
                self._x.create()
                self._x.setSizes((self._ndof, None))
                self._x.setFromOptions()
                self._y = self._x.duplicate()

                op = PETSc.Mat().create()
                op.setSizes(((self._ndof, None), (self._ndof, None)))
                op.setType(op.Type.PYTHON)
                op.setPythonContext(MixedOperatorUP(self._W2,self._W3,
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
                pc.setPythonContext(MixedPreconditionerUP(self._W2,self._W3,self._Wb,
                                                          self._dt,self._N,self._c,
                                                          self._pressure_solver,
                                                          self._schur_diagonal_only))
            else:
                self._Wmixed = W2 * W3
                self._W2 = self._Wmixed.sub(0)
                self._W3 = self._Wmixed.sub(1)
            
        # Set up test- and trial function spaces
        self._u = Function(self._W2)
        self._p = Function(self._W3)
        self._b = Function(self._Wb)
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
        if (self._matrixfree):
            self._pressure_solver.add_to_xml(e,"pressure_solver")
        e.set("ksp_type",str(self._ksp_type))
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
                
        btest = TestFunction(self._Wb)

        if self._orography:
            utest = TestFunction(self._W2)
            ptest = TestFunction(self._W3)
            # Solve UPB system
            f_u = assemble(dot(utest,r_u)*self._dx)
            f_p = assemble(ptest*r_p*self._dx)
            f_b = assemble(btest*r_b*self._dx)
            # Copy data in
            with f_u.dat.vec_ro as u, \
                 f_p.dat.vec_ro as p, \
                 f_b.dat.vec_ro as b:
                self._mixedarray.combine(self._y,u,p,b)
            # PETSc ksp solve
            with self._ksp_monitor:
                self._ksp.solve(self._y,self._x)
            # Copy data out
            with self._u.dat.vec as u, \
                self._p.dat.vec as p, \
                self._b.dat.vec as b:
                self._mixedarray.split(self._x,u,p,b)
        else:
            vert_norm = VerticalNormal(self._W3.mesh())
            if (self._matrixfree):
                utest = TestFunction(self._W2)
                ptest = TestFunction(self._W3)
                # Solve UPB system
                f_u = assemble((dot(utest,r_u) \
                               + self._dt_half*dot(utest,vert_norm.zhat*r_b))*self._dx)
                f_p = assemble(ptest*r_p*self._dx)

                # Copy data in
                with f_u.dat.vec_ro as u, \
                     f_p.dat.vec_ro as p:
                    self._mixedarray.combine(self._y,u,p)
                # PETSc ksp solve
                with self._ksp_monitor:
                    self._ksp.solve(self._y,self._x)
                # Copy data out
                with self._u.dat.vec as u, \
                     self._p.dat.vec as p:
                    self._mixedarray.split(self._x,u,p)
            else:
                # Calculate RHS
                utest, ptest = TestFunctions(self._Wmixed)
                utrial, ptrial = TrialFunctions(self._Wmixed)
                bcs = [DirichletBC(self._Wmixed.sub(0), 0.0, "bottom"),
                       DirichletBC(self._Wmixed.sub(0), 0.0, "top")]
                sparams={'pc_type': 'fieldsplit',
                         'pc_fieldsplit_type': 'schur',
                         'ksp_type': 'gmres',
                         'ksp_max_it': 30,
                         'ksp_rtol':self._tolerance,
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
                         'ksp_monitor': False}
                a_up = (  ptest*ptrial \
                        + self._dt_half_c2*ptest*div(utrial) \
                        - self._dt_half*div(utest)*ptrial \
                        + (dot(utest,utrial) + self._omega_N2 \
                            * dot(utest,vert_norm.zhat) \
                            * dot(utrial,vert_norm.zhat)) \
                       )*self._dx
                vmixed = Function(self._Wmixed)
                L_up = ( dot(utest,r_u) + self._dt_half*dot(utest,vert_norm.zhat*r_b) \
                       + ptest*r_p) * self._dx
                up_problem = LinearVariationalProblem(a_up, L_up, vmixed, bcs=bcs)
                up_solver = LinearVariationalSolver(up_problem, solver_parameters=sparams)
                ksp = up_solver.snes.getKSP()
                ksp.setMonitor(self._ksp_monitor)
                with self._ksp_monitor:
                    up_solver.solve()
                self._u.assign(vmixed.sub(0))
                self._p.assign(vmixed.sub(1))

            L_b = dot(btest*vert_norm.zhat,self._u)*self._dx
            a_b = btest*TrialFunction(self._Wb)*self._dx
            b_tmp = Function(self._Wb)
            b_problem = LinearVariationalProblem(a_b,L_b, b_tmp)
            b_solver = LinearVariationalSolver(b_problem,solver_parameters={'ksp_type':'cg',
                                                                            'pc_type':'jacobi'})
            with timed_region('Mb_divide'):
                b_solver.solve()
            self._b = assemble(r_b-self._dt_half_N2*b_tmp)

        return self._u, self._p, self._b

