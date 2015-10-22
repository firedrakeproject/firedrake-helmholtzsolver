from firedrake import *
import os, sys, petsc4py
import numpy as np
from mixedarray import *
from mixedoperators import *
from mixedpreconditioners import *
from auxilliary.logger import *
from pressuresolver.vertical_normal import *
import xml.etree.cElementTree as ET

petsc4py.init(sys.argv)

from petsc4py import PETSc

'''Solve Linear gravity wave system in mixed formulation.

    This module contains different classes for solving the linear gravity wave system

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
'''

class IterativeSolver(object):
    def __init__(self,
                 Wb,
                 mixed_array,
                 mixed_operator,
                 mixed_preconditioner,
                 ksp_type,
                 schur_diagonal_only,
                 ksp_monitor,
                 maxiter,
                 tolerance,
                 pressure_solver):
        self._ksp_type = ksp_type
        self._logger = Logger()
        self._Wb = Wb
        self._dt_half = mixed_operator._dt_half
        self._dt_half_N2 = mixed_operator._dt_half_N2
        self._dt_half_c2 = mixed_operator._dt_half_c2
        self._omega_N2 = mixed_operator._omega_N2
        self._maxiter = maxiter
        self._tolerance = tolerance
        self._pressure_solver = pressure_solver
        self._mixed_operator = mixed_operator
        self._schur_diagonal_only = schur_diagonal_only
        self._ksp_monitor = ksp_monitor
        self._W2 = mixed_operator._W2
        self._W3 = mixed_operator._W3
        self._mixedarray = mixed_array
        self._ndof = self._mixedarray.ndof
        self._x = PETSc.Vec()
        self._x.create()
        self._x.setSizes((self._ndof, None))
        self._x.setFromOptions()
        self._y = self._x.duplicate()

        op = PETSc.Mat().create()
        op.setSizes(((self._ndof, None), (self._ndof, None)))
        op.setType(op.Type.PYTHON)
        op.setPythonContext(mixed_operator)
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
        pc.setPythonContext(mixed_preconditioner)
            
        # Set up test- and trial function spaces
        self._u = Function(self._W2)
        self._p = Function(self._W3)
        self._b = Function(self._Wb)
        self._dx = self._W3._mesh._dx

class MatrixFreeSolver(IterativeSolver):
    '''Matrix-free solver for the gravity wave system without orography

        :arg mixed_operator: Mixed operator (:class:`.Mutilde`
        :arg mutilde: Modfied velocity mass matrix (:class:`.Mutilde`)
        :arg Wb: Function space for buoyancy field :math:`b`
        :arg ksp_type: String describing the PETSc KSP solver (e.g. ``gmres``)
        :arg pressure_solver: Solver for Schur complement pressure system.
            This is an instance of :class:`.IterativeSolver`
        :arg schur_diagonal_only: Only use the diagonal part in the 
            Schur complement preconditioner, see :class:`MixedPreconditioner`.
        :arg ksp_monitor: KSP monitor instance, see e.g. :class:`KSPMonitor`
        :arg maxiter: Maximal number of iterations for outer iteration
        :arg tolerance: Tolerance for outer iteration
    '''
    def __init__(self,
                 Wb,
                 mixed_operator,
                 mutilde,
                 ksp_type='gmres',
                 schur_diagonal_only=False,
                 ksp_monitor=None,
                 maxiter=100,
                 tolerance=1.E-6,
                 pressure_solver=None):
        mixed_array = MixedArray(mixed_operator._W2,mixed_operator._W3)
        mixed_preconditioner = MixedPreconditioner(mixed_operator,
                                                   mutilde,
                                                   Wb,
                                                   pressure_solver,
                                                   schur_diagonal_only)
        super(MatrixFreeSolver,self).__init__(Wb,
                                              mixed_array,
                                              mixed_operator,
                                              mixed_preconditioner,
                                              ksp_type,
                                              schur_diagonal_only,
                                              ksp_monitor,
                                              maxiter,
                                              tolerance,
                                              pressure_solver)
        

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
        e.set("ksp_type",str(self._ksp_type))
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

        vert_norm = VerticalNormal(self._W3.mesh())
        utest = TestFunction(self._W2)
        ptest = TestFunction(self._W3)
        # Solve UP system
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
        with timed_region('matrixfree buoyancy solve'):
            L_b = dot(btest*vert_norm.zhat,self._u)*self._dx
            a_b = btest*TrialFunction(self._Wb)*self._dx
            b_tmp = Function(self._Wb)
            b_problem = LinearVariationalProblem(a_b,L_b, b_tmp)
            b_solver = LinearVariationalSolver(b_problem,solver_parameters={'ksp_type':'cg',
                                                                        'pc_type':'jacobi'})
            b_solver.solve()
        self._b = assemble(r_b-self._dt_half_N2*b_tmp)
        return self._u, self._p, self._b

class MatrixFreeSolverOrography(IterativeSolver):
    '''Matrix-free solver for the gravity wave system with orography

        :arg mixed_operator: Mixed operator (:class:`.Mutilde`
        :arg mutilde: Modfied velocity mass matrix (:class:`.Mutilde`)
        :arg Wb: Function space for buoyancy field :math:`b`
        :arg ksp_type: String describing the PETSc KSP solver (e.g. ``gmres``)
        :arg pressure_solver: Solver for Schur complement pressure system.
            This is an instance of :class:`.IterativeSolver`
        :arg schur_diagonal_only: Only use the diagonal part in the 
            Schur complement preconditioner, see :class:`MixedPreconditioner`.
        :arg ksp_monitor: KSP monitor instance, see e.g. :class:`KSPMonitor`
        :arg maxiter: Maximal number of iterations for outer iteration
        :arg tolerance: Tolerance for outer iteration
    '''
    def __init__(self,
                 Wb,
                 mixed_operator,
                 mutilde,
                 ksp_type='gmres',
                 schur_diagonal_only=False,
                 ksp_monitor=None,
                 maxiter=100,
                 tolerance=1.E-6,
                 pressure_solver=None):
        mixed_array = MixedArray(W2,W3,Wb)
        mixed_preconditioner = MixedPreconditionerOrography(mixed_operator,
                                                            mutilde,
                                                            Wb,
                                                            pressure_solver,
                                                            schur_diagonal_only)
        super(MatrixFreeSolverOrography,self).__init__(Wb,
                                                       mixed_array,
                                                       mixed_operator,
                                                       mixed_preconditioner,
                                                       ksp_type,
                                                       schur_diagonal_only,
                                                       ksp_monitor,
                                                       maxiter,
                                                       tolerance,
                                                       pressure_solver)
        

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

        return self._u, self._p, self._b

class PETScSolver(object):
    '''Solver with PETSc preconditioner.

        :arg W2: HDiv Function space for velocity field :math:`\\vec{u}`
        :arg W3: L2 Function space for pressure field :math:`p`
        :arg Wb: Function space for buoyancy field :math:`b`
        :arg ksp_type: String describing the PETSc KSP solver (e.g. ``gmres``)
        :arg dt: Positive real number, time step size :math:`\Delta t`
        :arg c: Positive real number, speed of sound waves
        :arg N: Positive real number, buoyancy frequency
        :arg ksp_monitor: KSP monitor instance, see e.g. :class:`KSPMonitor`
        :arg maxiter: Maximal number of iterations for outer iteration
        :arg tolerance: Tolerance for outer iteration
    '''
    def __init__(self,
                 W2,W3,Wb,
                 dt,c,N,
                 ksp_type='gmres',
                 ksp_monitor=None,
                 maxiter=100,
                 tolerance=1.E-6,
                 multigrid=True):
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
        self._ksp_monitor = ksp_monitor
        self._multigrid = multigrid
        self._Wmixed = W2 * W3
        self._W2 = self._Wmixed.sub(0)
        self._W3 = self._Wmixed.sub(1)
            
        # Set up test- and trial function spaces
        self._u = Function(self._W2)
        self._p = Function(self._W3)
        self._b = Function(self._Wb)
        self._dx = self._W3._mesh._dx
        self.vert_norm = VerticalNormal(self._W3.mesh())

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
        e.set("ksp_type",str(self._ksp_type))
        e.set("maxiter",str(self._maxiter))
        e.set("tolerance",str(self._tolerance))

    def up_solver_setup(self,r_u,r_p,r_b,vmixed):
        '''Set up the solver for the mixed system for given RHS.

            This method constructs the LinearVariationalProblem and the
            LinearVariationalSolver for the mixed system. Setup is moved to a separate
            method so that we can extract the solver without actually solving.

            :arg r_u: RHS for velocity
            :arg r_p: RHS for pressure
            :arg r_p: RHS for buoyancy
            :arg vmixed: Mixed solution vector (output)
        '''
        # Calculate RHS
        utest, ptest = TestFunctions(self._Wmixed)
        utrial, ptrial = TrialFunctions(self._Wmixed)
        bcs = [DirichletBC(self._Wmixed.sub(0), 0.0, "bottom"),
               DirichletBC(self._Wmixed.sub(0), 0.0, "top")]
        if (self._multigrid):
            pc_1_params = {'fieldsplit_1_pc_type': 'hypre',
                           'fieldsplit_1_pc_hypre_type': 'boomeramg',
                           'fieldsplit_1_pc_hypre_boomeramg_max_iter':1,
                           'fieldsplit_1_pc_hypre_boomeramg_agg_nl':0,
                           'fieldsplit_1_pc_hypre_boomeramg_coarsen_type':'Falgout',
                           'fieldsplit_1_pc_hypre_boomeramg_smooth_type':'Euclid',
                           'fieldsplit_1_pc_hypre_boomeramg_eu_bj':1,
                           'fieldsplit_1_pc_hypre_boomeramg_interptype':'classical', 
                           'fieldsplit_1_pc_hypre_boomeramg_P_max':0,
                           'fieldsplit_1_pc_hypre_boomeramg_agg_nl':0,
                           'fieldsplit_1_pc_hypre_boomeramg_strong_threshold':0.25,
                           'fieldsplit_1_pc_hypre_boomeramg_max_levels':25,
                           'fieldsplit_1_pc_hypre_boomeramg_no_CF':False}
        else:
            pc_1_params = {'fieldsplit_1_pc_type':'bjacobi',
                           'fieldsplit_1_sub_pc_type':'ilu'}

        sparams={'pc_type': 'fieldsplit',
                 'pc_fieldsplit_type': 'schur',
                 'ksp_type': 'gmres',
                 'ksp_max_it': 1000,
                 'ksp_rtol':self._tolerance,
                 'ksp_monitor': False,
                 'pc_fieldsplit_schur_fact_type': 'FULL',
                 'pc_fieldsplit_schur_precondition': 'selfp',
                 'fieldsplit_0_ksp_type': 'preonly',
                 'fieldsplit_0_pc_type': 'bjacobi',
                 'fieldsplit_0_sub_pc_type': 'ilu',
                 'fieldsplit_1_ksp_type': 'preonly'}
        sparams.update(pc_1_params)
        a_up = (  ptest*ptrial \
                + self._dt_half_c2*ptest*div(utrial) \
                - self._dt_half*div(utest)*ptrial \
                + (dot(utest,utrial) + self._omega_N2 \
                    * dot(utest,self.vert_norm.zhat) \
                    * dot(utrial,self.vert_norm.zhat)) \
               )*self._dx
        L_up = ( dot(utest,r_u) + self._dt_half*dot(utest,self.vert_norm.zhat*r_b) \
               + ptest*r_p) * self._dx
        up_problem = LinearVariationalProblem(a_up, L_up, vmixed, bcs=bcs)
        up_solver = LinearVariationalSolver(up_problem, solver_parameters=sparams)
        ksp = up_solver.snes.getKSP()
        ksp.setMonitor(self._ksp_monitor)
        return up_solver

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
        vmixed = Function(self._Wmixed)
        with timed_region('petsc solver setup'):
            self.up_solver = self.up_solver_setup(r_u,r_p,r_b,vmixed)
        with self._ksp_monitor:
            self.up_solver.solve()
        with timed_region('petsc buoyancy solve'):
            self._u.assign(vmixed.sub(0))
            self._p.assign(vmixed.sub(1))
            btest = TestFunction(self._Wb)
            L_b = dot(btest*self.vert_norm.zhat,self._u)*self._dx
            a_b = btest*TrialFunction(self._Wb)*self._dx
            b_tmp = Function(self._Wb)
            b_problem = LinearVariationalProblem(a_b,L_b, b_tmp)
            b_solver = LinearVariationalSolver(b_problem,solver_parameters={'ksp_type':'cg',
                                                                        'pc_type':'jacobi'})
            b_solver.solve()
            self._b = assemble(r_b-self._dt_half_N2*b_tmp)
        return self._u, self._p, self._b

