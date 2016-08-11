import xml.etree.cElementTree as ET
import sys, petsc4py
import numpy as np
from auxilliary.logger import *
from pyop2.profiling import timed_region

petsc4py.init(sys.argv)

from petsc4py import PETSc

class IterativeSolver(object):
    '''Abstract iterative solver base class.

    The solver converges if the relative residual has been reduced by at least a
    factor tolerance.

    :arg operator: Instance :math:`\hat{H}` of linear Schur complement
        :class:`.OperatorH` in pressure space
    :arg preconditioner: Preconditioner, e.g. :class:`.hMultigrid` or :class:`.hpMultigrid`
    :arg maxiter: Maximal number of iterations
    :arg tolerance: Relative tolerance for solve
    '''
    def __init__(self,operator,
                 preconditioner,
                 maxiter=100,
                 tolerance=1.E-5):
        self._operator = operator
        self._W3 = self._operator._W3
        self._preconditioner = preconditioner
        self._maxiter = maxiter
        self._tolerance = tolerance
        from firedrake import dx
        self._dx = dx(domain=self._W3.mesh())

    def solve(self,b,phi):
        '''Solve linear system :math:`\hat{H}\phi = b`.

        :arg b: Right hand side :math:`b` in pressure space
        :arg phi: State vector :math:`\phi` in pressure space
        '''
        pass

class PETScSolver(IterativeSolver):
    '''PETSc solver.

    This solver uses the PETSc solver algorithms, the exact algorithm is
    set at runtime.
    
    :arg operator: Instance :math:`H` of linear Schur complement
        :class:`.OperatorH` in pressure space
    :arg preconditioner: Preconditioner, e.g. :class:`.hMultigrid` or :class:`.hpMultigrid`
    :arg ksp_type: Name of iterative PETSc solver, e.g. ``cg``
    :arg ksp_monitor: KSP monitor class, e.g. instance of :class:`.KSPMonitor`
    :arg maxiter: Maximal number of iterations
    :arg tolerance: Relative tolerance for solve
    '''
    def __init__(self,
                 operator,
                 preconditioner,
                 ksp_type,
                 ksp_monitor,
                 maxiter=100,
                 tolerance=1.E-6):
        super(PETScSolver,self).__init__(operator,preconditioner,maxiter,tolerance)
        self._ksp_type = ksp_type
        self._logger = Logger()
        n = self._operator._W3.dof_dset.size
        self._u = PETSc.Vec()
        self._u.create()
        self._u.setSizes((n, None))
        self._u.setFromOptions()
        self._rhs = self._u.duplicate()

        op = PETSc.Mat().create()
        op.setSizes(((n, None), (n, None)))
        op.setType(op.Type.PYTHON)
        op.setPythonContext(self._operator)
        op.setUp()

        self._ksp = PETSc.KSP()
        self._ksp.create()
        self._ksp.setOptionsPrefix('pressure_')
        self._ksp.setOperators(op)
        self._ksp.setTolerances(rtol=self._tolerance,max_it=self._maxiter)
        self._ksp.setType(self._ksp_type)
        self._logger.write('  Pressure KSP type = '+str(self._ksp.getType()))
        self._ksp_monitor = ksp_monitor
        self._ksp.setMonitor(self._ksp_monitor)
        pc = self._ksp.getPC()
        pc.setType(pc.Type.PYTHON)
        pc.setPythonContext(self._preconditioner)

    def add_to_xml(self,parent,function):
        '''Add to existing xml tree.

        :arg parent: Parent node to be added to
        :arg function: Function of object
        '''
        e = ET.SubElement(parent,function)
        e.set("type",type(self).__name__)
        self._operator.add_to_xml(e,'operator')
        self._preconditioner.add_to_xml(e,'preconditioner')
        e.set("ksp_type",str(self._ksp.getType()))
        e.set("maxiter",str(self._maxiter))
        e.set("tolerance",str(self._tolerance))
       
    def solve(self,b,phi):
        '''Solve linear system :math:`H\phi = b`.

        :arg b: Right hand side :math:`b` in pressure space
        :arg phi: State vector :math:`\phi` in pressure space
        '''
        with self._ksp_monitor, timed_region('matrixfree pc_schur'):
            with b.dat.vec_ro as v:
                self._rhs.array[:] = v.array[:]
            self._ksp.solve(self._rhs,self._u)
            with phi.dat.vec as v:
                v.array[:] = self._u.array[:]
