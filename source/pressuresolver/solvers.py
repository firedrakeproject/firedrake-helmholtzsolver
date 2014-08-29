import xml.etree.cElementTree as ET
from operators import *
import sys, petsc4py
import numpy as np
from mpi_utils import Logger
from pyop2.profiling import timed_region

petsc4py.init(sys.argv)

from petsc4py import PETSc

class IterativeSolver(object):
    '''Abstract iterative solver base class.

    The solver converges if the relative residual has been reduced by at least a
    factor tolerance.

    :arg operator: Instance :math:`H` of linear Schur complement
        :class:`.Operator` in pressure space
    :arg preconditioner: Instance :math:`P` of :class:`.Preconditioner`
    :arg maxiter: Maximal number of iterations
    :arg tolerance: Relative tolerance for solve
    '''
    def __init__(self,operator,
                 preconditioner,
                 maxiter=100,
                 tolerance=1.E-6):
        self.operator = operator
        self.V_pressure = self.operator.V_pressure
        self.V_velocity = self.operator.V_velocity
        self.preconditioner = preconditioner
        self.maxiter = maxiter
        self.tolerance = tolerance
        self.dx = self.V_pressure.mesh()._dx

    def solve(self,b,phi):
        '''Solve linear system :math:`H\phi = b`.

        :arg b: Right hand side :math:`b` in pressure space
        :arg phi: State vector :math:`\phi` in pressure space
        '''
        pass

class PETScSolver(IterativeSolver):
    '''PETSc solver.

    This solver uses the PETSc solver algorithms, the exact algorithm is
    set at runtime.
    
    :arg operator: Instance :math:`H` of linear Schur complement
        :class:`.Operator` in pressure space
    :arg preconditioner: Instance :math:`P` of :class:`.Preconditioner`
    :arg maxiter: Maximal number of iterations
    :arg tolerance: Relative tolerance for solve
    '''
    def __init__(self,operator,
                 preconditioner,
                 ksp_monitor,
                 maxiter=100,
                 tolerance=1.E-6):
        super(PETScSolver,self).__init__(operator,preconditioner,maxiter,tolerance)
        self.logger = Logger()
        n = self.operator.V_pressure.dof_dset.size
        self.u = PETSc.Vec()
        self.u.create()
        self.u.setSizes((n, None))
        self.u.setFromOptions()
        self.rhs = self.u.duplicate()

        op = PETSc.Mat().create()
        op.setSizes(((n, None), (n, None)))
        op.setType(op.Type.PYTHON)
        op.setPythonContext(self.operator)
        op.setUp()

        self.ksp = PETSc.KSP()
        self.ksp.create()
        self.ksp.setOptionsPrefix('pressure_')
        self.ksp.setOperators(op)
        self.ksp.setTolerances(rtol=self.tolerance,max_it=self.maxiter)
        self.ksp.setFromOptions()
        self.logger.write('  Pressure KSP type = '+str(self.ksp.getType()))
        self.ksp_monitor = ksp_monitor
        self.ksp.setMonitor(self.ksp_monitor)
        pc = self.ksp.getPC()
        pc.setType(pc.Type.PYTHON)
        pc.setPythonContext(self.preconditioner)

    def add_to_xml(self,parent,function):
        '''Add to existing xml tree.

        :arg parent: Parent node to be added to
        :arg function: Function of object
        '''
        e = ET.SubElement(parent,function)
        e.set("type",type(self).__name__)
        self.operator.add_to_xml(e,'operator')
        self.preconditioner.add_to_xml(e,'preconditioner')
        e.set("ksp_type",str(self.ksp.getType()))
        e.set("maxiter",str(self.maxiter))
        e.set("tolerance",str(self.tolerance))
       
    def solve(self,b,phi):
        '''Solve linear system :math:`H\phi = b`.

        :arg b: Right hand side :math:`b` in pressure space
        :arg phi: State vector :math:`\phi` in pressure space
        '''
        with b.dat.vec_ro as v:
            self.rhs.array[:] = v.array[:]
        with self.ksp_monitor, timed_region('pressure_solve'):
            self.ksp.solve(self.rhs,self.u)
        with phi.dat.vec as v:
            v.array[:] = self.u.array[:]
