import numpy as np
from firedrake import *
import xml.etree.cElementTree as ET
from firedrake.petsc import PETSc
from vertical_normal import VerticalNormal
import sys

class Smoother(object):
    '''Base class for smoother for pressure system
    
    :arg W3: pressure space
    '''
    def __init__(self,operator):
        self._operator = operator
        self._W3 = self._operator._W3
        self._mesh = self._W3.mesh()
        self._b_tmp = Function(self._W3)
        self._phi_tmp = Function(self._W3)
        with self._b_tmp.dat.vec as v:
            ndof = self._W3.dof_dset.size
            self._iset = PETSc.IS().createStride(ndof,
                                                 first=v.owner_range[0],
                                                 step=1,
                                                 comm=v.comm)

class DirectSolver(Smoother):
    def __init__(self, operator, W2, dt, c, N, solver_type, op_H=None):
        super(DirectSolver,self).__init__(operator)
        self._solver_type = solver_type
        if not ( ( self._solver_type == "boomeramg") or \
                 (self._solver_type == "mumps" ) ):
            if (op2.MPI.comm.rank == 0):
                print "ERROR: Direct solver has to be either \'boomeramg\' or \'mumps\'"
            sys.exit()
        self._dx = self._mesh._dx
        utest = TestFunction(W2)
        utrial = TrialFunction(W2)
        ptest = TestFunction(self._W3)
        ptrial = TrialFunction(self._W3)

        # FIXME: Is this the right operator?
        pmass = ptest*ptrial*self._dx

        omega_c2 = (0.5*dt*c)**2
        omega_N2 = Constant((0.5*dt*N)**2)
        Div = ptest*div(utrial)*self._dx
        Grad = -div(utest)*ptrial*self._dx

        zhat = VerticalNormal(W2.mesh()).zhat

        ubcs = [DirichletBC(W2, 0.0, "bottom"),
                DirichletBC(W2, 0.0, "top")]

        umass = (dot(utest, utrial) +
                 omega_N2*dot(utest, zhat)*dot(utrial, zhat))*self._dx

        S = assemble(pmass).M.handle.copy()
        U = assemble(umass, bcs=ubcs).M.handle

        Div = assemble(Div).M.handle
        Grad = assemble(Grad).M.handle.copy()

        Udiaginv = U.getDiagonal()
        Udiaginv.reciprocal()

        Grad.diagonalScale(Udiaginv)
        divgrad = Div.matMult(Grad)

        S.axpy(-omega_c2, divgrad)

        solver = PETSc.KSP().create()
        if op_H is not None:
            A = PETSc.Mat().create(op2.MPI.comm)
            A.setSizes(S.getSizes())
            A.setType(A.Type.PYTHON)
            A.setPythonContext(op_H)
            A.setUp()
        else:
            A = S
        solver.setOperators(A, S)

        solver.setOptionsPrefix("coarse_solver_")
        opts = PETSc.Options()
        if (self._solver_type == "boomeramg"):
            # Set options to use one AMG V-cycle
            opts["coarse_solver_ksp_type"] = "preonly"
            opts["coarse_solver_pc_type"] = "hypre"
            opts["coarse_solver_pc_hypre_type"] = "boomeramg"
            opts["coarse_solver_pc_hypre_boomeramg_max_iter"] = 1
            opts["coarse_solver_pc_hypre_boomeramg_agg_nl"] = 0
            opts["coarse_solver_pc_hypre_boomeramg_coarsen_type"] = "Falgout"
            opts["coarse_solver_pc_hypre_boomeramg_smooth_type"] = "Euclid"
            opts["coarse_solver_pc_hypre_boomeramg_eu_bj"] = 1
            opts["coarse_solver_pc_hypre_boomeramg_interptype"] = "classical"
            opts["coarse_solver_pc_hypre_boomeramg_P_max"] = 0
            opts["coarse_solver_pc_hypre_boomeramg_strong_threshold"] = 0.25
            opts["coarse_solver_pc_hypre_boomeramg_max_level"] = 5
            opts["coarse_solver_pc_hypre_boomeramg_no_CF"] = 0
            solver.setFromOptions()
        elif (self._solver_type=="mumps"):
            opts["coarse_solver_ksp_type"] = "preonly"
            opts["coarse_solver_pc_type"] = "cholesky"
            solver.setFromOptions()
            pc = solver.getPC()
            pc.setFactorSolverPackage("mumps")
        self.ksp = solver
        self.ksp.setTolerances(rtol=1.E-15)

    def solve(self, b, phi):
        with b.dat.vec_ro as B:
            with phi.dat.vec as x:
                self.ksp.solve(B, x)

    def apply(self,pc,x,y):
        '''PETSc interface for preconditioner solve.

        PETSc interface wrapper for the :func:`solve` method.

        :arg x: PETSc vector representing the right hand side in pressure
            space
        :arg y: PETSc vector representing the solution pressure space.
        '''
        with self._b_tmp.dat.vec as v:
            tmp = x.getSubVector(self._iset)
            x.copy(v)
            x.restoreSubVector(self._iset, tmp)
        self.solve(self._b_tmp,self._phi_tmp)
        with self._phi_tmp.dat.vec_ro as v:
            y.array[:] = v.array[:]

    def add_to_xml(self, parent, function):
        pass


class Jacobi(Smoother):
    '''Jacobi smoother.

    Jacobi smoother for the linear Schur complement system.

    :arg operator: Schur complement operator, of type :class:`Operator_Hhat`.
    :arg mu_relax: Under-/Over-relaxation parameter :math:`mu`
    :arg n_smooth: Number of smoothing steps to apply in method
        :class:`smooth()`.
    '''
    def __init__(self,operator,
                 mu_relax=4./5.,
                 n_smooth=1,
                 level=-1,
                 *args):
        super(Jacobi,self).__init__(operator)
        self._mu_relax = mu_relax
        self._n_smooth = n_smooth
        self._dx = self._mesh._dx
        self._r_tmp = Function(self._W3)
        self._level = level

    def add_to_xml(self,parent,function):
        '''Add to existing xml tree.

        :arg parent: Parent node to be added to
        :arg function: Function of object
        '''
        e = ET.SubElement(parent,function)
        e.set("type",type(self).__name__)
        self._operator.add_to_xml(e,'operator')
        e.set("mu_relax",str(self._mu_relax))
        e.set("n_smooth",str(self._n_smooth))
       
    def solve(self,b,phi):
        '''Solve approximately with RHS :math:`b`.
        
        Repeatedy apply the smoother to solve the equation :math:`H\phi=b`
        approximately.

        :arg b: Right hand side :math:`b` in pressure space
        :arg phi: State vector :math:`\phi` in pressure space (out)
        '''
        self.smooth(b,phi,initial_phi_is_zero=True)

    def apply(self,pc,x,y):
        '''PETSc interface for preconditioner solve.

        PETSc interface wrapper for the :func:`solve` method.

        :arg x: PETSc vector representing the right hand side in pressure
            space
        :arg y: PETSc vector representing the solution pressure space.
        '''
        with self._b_tmp.dat.vec as v:
            tmp = x.getSubVector(self._iset)
            x.copy(v)
            x.restoreSubVector(self._iset, tmp)
        self.smooth(self._b_tmp,self._phi_tmp,initial_phi_is_zero=True)
        with self._phi_tmp.dat.vec_ro as v:
            y.array[:] = v.array[:]

    def smooth(self,b,phi,initial_phi_is_zero=False):
        '''Smooth.
        
        Apply the smoother 
        
        .. math::

            \phi \mapsto \phi + \mu \left(\hat{H}_z\\right)^{-1} (b-\hat{H}\phi)
            
        repeatedly to the state vector :math:`\phi`.
        If :class:`initial_phi_is_zero` is True, then the initial :math:`\phi`
        is assumed to be zero and in the first iteration the updated
        :math:`\phi` is just given by :math:`\left(\hat{H}_z\\right)^{-1}b`.

        :arg b: Right hand side :math:`b` in pressure space
        :arg phi: State vector :math:`\phi` in pressure space (out)
        :arg initial_phi_is_zero: Initialise with :math:`\phi=0`.
        '''
        for i in range(self._n_smooth):
            if ( (i==0) and (initial_phi_is_zero)):
                self._r_tmp.assign(b)
            else:
                self._r_tmp.assign(self._operator.residual(b,phi))
            # Apply inverse diagonal r -> \left(\hat{H}_z\right)^{-1} *r
            self._operator.apply_blockinverse(self._r_tmp)
            # Update phi
            if ( (i ==0) and (initial_phi_is_zero) ):
                self._r_tmp *= self._mu_relax
                phi.assign(self._r_tmp)
            else:
                phi.assign(phi+self._mu_relax*self._r_tmp)
