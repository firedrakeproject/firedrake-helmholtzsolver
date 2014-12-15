from firedrake import *
import sys, petsc4py
import numpy as np
from vertical_normal import *
from ksp_monitor import *

petsc4py.init(sys.argv)

from petsc4py import PETSc

class VelocityMassPrec(object):
    '''Preconditioner for the modified velocity mass matrix :math:`\\tilde{M}_u`
    
    The preconditioner is the inverse of the velocity matrix 
    :math:`M_u+\omega_N^2 M_u^{(v)}`, i.e. the velocity mass matrix plus 
    :math:`\omega_N^2` times the vertical part of the mass matrix.

    :arg W2: Velocity space
    '''
    def __init__(self,W2,omega_N):
        self._W2 = W2
        self._omega_N = omega_N
        self._mesh = self._W2.mesh()
        self._dx = self._mesh._dx
        u_test = TestFunction(self._W2)
        u_trial = TrialFunction(self._W2)
        self._u_tmp = Function(self._W2)
        self._P_u_tmp = Function(self._W2)
        vertical_normal = VerticalNormal(self._mesh)
        zhat = vertical_normal.zhat
        self._Mu = assemble((dot(u_test,u_trial)
                            +self._omega_N**2*dot(u_test,zhat)*dot(u_trial,zhat))*self._dx)

    def apply(self,pc,x,y):
        '''PETSc interface for preconditioner solve.

        PETSc interface wrapper for the :func:`solve` method.

        :arg x: PETSc vector representing the right hand side in velocity space
        :arg y: PETSc vector representing the solution in pressure space.
        '''
        with self._u_tmp.dat.vec as v:
            v.array[:] = x.array[:]
        solve(self._Mu,self._P_u_tmp,self._u_tmp,
              solver_parameters={'ksp_rtol':1.E-6,
                                 'ksp_type':'cg',
                                 'pc_type':'jacobi'})
        with self._P_u_tmp.dat.vec_ro as v:
            y.array[:] = v.array[:]


class Mutilde(object):
    def __init__(self,W2,Wb,omega_N,
                 tolerance_b=1.E-12,maxiter_b=1000,
                 tolerance_u=1.E-12,maxiter_u=1000):
        '''Class representing the operator :math:`\\tilde{M}_u`.

        The operator is defined as

        :math::
            \\tilde{M}_u = M_u+\omega_N^2 Q M_b^{-1} Q^T

        where :math:`M_u` and :math:`M_b` are the mass matrices in velocity and 
        buoyancy space and the matrix Q is defined as
    
        :math::
            Q_{ij} = \langle w_i,\gamma_j \hat{z}\\rangle

        where :math:`w_i` and :math:`\gamma_j` are basis functions in the 
        velocity and buoyancy spaces.
        This class defines methods for applying the matrix :math:`\\tilde{M}_u` and
        a PETSc interface, can be used to (approximately) invert the matrix
        via an iterative PETSc solver.

        :arg W2: HDiv function space for velocity
        :arg Wb: Function space for buoyancy
        :arg omega_c: Positive constant related to density fluctuations
        :arg tolerance_b: Tolerance for buoyancy mass solve
        :arg maxiter_b: Maximal number of iterations for buoyancy mass solve
        :arg tolerance_u: Tolerance for :math:`\\tilde{M}_u` solve
        :arg maxiter_u: Maximal number of iterations for :math:`\\tilde{M}_u` solve
        '''
        self._W2 = W2
        self._Wb = Wb
        self._mesh = self._W2.mesh()
        self._omega_N = omega_N
        self._tolerance_b = tolerance_b
        self._tolerance_u = tolerance_u
        self._maxiter_b = maxiter_b
        self._maxiter_u = maxiter_u
        self._u_tmp = Function(self._W2)
        self._res_tmp = Function(self._W2)
        self._u_test = TestFunction(self._W2)
        self._b_test = TestFunction(self._Wb)
        self._b_trial = TrialFunction(self._Wb)
        self._dx = self._mesh._dx
        self._Mb = assemble(self._b_test*self._b_trial*self._dx)
        self._solver_param_b = {'ksp_type':'cg',
                                'ksp_rtol':self._tolerance_b,
                                'ksp_max_it':self._maxiter_b,
                                'ksp_monitor':False,
                                'pc_type':'jacobi'}
        n = self._W2.dof_dset.size
        self._u = PETSc.Vec()
        self._u.create()
        self._u.setSizes((n, None))
        self._u.setFromOptions()
        self._rhs = self._u.duplicate()

        op = PETSc.Mat().create()
        op.setSizes(((n, None), (n, None)))
        op.setType(op.Type.PYTHON)
        op.setPythonContext(self)
        op.setUp()

        self._ksp = PETSc.KSP()
        self._ksp.create()
        self._ksp.setOptionsPrefix('Mutilde_')
        self._ksp.setOperators(op)
        self._ksp.setTolerances(rtol=self._tolerance_u,
                                max_it=self._maxiter_u)
        self._ksp.setType('cg')

        #self._ksp.setMonitor(KSPMonitor())

        pc = self._ksp.getPC()
        pc.setType(pc.Type.PYTHON)
        velocity_mass_prec = VelocityMassPrec(self._W2,self._omega_N)
        pc.setPythonContext(velocity_mass_prec)

        vertical_normal = VerticalNormal(self._mesh)
        self._zhat = vertical_normal.zhat

    def apply(self,u):
        '''Multiply a velocity function with :math:`\\tilde{M}_u` and return result.
        
        :arg u: Velocity function to be multiplied by :math:`\\tilde{M}_u`.
        '''
        Mbinv_QT_u = Function(self._Wb)
        QT_u = assemble(dot(self._zhat*self._b_test,u)*self._dx)
        solve(self._Mb,Mbinv_QT_u,QT_u,solver_parameters=self._solver_param_b)
        Q_Mbinv_QT_u = dot(self._u_test,self._zhat*Mbinv_QT_u)*self._dx
        Mu_u = dot(self._u_test,u)*self._dx
        return assemble(Mu_u+self._omega_N**2*Q_Mbinv_QT_u)

    def mult(self,mat,x,y):
        '''PETSc interface for operator application.

        PETSc interface wrapper for the :func:`apply` method.

        :arg x: PETSc vector representing the field to be multiplied.
        :arg y: PETSc vector representing the result.
        '''
        with self._u_tmp.dat.vec as v:
            v.array[:] = x.array[:]
        self._res_tmp = self.apply(self._u_tmp)
        with self._res_tmp.dat.vec_ro as v:
            y.array[:] = v.array[:]

    def divide(self,u):
        '''Multiply a velocity field by the inverse of the matrix.
        
        Calculate :math:`(\\tilde{M}_u)^{-1}u` via a CG iteration and return result

        :arg u: Velocity field to be multiplied
        '''
        w = Function(self._W2)
        with u.dat.vec_ro as v:
            self._rhs.array[:] = v.array[:]
        self._ksp.solve(self._rhs,self._u)
        with w.dat.vec as v:
            v.array[:] = self._u.array[:]
        return w
