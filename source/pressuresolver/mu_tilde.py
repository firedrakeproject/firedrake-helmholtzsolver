from firedrake import *
import sys, petsc4py
import numpy as np
from vertical_normal import *
from lumpedmass import *
from auxilliary.ksp_monitor import *

petsc4py.init(sys.argv)

from petsc4py import PETSc


class Mutilde(object):
    '''Class representing the operator :math:`\\tilde{M}_u`.

        The operator is defined as

        .. math::
            \\tilde{M}_u = M_u+\omega_N^2 Q M_b^{-1} Q^T

        where :math:`M_u` and :math:`M_b` are the mass matrices in velocity and 
        buoyancy space and the matrix Q is defined as
    
        .. math::
            Q_{ij} = \langle w_i,\gamma_j \hat{z}\\rangle

        where :math:`w_i` and :math:`\gamma_j` are basis functions in the 
        velocity and buoyancy spaces.

        In the absence of orography the buoyancy can be eliminated pointwise from
        the mixed system of equations. In this case :math:`Q=Q^T=M_b` and the matrix
        reduces to

        .. math::
            \\tilde{M}_u = M_u+\omega_N^2 M_b 

        i.e. the matrix application does not require an inverse of :math:`M_b`.
        In addition, in this case the matrix can be assembled explicitly and
        a simple Jacobi-preconditioner can be used.           
        
        This class defines methods for applying the matrix :math:`\\tilde{M}_u` and
        a PETSc interface, can be used to (approximately) invert the matrix
        via an iterative PETSc solver.

        :arg W2: HDiv function space for velocity
        :arg Wb: Function space for buoyancy
        :arg omega_N: Positive constant related to buoyancy frequency,
            :math:`\omega_N=\\frac{\Delta t}{2}N`
        :arg lumped: Lump mass matrix
        :arg maxiter_b: Maximal number of iterations for buoyancy mass solve
        :arg tolerance_u: Tolerance for :math:`\\tilde{M}_u` solve
        :arg maxiter_u: Maximal number of iterations for :math:`\\tilde{M}_u` solve
    '''
    def __init__(self,W2,Wb,omega_N,
                 lumped=True,
                 tolerance_u=1.E-5,maxiter_u=1000):
        self._W2 = W2
        self._Wb = Wb
        self._lumped = lumped
        self._mesh = self._W2.mesh()
        self._omega_N = omega_N
        self._omega_N2 = Constant(self._omega_N**2)
        self._tolerance_u = tolerance_u
        self._maxiter_u = maxiter_u
        self._u_tmp = Function(self._W2)
        self._res_tmp = Function(self._W2)
        self._u_test = TestFunction(self._W2)
        self._dx = self._mesh._dx
        vertical_normal = VerticalNormal(self._mesh)
        self._zhat = vertical_normal.zhat
        self._bcs = [DirichletBC(self._W2, 0.0, "bottom"),
                     DirichletBC(self._W2, 0.0, "top")]
        self._u_trial = TrialFunction(self._W2)
        ufl_form = (dot(self._u_test,self._u_trial) + \
                                 self._omega_N2*dot(self._u_test,self._zhat) * \
                                                dot(self._u_trial,self._zhat))*self._dx
        self._Mutilde = assemble(ufl_form,bcs=self._bcs)
        self._solver_param_u = {'ksp_type':'cg',
                                'ksp_rtol':self._tolerance_u,
                                'ksp_max_it':self._maxiter_u,
                                'ksp_monitor':False,
                               'pc_type':'jacobi'}
        if (self._lumped):
            self._lumped_mass = LumpedMass(ufl_form)

    def _apply_bcs(self,u):
        '''Apply boundary conditions to velocity function.

            :arg u: Function in velocity space
        '''
        for bc in self._bcs:
            bc.apply(u)

    def apply(self,u):
        '''Multiply a velocity function with :math:`\\tilde{M}_u` and return result.
        
        :arg u: Velocity function to be multiplied by :math:`\\tilde{M}_u`.
        '''
        self._apply_bcs(u)
        tmp = assemble((dot(self._u_test,u) + \
                         self._omega_N2*dot(self._u_test,self._zhat) \
                                       *dot(self._zhat,u))*self._dx)
        self._apply_bcs(tmp)
        return tmp

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

    def divide(self,u,r_u):
        '''Multiply a velocity field by the inverse of the matrix.
        
        Calculate :math:`(\\tilde{M}_u)^{-1}u` via a CG iteration and return result

        :arg u: Velocity field to be multiplied
        :arg r_u: Resulting velocity field
        '''
        self._apply_bcs(u)
        if self._lumped:
            r_u.assign(u)
            self._lumped_mass.divide(r_u)
            self._apply_bcs(r_u)
        else:
            solve(self._Mutilde,r_u,u,
                  solver_parameters=self._solver_param_u,
                  bcs=self._bcs)
