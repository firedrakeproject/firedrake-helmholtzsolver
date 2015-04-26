from firedrake import *
import sys, petsc4py
import numpy as np
from vertical_normal import *
from lumpedmass import *
from auxilliary.ksp_monitor import *
from pyop2.profiling import timed_function, timed_region

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

        :arg mixed_operator: Mixed operator containing the assembled matrix for the
            mixed system
        :arg omega_N: Positive constant related to buoyancy frequency,
            :math:`\omega_N=\\frac{\Delta t}{2}N`
        :arg lumped: Lump mass matrix
        :arg tolerance_u: Tolerance for :math:`\\tilde{M}_u` solve
        :arg maxiter_u: Maximal number of iterations for :math:`\\tilde{M}_u` solve
    '''
    def __init__(self,mixed_operator,
                 lumped=True,
                 tolerance_u=1.E-5,maxiter_u=1000):
        self._lumped = lumped
        self._tolerance_u = tolerance_u
        self._maxiter_u = maxiter_u
        self._mixed_operator = mixed_operator
        self._bcs = mixed_operator._bcs
        self._tmp_u = Function(mixed_operator._W2)
        self._tmp_v = Function(mixed_operator._W2)
        ufl_form = mixed_operator.form_uu
        if (self._lumped):
            self._lumped_mass = LumpedMass(ufl_form)
        else:
            solver_param = {'ksp_type':'preonly',
                            'ksp_rtol':self._tolerance_u,
                            'ksp_max_it':self._maxiter_u,
                            'ksp_monitor':False,
                            'pc_type':'bjacobi',
                            'sub_pc_type':'ilu'}
            if (mixed_operator._preassemble):
                self._mutilde = mixed_operator._op_uu
            else:
                self._mutilde = assemble(ufl_form,bcs=self._bcs)
            linearsolver = LinearSolver(self._mutilde,solver_parameters=solver_param)
            self._ksp = linearsolver.ksp

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
        if (self._lumped):
            self._tmp_u.assign(u)
            self._lumped_mass.multiply(self._tmp_u) 
        else:
            with self._tmp_u.dat.vec as v:
                with u.dat.vec_ro as x:
                    self._mutilde.M.handle.mult(x,v)
        self._apply_bcs(self._tmp_u)
        return self._tmp_u

    def mult(self,mat,x,y):
        '''PETSc interface for operator application.

        PETSc interface wrapper for the :func:`apply` method.

        :arg x: PETSc vector representing the field to be multiplied.
        :arg y: PETSc vector representing the result.
        '''
        with self._tmp_u.dat.vec as v:
            v.array[:] = x.array[:]
        self._tmp_v = self.apply(self._tmp_u)
        with self._tmp_v.dat.vec_ro as v:
            y.array[:] = v.array[:]

    def divide(self,u,r_u,tolerance=None,preonly=True):
        '''Multiply a velocity field by the inverse of the matrix.
        
        Calculate :math:`(\\tilde{M}_u)^{-1}u` via a CG iteration and return result

        :arg u: Velocity field to be multiplied
        :arg r_u: Resulting velocity field
        '''
        self._apply_bcs(u)
        if self._lumped:
            r_u.assign(u)
            self._lumped_mass.divide(r_u)
        else:
            if (tolerance != None):
                old_tolerance = self._ksp.rtol
                self._ksp.rtol = tolerance
            if (not preonly):
                old_type = self._ksp.type
                self._ksp.type = 'cg'
            with u.dat.vec_ro as v:
                with r_u.dat.vec as x:
                    self._ksp.solve(v,x)
            if (tolerance != None):
                self._ksp.rtol = old_tolerance
            if (not preonly):
                self._ksp.type = old_type
        self._apply_bcs(r_u)

