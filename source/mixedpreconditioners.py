from firedrake import *

from mixedarray import *
from auxilliary.logger import *
from pressuresolver.vertical_normal import *
from pressuresolver.mu_tilde import *
from pyop2.profiling import timed_function, timed_region

petsc4py.init(sys.argv)

from petsc4py import PETSc

        
class MixedPreconditioner(object):
    '''Schur complement preconditioner for the mixed gravity wave system.

        Use the following algorithm to precondition the mixed gravity wave system:

        1. Calculate

        .. math::

            M_p\\tilde{\\vec{R}}_p = M_p\\vec{R}_p
                - \\frac{\Delta t}{2}D\\tilde{M}_u^{-1}(M_u\\vec{R}_u)
        
        2. Solve :math:`H\\vec{P}=(M_p\\tilde{\\vec{R}}_p)` for :math:`\\vec{P}` using
            the specified pressure solver

        3. Calculate
        
        .. math::
            \\vec{U} = \\tilde{M}_u^{-1}((M_u\\tilde{\\vec{R}}_u)
                     + \\frac{\Delta t}{2}D^T \\vec{P})

        Here :math:`\\tilde{M_u} = M_u + \omega_N^2 M_u^{(v)}` is the
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
    '''
    def __init__(self,
                 W2,W3,Wb,
                 dt,N,c,
                 pressure_solver,
                 diagonal_only=False,
                 tolerance_u=1.E-5,maxiter_u=1000):
        self._pressure_solver = pressure_solver
        self._W2 = W2
        self._W3 = W3
        self._Wb = Wb
        self._omega_N = 0.5*dt*N
        self._omega_N2 = Constant(0.5*dt*N)
        self._dt_half = Constant(0.5*dt)
        self._dt_half_N2 = Constant(0.5*dt*N**2)
        self._dt_half_c2 = Constant(0.5*dt*c**2)
        self._diagonal_only = diagonal_only
        self._mesh = self._W3._mesh
        self._zhat = VerticalNormal(self._mesh)
        self._dx = self._mesh._dx
        self._bcs = [DirichletBC(self._W2, 0.0, "bottom"),
                     DirichletBC(self._W2, 0.0, "top")]
        self._utest = TestFunction(self._W2)
        self._ptest = TestFunction(self._W3)
        self._btest = TestFunction(self._Wb)
        self._mutilde = Mutilde(self._W2,self._Wb,self._omega_N,
                                lumped=self._pressure_solver._operator._mutilde._lumped,
                                tolerance_u=tolerance_u,maxiter_u=maxiter_u)
        # Temporary functions
        self._rtilde_u = Function(self._W2)
        self._rtilde_p = Function(self._W3)
        self._u = Function(self._W2)
        self._p = Function(self._W3)
        self._tmp_u = Function(self._W2)
        self._Pu = Function(self._W2)
        self._Pp = Function(self._W3)
        self._mixedarray = MixedArray(self._W2,self._W3)
        
    @timed_function("mixed_preconditioner") 
    def solve(self,r_u,r_p,u,p):
        '''Preconditioner solve.

        Given the RHS r_u and r_p, calculate the fields u and p 

        :arg r_u: RHS in velocity space
        :arg r_p: RHS in pressure space
        :arg u: Solution for velocity
        :arg p: Solution for pressure
        '''
       
        if (self._diagonal_only):
            # Pressure solve
            p.assign(0.0)
            self._pressure_solver.solve(r_p,p)
            # Velocity solve
            self._mutilde.divide(r_u,u)
        else:
            # Modified RHS for pressure
            with timed_region('mutilde_divide'):
                self._mutilde.divide(r_u,self._tmp_u)
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
            with timed_region('mutilde_divide'):
                self._mutilde.divide(self._tmp_u,u)

    def apply(self,pc,x,y):
        '''PETSc interface for preconditioner solve.

        PETSc interface wrapper for the :func:`solve` method.

        :arg x: PETSc vector representing the mixed right hand side
        :arg y: PETSc vector representing the mixed solution vector
        '''
        
        with self._u.dat.vec as u, \
             self._p.dat.vec as p:
            self._mixedarray.split(x,u,p)
        self.solve(self._u,self._p,self._Pu,self._Pp)
        with self._Pu.dat.vec_ro as u, \
             self._Pp.dat.vec_ro as p:
            self._mixedarray.combine(y,u,p)
        
