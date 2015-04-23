from firedrake import *

from mixedarray import *
from auxilliary.logger import *
from pressuresolver.vertical_normal import *
from pressuresolver.mu_tilde import *

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
    
        :arg mixed_operator: Mixed operator (:class:`.Mutilde`
        :arg mutilde: Modfied velocity mass matrix (:class:`.Mutilde`)
        :arg Wb: Function space for buoyancy 
        :arg pressure_solver: Solver in pressure space
        :arg diagonal_only: Only use diagonal matrix, ignore forward/backward
            substitution with triagular matrices
        :arg preassemble: Preassemble the operators for building the modified RHS
            and for the back-substitution
    '''
    def __init__(self,
                 mixed_operator,
                 mutilde,
                 Wb,
                 pressure_solver,
                 diagonal_only=False,
                 tolerance_u=1.E-5,maxiter_u=1000):
        self._pressure_solver = pressure_solver
        self._mixed_operator = mixed_operator

        self._W2 = mixed_operator._W2
        self._W3 = mixed_operator._W3
        self._Wb = Wb

        self._dt_half = mixed_operator._dt_half
        self._dt_half_N2 = mixed_operator._dt_half_N2
        self._dt_half_c2 = mixed_operator._dt_half_c2
        self._diagonal_only = diagonal_only
        self._preassemble = mixed_operator._preassemble
        self._mesh = self._W3._mesh
        self._zhat = VerticalNormal(self._mesh)
        self._dx = self._mesh._dx
        self._bcs = [DirichletBC(self._W2, 0.0, "bottom"),
                     DirichletBC(self._W2, 0.0, "top")]
        self._utest = TestFunction(self._W2)
        self._ptest = TestFunction(self._W3)
        self._btest = TestFunction(self._Wb)
        self._mutilde = mutilde
        # Temporary functions
        self._rtilde_u = Function(self._W2)
        self._rtilde_p = Function(self._W3)
        self._u = Function(self._W2)
        self._p = Function(self._W3)
        self._tmp_u = Function(self._W2)
        self._Pu = Function(self._W2)
        self._Pp = Function(self._W3)
        self._mixedarray = MixedArray(self._W2,self._W3)
        self._tolerance_u = tolerance_u
        
    @timed_function("matrixfree mixed preconditioner") 
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
            with timed_region('matrixfree pc_hdiv'):
                self._mutilde.divide(r_u,u,tolerance=self._tolerance_u)
        else:
            # Modified RHS for pressure
            with timed_region('matrixfree pc_hdiv'):
                self._mutilde.divide(r_u,self._tmp_u,tolerance=self._tolerance_u)
            if (self._preassemble):
                with self._rtilde_p.dat.vec as v:
                    with self._tmp_u.dat.vec_ro as x:
                        self._mixed_operator._mat_pu.mult(x,v)
                    v *= -1.0
            else:
                assemble(- self._dt_half_c2 * self._ptest * div(self._tmp_u) * self._dx,
                         tensor=self._rtilde_p)
            self._rtilde_p += r_p

            # Pressure solve
            p.assign(0.0)
            self._pressure_solver.solve(self._rtilde_p,p)
            # Backsubstitution for velocity 
            if (self._preassemble):
                with self._tmp_u.dat.vec as v:
                    with p.dat.vec_ro as x:
                        self._mixed_operator._mat_up.mult(x,v)
                        v *= -1.0                    
            else:
                assemble(self._dt_half * div(self._utest) * p*self._dx,
                    tensor=self._tmp_u)                    
            self._tmp_u += r_u
            with timed_region('matrixfree pc_hdiv'):
                self._mutilde.divide(self._tmp_u,u,tolerance=self._tolerance_u)

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

class MixedPreconditionerOrography(object):
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
    
        :arg mixed_operator: Mixed operator (:class:`.Mutilde`
        :arg mutilde: Modfied velocity mass matrix (:class:`.Mutilde`)
        :arg Wb: Function space for buoyancy
        :arg pressure_solver: Solver in pressure space
        :arg diagonal_only: Only use diagonal matrix, ignore forward/backward
            substitution with triagular matrices
    '''
    def __init__(self,
                 mixed_operator,
                 mutilde,
                 Wb,
                 pressure_solver,
                 diagonal_only=False,
                 tolerance_b=1.E-5,maxiter_b=1000,
                 tolerance_u=1.E-5,maxiter_u=1000):
        self._pressure_solver = pressure_solver
        self._W2 = mixed_operator._W2
        self._W3 = mixed_operator._W3
        self._Wb = Wb
        self._dt_half = mixed_operator._dt_half
        self._dt_half_N2 = mixed_operatpr._dt_half_N2
        self._dt_half_c2 = mixed_operator._dt_half_c2
        self._diagonal_only = diagonal_only
        self._mesh = self._W3._mesh
        self._zhat = VerticalNormal(self._mesh)
        self._dx = self._mesh._dx
        self._bcs = [DirichletBC(self._W2, 0.0, "bottom"),
                     DirichletBC(self._W2, 0.0, "top")]
        self._utest = TestFunction(self._W2)
        self._ptest = TestFunction(self._W3)
        self._btest = TestFunction(self._Wb)
        # Buoyancy mass matrix
        self._mutilde = mutilde
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
        self._tolerance_u = tolerance_u
        self._mixedarray = MixedArray(self._W2,self._W3,self._Wb)
        Mb = assemble(self._btest*TrialFunction(self._Wb)*self._dx)
        self._linearsolver_b = LinearSolver(Mb,solver_parameters={'ksp_type':'cg',
                                                                  'ksp_rtol':tolerance_b,
                                                                  'ksp_max_it':maxiter_b,
                                                                  'ksp_monitor':False,
                                                                  'pc_type':'jacobi'})

    @timed_function("matrixfree mixed preconditioner") 
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
            with timed_region('matrixfree pc_hdiv'):
                self._mutilde.divide(r_u,u,tolerance=self._tolerance_u)
            # Buoyancy solve
            self._linearsolver_b.solve(b,r_b)
        else:
            # Modified RHS for velocity 
            self._linearsolver_b.solve(self._tmp_b,r_b)
            assemble(self._dt_half * dot(self._utest,self._zhat.zhat) \
                                   * self._tmp_b * self._dx,
                     tensor=self._rtilde_u)
            self._rtilde_u += r_u
            # Modified RHS for pressure
            with timed_region('matrixfree pc_hdiv'):
                self._mutilde.divide(self._rtilde_u,self._tmp_u,tolerance=self._tolerance_u)
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
            with timed_region('matrixfree pc_hdiv'):
                self._mutilde.divide(self._tmp_u,u,tolerance=self._tolerance_u)
            # Backsubstitution for buoyancy
            assemble(- self._dt_half_N2 * self._btest*dot(self._zhat.zhat,u)*self._dx,
                     tensor=self._tmp_b)
            self._tmp_b += r_b
            self._linearsolver_b.solve(b,self._tmp_b)

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
        
