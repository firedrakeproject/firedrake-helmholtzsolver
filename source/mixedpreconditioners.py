from firedrake import *

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
    
        :arg W2: Hdiv function space for velocity
        :arg W3: L2 function space for velocity
        :arg Wb: Function space for buoyancy
        :arg dt: Time step size
        :arg N: Buoyancy frequency
        :arg c: Speed of sound waves
        :arg pressure_solver: Solver in pressure space
        :arg diagonal_only: Only use diagonal matrix, ignore forward/backward
            substitution with triagular matrices
        :arg use_petsc: Use PETSc to solve mixed system in :math:`(u,p)` space
    '''
    def __init__(self,
                 W2,W3,Wb,
                 dt,N,c,
                 pressure_solver,
                 diagonal_only=False,
                 matrixfree_prec=True,
                 tolerance_b=1.E-12,maxiter_b=1000,
                 tolerance_u=1.E-12,maxiter_u=1000):
        self._pressure_solver = pressure_solver
        self._W2 = W2
        self._W3 = W3
        self._Wb = Wb
        self._omega_N = 0.5*dt*N
        self._dt_half = Constant(0.5*dt)
        self._dt_half_N2 = Constant(0.5*dt*N**2)
        self._dt_half_c2 = Constant(0.5*dt*c**2)
        self._diagonal_only = diagonal_only
        self._mesh = self._W3._mesh
        self._zhat = VerticalNormal(self._mesh)
        self._dx = self._mesh._dx
        self._matrixfree_prec = matrixfree_prec
        if (not self._matrixfree_prec):
            self._Wmixed = self._W2 * self._W3
            self._mutest, self._mptest = TestFunctions(self._Wmixed)
            self._mutrial, self._mptrial = TrialFunctions(self._Wmixed)
            self._bcs = [DirichletBC(self._Wmixed.sub(0), 0.0, "bottom"),
                         DirichletBC(self._Wmixed.sub(0), 0.0, "top")]
            self._sparams={'pc_type': 'fieldsplit',
                           'pc_fieldsplit_type': 'schur',
                           'ksp_type': 'gmres',
                           'ksp_max_it': 30,
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
                           'ksp_monitor': True}

            self._a = (  self._mptest*self._mptrial \
                       + self._dt_half_c2*self._mptest*div(self._mutrial) \
                       - self._dt_half*div(self._mutest)*self._mptrial \
                       + (dot(self._mutest,self._mutrial) + self._omega_N**2 \
                            * dot(self._mutest,self._zhat.zhat) \
                            * dot(self._mutrial,self._zhat.zhat)) \
                      )*self._dx
            self._vmixed = Function(self._Wmixed)
        else:
            self._bcs = [DirichletBC(self._W2, 0.0, "bottom"),
                         DirichletBC(self._W2, 0.0, "top")]
        self._utest = TestFunction(self._W2)
        self._ptest = TestFunction(self._W3)
        self._btest = TestFunction(self._Wb)
        # Buoyancy mass matrix
        self._Mb = assemble(self._btest*TrialFunction(self._Wb)*self._dx)
        self._solver_param_b = {'ksp_type':'cg',
                                'ksp_rtol':tolerance_b,
                                'ksp_max_it':maxiter_b,
                                'ksp_monitor':False,
                                'pc_type':'jacobi'}
        self._mutilde = Mutilde(self._W2,self._Wb,self._omega_N,
                                tolerance_b=tolerance_b,maxiter_b=maxiter_b,
                                tolerance_u=tolerance_u,maxiter_u=maxiter_u)
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
        self._mixedarray = MixedArray(self._W2,self._W3,self._Wb)
        
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
            self._mutilde.divide(r_u,u)
            # Buoyancy solve
            solve(self._Mb,b,r_b,solver_parameters=self._solver_param_b)
        else:
            # Modified RHS for velocity 
            solve(self._Mb,self._tmp_b,r_b,solver_parameters=self._solver_param_b)
            assemble(self._dt_half * dot(self._utest,self._zhat.zhat) \
                                   * self._tmp_b * self._dx,
                     tensor=self._rtilde_u)
            self._rtilde_u += r_u
            if (self._matrixfree_prec):
                # Modified RHS for pressure
                self._mutilde.divide(self._rtilde_u,self._tmp_u)
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
                self._mutilde.divide(self._tmp_u,u)
            else:
                m_p = assemble(TestFunction(self._W3)*TrialFunction(self._W3)*self._dx)
                m_u = assemble(dot(TestFunction(self._W2),TrialFunction(self._W2))*self._dx)
                r_u.assign(self._rtilde_u)
                solve(m_p, self._rtilde_p, r_p)
                solve(m_u, self._rtilde_u, r_u)
                L = (  self._mptest*self._rtilde_p \
                     + dot(self._mutest,self._rtilde_u))*self._dx
                solve(self._a == L,self._vmixed,
                      solver_parameters=self._sparams,
                      bcs=self._bcs)
                u.assign(self._vmixed.sub(0))
                p.assign(self._vmixed.sub(1))
            # Backsubstitution for buoyancy
            assemble(- self._dt_half_N2 * self._btest*dot(self._zhat.zhat,u)*self._dx,
                     tensor=self._tmp_b)
            self._tmp_b += r_b
            solve(self._Mb,b,self._tmp_b,solver_parameters=self._solver_param_b)

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
            self._mixedarray.combine(u,p,b,y)
        
