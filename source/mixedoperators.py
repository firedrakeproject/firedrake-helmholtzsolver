from firedrake import *

from pressuresolver.vertical_normal import *
from mixedarray import *
from pyop2.profiling import timed_function

class MixedOperator(object):
    '''Matrix free operator for pressure-velocity subsystem of the mixed Gravity wave system

        This class can be used to apply the operator :math:`A` defined by

        :math:`(\\vec{R}_u, \\vec{R}_p)^T = A (\\vec{U},\\vec{P})^T`
        with
    
        .. math::

            \\vec{R}_u = \\tilde{M}_u \\vec{U} - \\frac{\Delta t}{2} D^T \\vec{P} 

            \\vec{R}_p = \\frac{\Delta t}{2} c^2 D \\vec{U} + M_p \\vec{P}

        :arg W2: HDiv function space for velocity
        :arg W3: L2 function space for pressure
        :arg dt: Positive real number, time step size
        :arg c: Positive real number, speed of sound waves
        :arg N: Positive real number, buoyancy frequency
    '''
    def __init__(self,W2,W3,
                      dt,c,N):
        self._W2 = W2
        self._W3 = W3
        self._dt_half = Constant(0.5*dt)
        self._dt_half_c2 = Constant(0.5*dt*c**2)
        self._omega_N2 = Constant((0.5*dt*N)**2)
        self._utest = TestFunction(self._W2)
        self._ptest = TestFunction(self._W3)
        self._utrial = TrialFunction(self._W2)
        self._ptrial = TrialFunction(self._W3)
        self._mixedarray = MixedArray(self._W2,self._W3)
        self._u_tmp = Function(self._W2)
        self._p_tmp = Function(self._W3)
        self._r_u_tmp = Function(self._W2)
        self._r_p_tmp = Function(self._W3)
        self._mesh = self._W3._mesh
        self._zhat = VerticalNormal(self._mesh)
        self._dx = self._mesh._dx
        self._bcs = [DirichletBC(self._W2, 0.0, "bottom"),
                     DirichletBC(self._W2, 0.0, "top")]
        self._mat_uu = assemble( (  dot(self._utest,self._utrial) \
                   + self._omega_N2 \
                       * dot(self._utest,self._zhat.zhat) \
                       * dot(self._zhat.zhat,self._utrial) ) * self._dx).M.handle
        self._mat_up = assemble(-self._dt_half*div(self._utest)*self._ptrial*self._dx).M.handle
        self._mat_pp = assemble( self._ptest * self._ptrial * self._dx).M.handle
        self._mat_pu = assemble(self._ptest*self._dt_half_c2*div(self._utrial)*self._dx).M.handle

    @timed_function("mixed_operator") 
    def apply(self,u,p,r_u,r_p):
        '''Apply the operator to a mixed field.
        
            Calculate
            :math:`(\\vec{R}_u, \\vec{R}_p)^T = A (\\vec{U},\\vec{P})^T`

            :arg u: Velocity field :math:`u`
            :arg p: Pressure field :math:`p`
            :arg r_u: Resulting velocity field :math:`u`
            :arg r_p: Resulting pressure field :math:`p`
        '''
        # Apply BCs to u
        self._apply_bcs(u)
        with r_u.dat.vec as v_u, r_p.dat.vec as v_p:
            with u.dat.vec_ro as x_u, p.dat.vec_ro as x_p:
                self._mat_uu.mult(x_u,v_u)
                self._mat_up.multAdd(x_p,v_u,v_u)
                self._mat_pu.mult(x_u,v_p)
                self._mat_pp.multAdd(x_p,v_p,v_p)

        # Apply BCs to R_u
        self._apply_bcs(r_u)

    def _apply_bcs(self,u):
        '''Apply boundary conditions to velocity field.

            :arg u: Field to apply to
        '''
        for bc in self._bcs:
            bc.apply(u)

    def mult(self,mat,x,y):
        '''PETSc interface for operator application

            PETSc interface wrapper for the :func:`apply` method.

            :arg x: PETSc vector representing the field to be multiplied.
            :arg y: PETSc vector representing the result.
        '''
        with self._u_tmp.dat.vec as u, \
             self._p_tmp.dat.vec as p:
            self._mixedarray.split(x,u,p)
        self.apply(self._u_tmp,self._p_tmp,self._r_u_tmp,self._r_p_tmp)

        with self._r_u_tmp.dat.vec_ro as u, \
             self._r_p_tmp.dat.vec_ro as p:
            self._mixedarray.combine(y,u,p)

class MixedOperatorOrography(object):
    '''Matrix free operator for mixed Gravity wave system with orography

        This class can be used to apply the operator :math:`A` defined by

        :math:`(\\vec{R}_u, \\vec{R}_p, \\vec{R}_b)^T = A (\\vec{U},\\vec{P},\\vec{B})^T`
        with
    
        .. math::

            \\vec{R}_u = M_u \\vec{U} - \\frac{\Delta t}{2} D^T \\vec{P} 
                         - \\frac{\Delta t}{2} Q \\vec{B} 

            \\vec{R}_p = \\frac{\Delta t}{2} c^2 D \\vec{U} + M_p \\vec{P}

            \\vec{R}_b = \\frac{\Delta t}{2} N^2 Q^T \\vec{U} + M_b\\vec{B}

        :arg W2: HDiv function space for velocity
        :arg W3: L2 function space for pressure
        :arg Wb: Function space for buoyancy
        :arg dt: Positive real number, time step size
        :arg c: Positive real number, speed of sound waves
        :arg N: Positive real number, buoyancy frequency
    '''
    def __init__(self,W2,W3,Wb,
                      dt,c,N):
        self._W2 = W2
        self._W3 = W3
        self._Wb = Wb
        self._dt_half = Constant(0.5*dt)
        self._dt_half_c2 = Constant(0.5*dt*c**2)
        self._dt_half_N2 = Constant(0.5*dt*N**2)
        self._utest = TestFunction(self._W2)
        self._ptest = TestFunction(self._W3)
        self._btest = TestFunction(self._Wb)
        self._mixedarray = MixedArray(self._W2,self._W3,self._Wb)
        self._u_tmp = Function(self._W2)
        self._p_tmp = Function(self._W3)
        self._b_tmp = Function(self._Wb)
        self._r_u_tmp = Function(self._W2)
        self._r_p_tmp = Function(self._W3)
        self._r_b_tmp = Function(self._Wb)
        self._mesh = self._W3._mesh
        self._zhat = VerticalNormal(self._mesh)
        self._dx = self._mesh._dx
        self._bcs = [DirichletBC(self._W2, 0.0, "bottom"),
                     DirichletBC(self._W2, 0.0, "top")]

    @timed_function("mixed_operator") 
    def apply(self,u,p,b,r_u,r_p,r_b):
        '''Apply the operator to a mixed field.
        
            Calculate
            :math:`(\\vec{R}_u, \\vec{R}_p, \\vec{R}_b)^T = A (\\vec{U},\\vec{P},\\vec{B})^T`

            :arg u: Velocity field :math:`u`
            :arg p: Pressure field :math:`p`
            :arg b: Buoyancy field :math:`b`
            :arg r_u: Resulting velocity field :math:`u`
            :arg r_p: Resulting pressure field :math:`p`
            :arg r_b: Resulting buoyancy field :math:`b`
        '''
        # Apply BCs to u
        self._apply_bcs(u)
        assemble( (  dot(self._utest,u) 
                   - self._dt_half*div(self._utest)*p
                   - self._dt_half*dot(self._utest,self._zhat.zhat)*b
                  ) * self._dx,
                 tensor=r_u)
        assemble( self._ptest * (p + self._dt_half_c2*div(u)) * self._dx,
                 tensor=r_p)
        assemble( self._btest * (b  + self._dt_half_N2*dot(self._zhat.zhat,u)) * self._dx,
                 tensor =r_b)
        # Apply BCs to R_u
        self._apply_bcs(r_u)

    def _apply_bcs(self,u):
        '''Apply boundary conditions to velocity field.

            :arg u: Field to apply to
        '''
        for bc in self._bcs:
            bc.apply(u)

    def mult(self,mat,x,y):
        '''PETSc interface for operator application

            PETSc interface wrapper for the :func:`apply` method.

            :arg x: PETSc vector representing the field to be multiplied.
            :arg y: PETSc vector representing the result.
        '''
        with self._u_tmp.dat.vec as u, \
             self._p_tmp.dat.vec as p, \
             self._b_tmp.dat.vec as b:
            self._mixedarray.split(x,u,p,b)
        self.apply(self._u_tmp,self._p_tmp,self._b_tmp,
                   self._r_u_tmp,self._r_p_tmp,self._r_b_tmp)

        with self._r_u_tmp.dat.vec_ro as u, \
             self._r_p_tmp.dat.vec_ro as p, \
             self._r_b_tmp.dat.vec_ro as b:
            self._mixedarray.combine(y,u,p,b)

