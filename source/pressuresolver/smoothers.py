import numpy as np
from firedrake import *
import xml.etree.cElementTree as ET
from firedrake.petsc import PETSc

class Jacobi(object):
    '''Jacobi smoother.

    Base class for matrix-free smoother for the linear Schur complement system.

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
        self._operator = operator
        self._W3 = self._operator._W3
        self._mesh = self._W3.mesh()
        self._mu_relax = mu_relax
        self._n_smooth = n_smooth
        self._dx = self._mesh._dx
        self._r_tmp = Function(self._W3)
        self._b_tmp = Function(self._W3)
        self._phi_tmp = Function(self._W3)
        self._level = level
        with self._b_tmp.dat.vec as v:
            ndof = self._W3.dof_dset.size
            self._iset = PETSc.IS().createStride(ndof,
                                                 first=v.owner_range[0],
                                                 step=1,
                                                 comm=v.comm)

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
