from operators import *

class Jacobi(object):
    '''Jacobi smoother.

    Matrix-free smoother for the linear Schur complement system.
    The diagonal matrix :math:`D` used in the :class:`smooth()` method is constructed as 
    described in `Notes in LaTeX <./FEMmultigrid.pdf>`_:
    
    .. math::
        
        D_{ii} = (M_\phi)_{ii} + 2 \sum_{e'\in e(i)} \\frac{1}{(M_u^*)_{e'e'}}

    (where :math:`e(i)` are all facets adjacent to cell :math:`i`.)

    :arg operator: Schur complement operator, of type :class:`Operator`.
    :arg mu_relax: Under-/Over-relaxation parameter :math:`mu`
    :arg n_smooth: Number of smoothing steps to apply in method :class:`smooth()`.
    '''
    def __init__(self,operator,
                 mu_relax=2./3.,
                 n_smooth=1):
        self.operator = operator
        self.V_pressure = self.operator.V_pressure
        self.mu_relax = mu_relax
        self.n_smooth = n_smooth
        self.dx = self.operator.V_pressure.mesh()._dx
        # Construct lumped mass matrix
        self.lumped_mass = self.operator.lumped_mass
        self._build_D_diag()

    def _build_D_diag(self):
        '''Construct diagonal matrix for smoothing step.
        
        Calculate the diagonal matrix :math:`D`.
        '''
        one_pressure = Function(self.V_pressure)
        one_pressure.assign(1.0)
        D_diag = assemble(TestFunction(self.V_pressure)*one_pressure*self.dx)
        kernel_add_vterm = 'for(int i=0; i<M_u_lumped.dofs; i++) {D_diag[0][0] += 2./M_u_lumped[i][0];}'
        M_u_lumped = self.lumped_mass.get()
        par_loop(kernel_add_vterm,self.dx,{'D_diag':(D_diag,INC),'M_u_lumped':(M_u_lumped,READ)})
        kernel_inv = '{ D_diag_inv[0][0] = 1./D_diag[0][0]; }'
        self.D_diag_inv = Function(self.V_pressure)
        par_loop(kernel_inv,self.dx,{'D_diag_inv':(self.D_diag_inv,WRITE),
                                'D_diag':(D_diag,READ)})
       
    def solve(self,b,phi):
        '''Solve approximately with RHS :math:`b`.
        
        Repeatedy apply the smoother to solve the equation :math:`H\phi=b` approximately.
        :arg b: Right hand side :math:`b` in pressure space
        :arg phi: State vector :math:`\phi` in pressure space (out)
        '''
        phi.assign(0.0)
        self.smooth(b,phi)

    def smooth(self,b,phi,initial_phi_is_zero=False):
        '''Smooth.
        
        Apply the smoother 
        
        .. math::

            \phi \mapsto \phi + 2\mu D^{-1} (b-H\phi)
            
        repeatedly to the state vector :math:`\phi`. If :class:`initial_phi_is_zero` is
        True, then the initial :math:`\phi` is assumed to be zero and in the first iteration
        the updated :math:`\phi` is just given by :math:`D^{-1}b`.

        :arg b: Right hand side :math:`b` in pressure space
        :arg phi: State vector :math:`\phi` in pressure space (out)
        :arg initial_phi_is_zero: Initialise with :math:`\phi=0`.
        '''
        r = Function(self.V_pressure)
        for i in range(self.n_smooth):
            if ( (i==0) and (initial_phi_is_zero)):
                r.assign(b)
            else:
                r.assign(self.operator.residual(b,phi))
            # Apply inverse diagonal r_i -> D^{-1}_ii *r_i
            kernel_inv_diag = '{ (*r) *= (*D_diag_inv); }'
            par_loop(kernel_inv_diag,direct,{'r':(r,RW),'D_diag_inv':(self.D_diag_inv,READ)})
            # Update phi 
            phi += 2.*self.mu_relax*r

class SmootherHierarchy(object):
    '''Hierarchy of smoothers.
    
    Set of smoothers on different levels of the function space
    hierarchy, as needed by the multigrid solver.

    :arg Type: the type (class) of the smoother
    :arg operator_hierarchy: An :class:`.OperatorHierarchy` of linear Schur 
        complement operators in pressure space
    :arg mu_relax: Under-/Over-relaxation parameter :math:`mu`
    :arg n_smooth: Number of smoothing steps to apply in method :class:`smooth()`.
    '''
    def __init__(self,Type,
                 operator_hierarchy,
                 mu_relax=2./3.,
                 n_smooth=1):
        self.operator_hierarchy = operator_hierarchy
        self.mu_relax = mu_relax
        self.n_smooth = n_smooth
        self._hierarchy = [Type(operator,
                                self.mu_relax,
                                self.n_smooth)
                           for operator in self.operator_hierarchy]

    def __getitem__(self,level):
        '''Return smoother on a particular level.
            
        :arg level: Multigrid level
        '''
        return self._hierarchy[level]

    def __len__(self):
        '''Return number of multigrid levels.'''
        return len(self._hierarchy)

