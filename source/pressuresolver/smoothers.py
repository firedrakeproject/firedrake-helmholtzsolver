import numpy as np
from operators import *
from firedrake.ffc_interface import compile_form

class Jacobi(object):
    '''Jacobi smoother.

    Base class for matrix-free smoother for the linear Schur complement system.
    :arg operator: Schur complement operator, of type :class:`Operator`.
    :arg mu_relax: Under-/Over-relaxation parameter :math:`mu`
    :arg n_smooth: Number of smoothing steps to apply in method :class:`smooth()`.
    '''
    def __init__(self,operator,
                 mu_relax=2./3.,
                 n_smooth=1,
                 *args):
        self.operator = operator
        self.V_pressure = self.operator.V_pressure
        self.V_velocity = self.operator.V_velocity
        self.mesh = self.V_pressure.mesh()
        self.mu_relax = mu_relax
        self.n_smooth = n_smooth
        self.dx = self.mesh._dx
        # Construct lumped mass matrix
        self.lumped_mass = self.operator.lumped_mass
       
    def solve(self,b,phi):
        '''Solve approximately with RHS :math:`b`.
        
        Repeatedy apply the smoother to solve the equation :math:`H\phi=b`
        approximately.

        :arg b: Right hand side :math:`b` in pressure space
        :arg phi: State vector :math:`\phi` in pressure space (out)
        '''
        self.smooth(b,phi,initial_phi_is_zero=True)

    def smooth(self,b,phi,initial_phi_is_zero=False):
        '''Smooth.
        
        Apply the smoother 
        
        .. math::

            \phi \mapsto \phi + 2\mu D^{-1} (b-H\phi)
            
        repeatedly to the state vector :math:`\phi`.
        If :class:`initial_phi_is_zero` is True, then the initial :math:`\phi`
        is assumed to be zero and in the first iteration the updated
        :math:`\phi` is just given by :math:`D^{-1}b`.

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
            self.divide_by_Ddiag(r)
            # Update phi
            if ( (i ==0) and (initial_phi_is_zero) ):
                phi.assign(r)
                r *= 2.*self.mu_relax
            else:
                phi.assign(phi+2.*self.mu_relax*r)

class Jacobi_LowestOrder(Jacobi):
    '''Lowest order Jacobi smoother.

    Matrix-free smoother for the linear Schur complement system.
    The diagonal matrix :math:`D` used in the :class:`smooth()` method is
    constructed as described in `Notes in LaTeX <./FEMmultigrid.pdf>`_:
    
    .. math::
        
        D_{ii} = (M_\phi)_{ii} + 2 \sum_{e'\in e(i)} \\frac{1}{(M_u^*)_{e'e'}}

    (where :math:`e(i)` are all facets adjacent to cell :math:`i`.)

    :arg operator: Schur complement operator, of type :class:`Operator`.
    :arg mu_relax: Under-/Over-relaxation parameter :math:`mu`
    :arg n_smooth: Number of smoothing steps to apply in method :class:`smooth()`.
    :arg use_maximal_eigenvalue: If this is true, then :math:`D` with be set 
        to :math:`\max_i\{D_{ii}\} Id`, i.e. the unit matrix times the maximal
        eigenvalue. This means that the smoother is symmetric, which is not
        necessarily the case otherwise.
    '''
    def __init__(self,operator,
                 mu_relax=2./3.,
                 n_smooth=1,
                 use_maximal_eigenvalue=False):
        super(Jacobi_LowestOrder,self).__init__(operator,mu_relax,n_smooth)
        self.use_maximal_eigenvalue=use_maximal_eigenvalue
        self._build_D_diag()

    def _build_D_diag(self):
        '''Construct diagonal matrix for smoothing step.
        
        Calculate the diagonal matrix :math:`D`.
        '''
        one_pressure = Function(self.V_pressure)
        one_pressure.assign(1.0)
        D_diag = assemble(TestFunction(self.V_pressure)*one_pressure*self.dx)
        kernel_add_vterm = 'for(int i=0; i<M_u_lumped.dofs; i++) {D_diag[0][0] += 2.*omega2[0]/M_u_lumped[i][0];}'
        M_u_lumped = self.lumped_mass.get()
        omega2 = Constant(self.operator.omega**2)
        par_loop(kernel_add_vterm,self.dx,
                 {'D_diag':(D_diag,INC),
                  'M_u_lumped':(M_u_lumped,READ),
                  'omega2':(omega2,READ)})
        if (self.use_maximal_eigenvalue):
            max_D_diag = np.max(D_diag.dat.data)
            D_diag.dat.data[:] = max_D_diag
        kernel_inv = '{ (*D_diag_inv) = 1./(*D_diag); }'
        self.D_diag_inv = Function(self.V_pressure)
        par_loop(kernel_inv,direct,{'D_diag_inv':(self.D_diag_inv,WRITE),
                                    'D_diag':(D_diag,READ)})

    def divide_by_Ddiag(self,r):
        kernel_inv_diag = '{ (*r) *= (*D_diag_inv); }'
        par_loop(kernel_inv_diag,direct,{'r':(r,RW),
                                         'D_diag_inv':(self.D_diag_inv,READ)})

class Jacobi_HigherOrder(Jacobi):
    '''Higher order Jacobi smoother.

    Matrix-free smoother for the linear Schur complement system.
    The block-diagonal matrix :math:`D` used in the :class:`smooth()` method
    is constructed as described in `Notes in LaTeX <./FEMmultigrid.pdf>`_:
    
    .. math::
        
        D_{ii} = (M_\phi)_{ii} + 2 \sum_{e'\in e(i)} B_{ie'}{(M_u^*)^{-1}_{e'e'}}B^T_{e'i}

    (where :math:`e(i)` are all facets adjacent to cell :math:`i`.)

    :arg operator: Schur complement operator, of type :class:`Operator`.
    :arg mu_relax: Under-/Over-relaxation parameter :math:`mu`
    :arg n_smooth: Number of smoothing steps to apply in method
        :class:`smooth()`.
    '''
    def __init__(self,operator,
                 mu_relax=2./3.,
                 n_smooth=1):
        super(Jacobi_HigherOrder,self).__init__(operator,mu_relax,n_smooth)
        # Only works if diagonal lumped mass matrix is used
        if (not self.lumped_mass.diagonal_matrix):
            raise NotImplementedError('Higher order Jacobi smoother only implemented for diagonal lumped BDFM1 mass matrices')
        self._build_D_diag()

    def _build_D_diag(self):
        '''Construct diagonal matrix for smoothing step.
        
        Calculate the diagonal matrix :math:`D`.
        '''
        V_DG0 = FunctionSpace(self.mesh,'DG',0)
        w = TrialFunction(self.V_velocity)
        phi = TestFunction(self.V_pressure)
        psi = TrialFunction(self.V_pressure)

        # Construct diagonal matrix for smoother
        # * Step 1 *
        # Use UFL to build mass matrix and store in a 
        # 3x3 matrix on each cell
        mass = psi*phi*dx
        mass_kernel = compile_form(mass, 'mass')[0][6]
        self.D_diag_inv = Function(V_DG0, val=op2.Dat(V_DG0.node_set**(3*3)))
        op2.par_loop(mass_kernel,self.D_diag_inv.cell_set,
                     self.D_diag_inv.dat(op2.INC,
                                  self.D_diag_inv.cell_node_map()[op2.i[0]]),
                     self.mesh.coordinates.dat(op2.READ,
                       self.mesh.coordinates.cell_node_map(),
                       flatten=True),
                     self.mesh.coordinates.dat(op2.READ,
                       self.mesh.coordinates.cell_node_map(),
                       flatten=True)
                    )

        # * Step 2 *
        # Use UFL to build divergence matrix and store it as a
        # 3x9 matrix on each cell.
        bdiv = div(w)*phi*dx
        bdiv_kernel = compile_form(bdiv, 'bdiv')[0][6]
        bdiv_dat = Function(V_DG0, val=op2.Dat(V_DG0.node_set**(3*9)))
        op2.par_loop(bdiv_kernel,bdiv_dat.cell_set,
                     bdiv_dat.dat(op2.INC,bdiv_dat.cell_node_map()[op2.i[0]]),
                     self.mesh.coordinates.dat(op2.READ,
                       self.mesh.coordinates.cell_node_map(),
                       flatten=True),
                     self.mesh.cell_orientations().dat(op2.READ,
                       self.mesh.cell_orientations().cell_node_map(),
                       flatten=True),
                     self.mesh.coordinates.dat(op2.READ,
                       self.mesh.coordinates.cell_node_map(),
                       flatten=True)
                    )

        # * Step 3 *
        # Loop over edges and on each edge add the 3x3 matrix
        # omega^2 B_{ie} (M_u^*)^{-1} B^T_{ei} to the cells i which 
        # are adjacent to this edge
        kernel_code = '''void build_diagonal(double **bdiv,
                                             unsigned int *localidx,
                                             double **Mulumpedinv,
                                             double **ddiag) {
            /* ************************************************ *
             *   P A R A M E T E R S
             * bdiv:        3x9 divergence matrix on each cell
             * localidx:    local index of edge e on each 
             *              cell to which we redirect
             * Mulumpedinv: inverse of lumped mass matrix
             * ddiag:       3x3 matrix in each cell
             * ************************************************ */
            // Loop over indirection map, i.e. the two cells 
            // adjacent to the edge
            for (int icell=0;icell<2;++icell) {
              // Loop over row- and column- indices j,k to
              // calculate the entries Ddiag_{jk} of the matrix Ddiag
              for (int j=0;j<3;++j) {
                for (int k=0;k<3;++k) {
                  // Work out the three indices for which B_{ie} does
                  // not vanish on this cell
                  int idx[3];
                  idx[0] = 2*localidx[icell];
                  idx[1] = 2*localidx[icell]+1;
                  idx[2] = 6+localidx[icell];
                  for (int ell=0;ell<3;++ell) {
                    int ell_lumped = ell;
                    if (ell > 2) {
                      ell_lumped += icell;
                    }
                    // Work out index to use in lumped mass matrix
                    ddiag[icell][3*j+k] += 2.*omega2 
                                         * bdiv[icell][9*j+idx[ell]]
                                         * Mulumpedinv[0][ell_lumped]
                                         * bdiv[icell][9*k+idx[ell]];
                }
              }
            }
          }
        }'''

        kernel = op2.Kernel(kernel_code,'build_diagonal')
        dof_map = V_DG0.interior_facet_node_map()
        omega2 = op2.Const(1, self.operator.omega**2,
                           name="omega2", dtype=float)
        op2.par_loop(kernel,self.mesh.interior_facets.set,
                     bdiv_dat.dat(op2.READ,dof_map),
                     self.mesh.interior_facets.local_facet_dat(op2.READ),
                     self.lumped_mass.data_inv.dat(op2.READ,
                       self.lumped_mass.facet2dof_map_facets),
                     self.D_diag_inv.dat(op2.INC,dof_map)
                    )
        # * Step 4 *
        # invert local 3x3 matrix
        kernel_inv3x3mat = '''{
            double b[9];
            for (int i=0;i<9;++i) b[i] = a[i];
            a[3*0+0] = +(b[3*1+1]*b[3*2+2]-b[3*1+2]*b[3*2+1]);
            a[3*0+1] = -(b[3*0+1]*b[3*2+2]-b[3*0+2]*b[3*2+1]);
            a[3*0+2] = +(b[3*0+1]*b[3*1+2]-b[3*0+2]*b[3*1+1]);
            a[3*1+0] = -(b[3*1+0]*b[3*2+2]-b[3*1+2]*b[3*2+0]);
            a[3*1+1] = +(b[3*0+0]*b[3*2+2]-b[3*0+2]*b[3*2+0]);
            a[3*1+2] = -(b[3*0+0]*b[3*1+2]-b[3*0+2]*b[3*1+0]);
            a[3*2+0] = +(b[3*1+0]*b[3*2+1]-b[3*1+1]*b[3*2+0]);
            a[3*2+1] = -(b[3*0+0]*b[3*2+1]-b[3*0+1]*b[3*2+0]);
            a[3*2+2] = +(b[3*0+0]*b[3*1+1]-b[3*0+1]*b[3*1+0]);
            double invDet = 1./( 
                +b[3*0+0]*(+(b[3*1+1]*b[3*2+2]-b[3*1+2]*b[3*2+1]))
                +b[3*0+1]*(-(b[3*1+0]*b[3*2+2]-b[3*1+2]*b[3*2+0]))
                +b[3*0+2]*(+(b[3*1+0]*b[3*2+1]-b[3*1+1]*b[3*2+0])));
            for (int i=0;i<9;++i) a[i] *= invDet;
        }'''
        par_loop(kernel_inv3x3mat,
                 direct,{'a':(self.D_diag_inv,RW)})

    def divide_by_Ddiag(self,r):
        kernel_inv_diag = '''{ 
        double r_old[3];
        for (int i=0;i<3;++i) { r_old[i] = r[0][i]; r[0][i] = 0; }
          for (int i=0;i<3;++i) {
            for (int j=0;j<3;++j) {
              r[0][i] += D_diag_inv[0][3*i+j]*r_old[j]; 
            }
          }
        }'''
        par_loop(kernel_inv_diag,self.dx,
                 {'r':(r,RW),
                  'D_diag_inv':(self.D_diag_inv,READ)})

class SmootherHierarchy(object):
    '''Hierarchy of smoothers.
    
    Set of smoothers on different levels of the function space
    hierarchy, as needed by the multigrid solver.

    :arg Type: the type (class) of the smoother
    :arg operator_hierarchy: An :class:`.OperatorHierarchy` of linear Schur 
        complement operators in pressure space
    :arg mu_relax: Under-/Over-relaxation parameter :math:`mu`
    :arg n_smooth: Number of smoothing steps to apply in method :class:`smooth()`.
    :arg use_maximal_eigenvalue: If this is true, then :math:`D` with be set to :math:`\max_i\{D_{ii}\} Id`, i.e. the unit matrix times the maximal eigenvalue. This means that the smoother is symmetric, which is not necessarily the case otherwise. Only supported at lowest order, i.e. DG0 + RT0 elements. 
    '''
    def __init__(self,Type,
                 operator_hierarchy,
                 mu_relax=2./3.,
                 n_smooth=1,
                 use_maximal_eigenvalue=False):
        self.operator_hierarchy = operator_hierarchy
        self.mu_relax = mu_relax
        self.n_smooth = n_smooth
        self.use_maximal_eigenvalue=use_maximal_eigenvalue
        self._hierarchy = [Type(operator,
                                self.mu_relax,
                                self.n_smooth,
                                self.use_maximal_eigenvalue)
                           for operator in self.operator_hierarchy]

    def __getitem__(self,level):
        '''Return smoother on a particular level.
            
        :arg level: Multigrid level
        '''
        return self._hierarchy[level]

    def __len__(self):
        '''Return number of multigrid levels.'''
        return len(self._hierarchy)

