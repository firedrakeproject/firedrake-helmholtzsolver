import fractions
import math
from firedrake import *
from firedrake.ffc_interface import compile_form
from cellindirection import *

class BandedMatrix(object):
    def __init__(self,fs_row,fs_col,gamma_m=None,gamma_p=None):
        '''Generalised block banded matrix.

        :math:`n_{row}\times n_{col}` matrix over the field of dense
        :math:`n^{(h)}_{row} \times n^{(h)}_{col}` 
        matrices with entries only for row-indices :math:`k` and column 
        index :math:`\ell` for which

            :math::
                -\gamma_- \le \alpha k - \beta \ell \le \gamma_+
        
            :arg fs_row: Row function space
            :arg fs_col: Column function space
            :arg gamma_m: Lower bound :math:`\gamma_-`
            :arg gamma_p: Upper bound :math:`\gamma_+`
        '''
        # Function spaces
        self._fs_row = fs_row
        self._fs_col = fs_col
        self._mesh = fs_row.mesh()
        self._ncelllayers = self._mesh.layers-1
        self._hostmesh = self._mesh._old_mesh
        self._ind_row = CellIndirection(self._fs_row)
        self._ind_col = CellIndirection(self._fs_col)
        self._n_row = ( self._ind_row.ndof_cell*self._ncelllayers \
                      + self._ind_row.ndof_bottom_facet*(self._ncelllayers+1) ) \
                      / self._ind_row.horiz_extent
        self._n_col = ( self._ind_col.ndof_cell*self._ncelllayers \
                      + self._ind_col.ndof_bottom_facet*(self._ncelllayers+1) ) \
                      / self._ind_col.horiz_extent
        
        self._gamma_m = (self._ind_row.ndof / self._ind_row.horiz_extent - 1) \
                      * (self._ind_col.ndof_cell+self._ind_col.ndof_bottom_facet) \
                      / self._ind_col.horiz_extent
        self._gamma_p = (self._ind_col.ndof / self._ind_col.horiz_extent - 1) \
                      * (self._ind_row.ndof_cell+self._ind_row.ndof_bottom_facet) \
                      / self._ind_row.horiz_extent
        if (gamma_m):
            self._gamma_m = max(self._gamma_m,gamma_m)
        if (gamma_p):
            self._gamma_p = max(self._gamma_p,gamma_p)
        self._alpha = self._ind_col.ndof / self._ind_col.horiz_extent
        self._beta  = self._ind_row.ndof / self._ind_row.horiz_extent
        print self._alpha, self._beta, self._gamma_m, self._gamma_p
        self._divide_by_gcd()
        self._Vcell = FunctionSpace(self._hostmesh,'DG',0)
        self._data = op2.Dat(self._Vcell.node_set**(self.bandwidth * self._n_row *
                                                   self._ind_row.horiz_extent * 
                                                   self._ind_col.horiz_extent))
        self._lu_decomposed = False
        self._param_dict = {'n_row':self._n_row,
                            'n_col':self._n_col,
                            'gamma_m':self._gamma_m,
                            'gamma_p':self._gamma_p,
                            'alpha':self._alpha,
                            'beta':self._beta,
                            'bandwidth':self.bandwidth,
                            'ncelllayers':self._ncelllayers,
                            'indirectiontable_row':self._ind_row.maptable(),
                            'indirectiontable_col':self._ind_col.maptable(),
                            'horiz_extent_col':self._ind_col.horiz_extent,
                            'horiz_extent_row':self._ind_row.horiz_extent}

    @property
    def fs_row(self):
        '''Row function space.'''
        return self._fs_row

    @property
    def fs_col(self):
        '''Column function space.'''
        return self._fs_col

    @property
    def is_square(self):
        '''Is this a square matrix?'''
        return (self._n_row == self._n_col)

    @property
    def is_sparsity_symmetric(self):
        '''Is the sparsity pattern symmetric (has to be square matrix)?'''
        return ( (self.is_square) and \
                 (self._alpha == self._beta) and \
                 (self._gamma_m == self._gamma_p) )

    @property
    def bandwidth(self):
        '''Matrix bandwidth.'''
        return 1+int(math.ceil((self._gamma_m+self._gamma_p)/float(self._beta)))

    @property
    def alpha(self):
        '''Bandedness parameter :math:`alpha`'''
        return self._alpha
    
    @property
    def beta(self):
        '''Bandedness parameter :math:`beta`'''
        return self._beta
    
    @property
    def gamma_m(self):
        '''Bandedness parameter :math:`gamma_-`'''
        return self._gamma_m
    
    @property
    def gamma_p(self):
        '''Bandedness parameter :math:`gamma_+`'''
        return self._gamma_p
    
    def _divide_by_gcd(self):
        '''If alpha and beta have a gcd > 1, divide by this.
        '''
        gcd = fractions.gcd(self._alpha,self._beta)
        if (gcd > 1):
            self._alpha /= gcd
            self._beta /= gcd
            self._gamma_m /= gcd
            self._gamma_p /=gcd

    def assemble_ufl_form(self,ufl_form):
        '''Assemble the matrix form a UFL form.

            :arg ufl_form: UFL form to assemble
        '''
        compiled_form = compile_form(ufl_form, 'ufl_form')[0]
        kernel = compiled_form[6]
        coords = compiled_form[3]
        coefficients = compiled_form[4]
        arguments = ufl_form.arguments()
        assert len(arguments) == 2, 'Not a bilinear form'
        nrow = arguments[0].cell_node_map().arity
        ncol = arguments[1].cell_node_map().arity
        V_lma = FunctionSpace(self._mesh,'DG',0)
        lma = Function(V_lma, val=op2.Dat(V_lma.node_set**(nrow*ncol)))
        args = [lma.dat(op2.INC, lma.cell_node_map()[op2.i[0]]), 
                coords.dat(op2.READ, coords.cell_node_map(), flatten=True)]
        for c in coefficients:
            args.append(c.dat(op2.READ, c.cell_node_map(), flatten=True))
        op2.par_loop(kernel,lma.cell_set, *args)
        self._assemble_lma(lma)
        
    def _assemble_lma(self,lma):
        param_dict = {'SELF_'+x:y for (x,y) in self._param_dict.iteritems()}
        ind_dict = {'IND_map_row':,
                    'IND_map_col':self._ind_col.ki_to_local_index('ell_local','j'),
                    'IND_nrow':self._ind_row.ndof,
                    'IND_ncol':self._ind_col.ndof,
                    'IND_ndof_cell_row':self._ind_row.ndof_cell,
                    'IND_ndof_facet_row':self._ind_row.ndof_bottom_facet,
                    'IND_ndof_cell_col':self._ind_col.ndof_cell,
                    'IND_ndof_facet_col':self._ind_col.ndof_bottom_facet}
        kernel_code = ''' void assemble_lma(double **lma,
                                            double **A) {
          const int alpha = %(SELF_alpha)d;
          const double beta = %(SELF_beta)d;
          const int gamma_p = %(SELF_gamma_p)d;
          const int bandwidth = %(SELF_bandwidth)d;
          const int horiz_extent_col = %(SELF_horiz_extent_col)d;
          const int horiz_extent_row = %(SELF_horiz_extent_row)d;
          const int vert_extent_cell_row = %(IND_ndof_cell_row)d/horiz_extent_row;
          const int vert_extent_facet_row = %(IND_ndof_facet_row)d/horiz_extent_row;
          const int vert_extent_cell_col = %(IND_ndof_cell_col)d/horiz_extent_col;
          const int vert_extent_facet_col = %(IND_ndof_facet_col)d/horiz_extent_col;
          // Create indirection tables
          // row map: (k_local,i) -> nu(k_local,i)
          %(SELF_indirectiontable_row)s;
          // column map: (ell_local,j) -> nu(ell_local,j)
          %(SELF_indirectiontable_col)s;
          double *layer_lma = lma[0];
          // Loop over vertical layers
          for (int celllayer=0;celllayer<%(SELF_ncelllayers)d;++celllayer) {
            // Loop over local vertical dofs in row space
            for (int k_local=0;
                 k_local<(vert_extent_cell_row+2*vert_extent_facet_row);
                 ++k_local) {
              // Loop over local vertical dofs in column space
              for (int ell_local=0;
                   ell_local<(vert_extent_cell_col+2*vert_extent_facet_col);
                   ++ell_local) {
                // Work out global vertical indices (for accessing A)
                int k = celllayer*(vert_extent_cell_row+vert_extent_facet_row)+k_local;
                int ell = celllayer*(vert_extent_cell_col+vert_extent_facet_col)+ell_local;
                int ell_m = (int) ceil((alpha*k-gamma_p)/beta);
                // Loop over horizontal indices in row space
                for (int i=0;i<horiz_extent_row;++i) {
                  // Loop over horizontal indices in column space
                  for (int j=0;j<horiz_extent_col;++j) {
                    A[0][horiz_extent_col*horiz_extent_row*(bandwidth*k+(ell-ell_m))+(i*horiz_extent_col+j)]
                      += layer_lma[%(IND_map_row)s * (%(IND_ncol)d)+%(IND_map_col)s];
                  }
                }
              }
            }
            // point to next vertical layer
            layer_lma += (%(IND_nrow)d) * (%(IND_ncol)d);
          }
        }'''
        self._data.zero()
        param_dict.update(ind_dict)
        kernel_code = kernel_code % param_dict
        kernel = op2.Kernel(kernel_code,'assemble_lma',cpp=True)
        op2.par_loop(kernel,
                     self._hostmesh.cell_set,
                     lma.dat(op2.READ,lma.cell_node_map()),
                     self._data(op2.INC,self._Vcell.cell_node_map()))

    def axpy(self,u,v):
        '''axpy Matrix-vector mutiplication :math:`v\mapsto v+Au`

            :arg u: Vector to multiply
            :arg v: Resulting vector
        '''
        assert(u.function_space() == self._fs_col)
        assert(v.function_space() == self._fs_row)
        ind_dict = {'IND_map_row':self._ind_row.ki_to_index('k','i'),
                    'IND_map_col':self._ind_col.ki_to_index('ell','j')}
        param_dict = {'SELF_'+x:y for (x,y) in self._param_dict.iteritems()}
        kernel_code = '''void axpy(double **A,
                                   double **u,
                                   double **v) {
          const int alpha = %(SELF_alpha)d;
          const double beta = %(SELF_beta)d;
          const int gamma_m = %(SELF_gamma_m)d;
          const int gamma_p = %(SELF_gamma_p)d;
          const int horiz_extent_col = %(SELF_horiz_extent_col)d;
          const int horiz_extent_row = %(SELF_horiz_extent_row)d;
          const int bandwidth = %(SELF_bandwidth)d;
          // Create indirection tables
          // row map: (k,i) -> nu(k,i)
          %(SELF_indirectiontable_row)s;
          // column map: (ell,j) -> nu(ell,j)
          %(SELF_indirectiontable_col)s;
          // Loop over matrix rows
          for (int k=0;k<%(SELF_n_row)d;++k) {
            double s[%(SELF_horiz_extent_row)d];
            // Work out column loop bounds
            int ell_m = (int) ceil((alpha*k-gamma_p)/beta);
            int ell_p = (int) ceil((alpha*k+gamma_m)/beta);
            for (int i=0;i<horiz_extent_row;++i) {
              s[i]=0.0;
            }
            // Loop over columns
            for (int ell=std::max(0,ell_m);ell<std::min(%(SELF_n_col)d,ell_p);++ell) {
              // Calculate s_i += \sum_j [A_{kl}]_{ij}
              for (int i=0;i<horiz_extent_row;++i) {
                for (int j=0;j<horiz_extent_col;++j) {
                  s[i] += A[0][horiz_extent_col*horiz_extent_row*(bandwidth*k+(ell-ell_m)) 
                               +(i*horiz_extent_col+j)]
                        * u[0][%(IND_map_col)s];
                }
              }
            }
            // Update [v_k]_i += s_i = \sum_{l,j} [A_{kl}]_{ij} [u_l]_j
            for (int i=0;i<horiz_extent_row;++i) {
              v[0][%(IND_map_row)s] += s[i];
            }
          }
        }'''
        param_dict.update(ind_dict)
        kernel_code = kernel_code % param_dict
        kernel = op2.Kernel(kernel_code,'axpy',cpp=True)
        op2.par_loop(kernel,
                     self._hostmesh.cell_set,
                     self._data(op2.READ,self._Vcell.cell_node_map()),
                     u.dat(op2.READ,u.cell_node_map()),
                     v.dat(op2.INC,v.cell_node_map()))

    def multiply(self,other,result=None):
        '''Calculate matrix product self*other.

        If result is None, allocate a new matrix, otherwise write data to
        already allocated matrix.

            :arg other: matrix to multiply
            :arg result: resulting matrix
        '''
        # Check that matrices can be multiplied
        assert (self.n_col == other.n_row)
        if (result):
            assert(result.n_row == self.n_row)
            assert(result.n_col == other.n_col)
        else:
            gamma_m = other.alpha * self.gamma_m + self.beta*other.gamma_m
            gamma_p = other.alpha * self.gamma_p + self.beta*other.gamma_p
            result = BandedMatrix(self.fspace_row,other.fspace_col,
                                  gamma_m=gamma_m,gamma_p=gamma_p)
        kernel_code = '''void malmul(double **A,
                                     double **B,
                                     double **C) {          
          const int alpha_A = %(A_alpha)d;
          const double beta_A = %(A_beta)d;
          const int gamma_m_A = %(A_gamma_m)d;
          const int gamma_p_A = %(A_gamma_p)d;
          const int horiz_extent_col_A = %(A_COL_horiz_extent)d;
          const int horiz_extent_row_A = %(A_ROW_horiz_extent)d;
          const int bandwidth_A = %(A_bandwidth)d;
        }'''
        return result

    def add(self,other,result=None):
        '''Calculate matrix sum self*other.

        If result is None, allocate a new matrix, otherwise write data to
        already allocated matrix.

            :arg other: matrix to mass
            :arg result: resulting matrix
        '''
        assert(self.n_row == other.n_row)
        assert(self.n_col == other.n_col)
        if (result):
            assert(result.n_row == self.n_row)
            assert(result.n_col == self.n_col)
        pass

    def scale(self,alpha):
        '''Scale all entries by a factor, i.e. calculate :math:`\alpha A`.

            :arg alpha: scaling factor
        '''
        pass

    def lu_decompose(self):
        '''Construct LU decomposition :math:`A=LU` on the fly.

            Replace A by matrix which stores the lower (L) and
            upper (U) factors of the factorisation, where L is
            assumened to have ones on the diagonal.
        ''' 
        self.lu_decomposed = True
        pass

    def lu_solve(self):
        assert(self._lu_decomposeed)
