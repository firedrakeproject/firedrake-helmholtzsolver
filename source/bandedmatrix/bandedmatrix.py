import fractions
import math
from ufl import HDiv
from firedrake import *
from firedrake.ffc_interface import compile_form
from firedrake.fiat_utils import *

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
        self._ndof_cell_row, self._ndof_bottom_facet_row = self._get_ndof_cell(self._fs_row)
        self._ndof_cell_col, self._ndof_bottom_facet_col = self._get_ndof_cell(self._fs_col)
        self._ndof_row = self._ndof_cell_row+2*self._ndof_bottom_facet_row
        self._ndof_col = self._ndof_cell_col+2*self._ndof_bottom_facet_col
        self._n_row = self._ndof_cell_row*self._ncelllayers \
                    + self._ndof_bottom_facet_row*(self._ncelllayers+1)
        self._n_col = self._ndof_cell_col*self._ncelllayers \
                    + self._ndof_bottom_facet_col*(self._ncelllayers+1)
        
        self._gamma_m = (self._ndof_col-1)*(self._ndof_cell_row+self._ndof_bottom_facet_row)
        self._gamma_p = (self._ndof_row-1)*(self._ndof_cell_col+self._ndof_bottom_facet_col)
        if (gamma_m):
            self._gamma_m = max(self._gamma_m,gamma_m)
        if (gamma_p):
            self._gamma_p = max(self._gamma_p,gamma_p)
        self._alpha = self._ndof_cell_col+self._ndof_bottom_facet_col
        self._beta  = self._ndof_cell_row+self._ndof_bottom_facet_row
        self._divide_by_gcd()
        self._Vcell = FunctionSpace(self._hostmesh,'DG',0)
        self._data = op2.Dat(self._Vcell.node_set**(self.bandwidth * self._n_row))
        self._lu_decomposed = False
        self._param_dict = {'n_row':self._n_row,
                            'n_col':self._n_col,
                            'gamma_m':self._gamma_m,
                            'gamma_p':self._gamma_p,
                            'alpha':self._alpha,
                            'beta':self._beta,
                            'bandwidth':self.bandwidth,
                            'ncelllayers':self._ncelllayers,
                            'ndof_cell_row':self._ndof_cell_row,
                            'ndof_facet_row':self._ndof_bottom_facet_row,
                            'ndof_cell_col':self._ndof_cell_col,
                            'ndof_facet_col':self._ndof_bottom_facet_col}

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
    
    def _get_ndof_cell(self, fs):
        """Count the number of dofs associated with (tdom-1,n) in a function space

        :arg fs: the function space to inspect for the element dof
        ordering.
        """
        ufl_ele = fs.ufl_element()
        # Unwrap non HDiv'd element if necessary
        if isinstance(ufl_ele, HDiv):
            ele = ufl_ele._element
        else:
            ele = ufl_ele
        tdim = ele.cell().topological_dimension()
        element = fiat_from_ufl_element(ele)
        ndof_cell = len(element.entity_dofs()[(tdim-1, 1)][0])
        ndof_bottom_facet = len(element.entity_dofs()[(tdim-1, 0)][0])
        return ndof_cell, ndof_bottom_facet

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
        param_dict = {'A_'+x:y for (x,y) in self._param_dict.iteritems()}
        kernel_code = ''' void assemble_lma(double **lma,
                                            double **A) {
          const int alpha = %(A_alpha)d;
          const double beta = %(A_beta)d;
          const int gamma_p = %(A_gamma_p)d;
          const int bandwidth = %(A_bandwidth)d;
          const int ndof_cell_row = %(A_ndof_cell_row)d;
          const int ndof_facet_row = %(A_ndof_facet_row)d;
          const int ndof_cell_col = %(A_ndof_cell_col)d;
          const int ndof_facet_col = %(A_ndof_facet_col)d;
          const int ndof_row = ndof_cell_row + 2*ndof_facet_row;
          const int ndof_col = ndof_cell_col + 2*ndof_facet_col;
          double *layer_lma = lma[0];
          for (int celllayer=0;celllayer<%(A_ncelllayers)d;++celllayer) {
            // Loop over local vertical dofs in row space
            for (int i_local=0;i_local<ndof_row;++i_local) {
              // Loop over local vertical dofs in column space
              for (int j_local=0;j_local<ndof_col;++j_local) {
                // Work out global vertical indices (for accessing A)
                int i = celllayer*(ndof_cell_row+ndof_facet_row)+i_local;
                int j = celllayer*(ndof_cell_col+ndof_facet_col)+j_local;
                int j_m = (int) ceil((alpha*i-gamma_p)/beta);
                A[0][bandwidth*i+(j-j_m)] += layer_lma[i_local * ndof_col + j_local];
              }
            }
            // point to next vertical layer
            layer_lma += ndof_row * ndof_col;
          }
        }'''
        self._data.zero()
        kernel = op2.Kernel(kernel_code % param_dict,'assemble_lma',cpp=True)
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
        param_dict = {'A_'+x:y for (x,y) in self._param_dict.iteritems()}
        kernel_code = '''void axpy(double **A,
                                   double **u,
                                   double **v) {
          const int alpha = %(A_alpha)d;
          const double beta = %(A_beta)d;
          const int gamma_m = %(A_gamma_m)d;
          const int gamma_p = %(A_gamma_p)d;
          const int bandwidth = %(A_bandwidth)d;
          // Loop over matrix rows
          for (int i=0;i<%(A_n_row)d;++i) {
            double s = 0;
            // Work out column loop bounds
            int j_m = (int) ceil((alpha*i-gamma_p)/beta);
            int j_p = (int) floor((alpha*i+gamma_m)/beta);
            // Loop over columns
            for (int j=std::max(0,j_m);j<std::min(%(A_n_col)d,j_p+1);++j) {
               s += A[0][bandwidth*i+(j-j_m)] * u[0][j];
            }
            v[0][i] += s;
          }
        }'''
        kernel = op2.Kernel(kernel_code % param_dict,'axpy',cpp=True)
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
