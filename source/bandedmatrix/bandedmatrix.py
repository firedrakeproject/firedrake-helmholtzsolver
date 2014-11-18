import fractions
import math

class BandedMatrix(object):
    def __init__(self,n_row,n_col,ind_n_row,ind_n_col,gamma_m,gamma_p,alpha,beta):
        '''Generalised block banded matrix.

        :math:`n_{to}\times n_{from}` matrix over the field of dense :math:`n_h \times n_h` 
        matrices with entries only for row-indices :math:`k` and column 
        index :math:`\ell` for which

            :math::
                -\gamma_- \le \alpha k - \beta \ell \le \gamma_+
        
            :arg n_row: Number of rows
            :arg n_col: Number of columns
            :arg gamma_m: band-parameter :math:`\gamma_-`
            :arg gamma_p: band-parameter :math:`\gamma_+`
            :arg alpha: band-parameter :math:`\alpha`
            :arg beta: band-parameter :math:`\beta`
        '''
        self.n_row = n_row
        self.n_col = n_col
        self.gamma_m = gamma_m
        self.gamma_p = gamma_p
        self.beta = beta
        self.alpha = alpha
        # Bandwidth
        self.bandwidth = 1+int(math.ceil((self.gamma_h+self.g)/float(self.beta)))
        self._divide_by_gcd()
        self.is_square = (self.n_row == self.n_col)
        self.is_sparsity_symmetric = ( (self.is_square) and \
                                       (self.alpha == self.beta) and \
                                       (self.gamma_m == self.gamma_p) )
        self.lu_decomposed = False
        self.param_dict = {'SELF_n_row':self.n_row,
                           'SELF_n_col':self.n_col,
                           'SELF_gamma_m':self.gamma_m,
                           'SELF_gamma_p':self.gamma_p,
                           'SELF_alpha':self.alpha,
                           'SELF_beta':self.beta,
                           'SELF_bandwidth':self.bandwidth,
                           'SELF_COL_table':self.ind_col.maptable(),
                           'SELF_ROW_table':self.ind_row.maptable(),
                           'SELF_ndof_h_col':self.ind_col.ndof_h(),
                           'SELF_ndof_h_row':self.ind_row.ndof_h()
                          }
    
    def _divide_by_gcd(self):
        '''If alpha and beta have a gcd > 1, divide by this.
        '''
        gcd = fractions.gcd(self.alpha,self.beta)
        if (gcd > 1):
            self.alpha /= gcd
            self.beta /= gcd
            self.gamma_m /= gcd
            self.gamma_p /=gcd
        
    def axpy(self,u,v):
        '''axpy Matrix-vector mutiplication :math:`v\mapsto v+Au`

            :arg u: Vector to multiply
            :arg v: Resulting vector
        '''
        ind_dict = {'IND_COL_map':self.ind_col.('ell','i'),
                    'IND_ROW_map':self.ind_row.map('k','i')}
        kernel_code = '''void axpy(double **data,
                                   double **u,
                                   double **v) {
          const int alpha = %(SELF_alpha)d;
          const double beta = %(SELF_beta)d;
          const int gamma_m = %(SELF_gamma_m)d;
          const int gamma_p = %(SELF_gamma_p)d;
          const int ndof_h_col = %(SELF_COL_ndof_h)d;
          const int ndof_h_row = %(SELF_ROW_ndof_h)d;
          const int bandwidth = %(SELF_bandwidth)d;
          %(SELF_COL_map)s;
          %(SELF_ROW_map)s;
          for (int k=0;k<%(SELF_n_row)d;++k) {
            double s[ndof_h_row];
            int ell_m = int(ceil(alpha*k-gamma_p)/beta);
            int ell_p = int(ceil(alpha*k+gamma_m)/beta);
            for (int i=0;i<ndof_h_row;++i) {
              s[i]=0.0;
            }
            for (int ell=max(0,ell_m);ell<min(%(SELF_n_col)d,ell_p);++ell) {
              for (int i=0;i<ndof_h_row;++i) {
                for (int j=0;j<ndof_h_col;++j) {
                  s[i] += data[0][ndof_h_col*ndof_h_row*(bandwidth*k+(ell-ell_-))+j]
                        * u[0][%(IND_COL_map)s];
                }
              }
            }
            for (int i=0;i<ndof_h_row;++i) {
              v[0][%(IND_ROW_map)s] += s[i];
            }
          }
        }'''
        kernel = op2.Kernel(kernel_code,'axpy')
        op2.par_loop(kernel,
                     self.data(op2.READ,cell->DG0),
                     u.dat(op2.READ,???),
                     v.dat(op2.INC,???))

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
            assert(result.n_col = other.n_col)
        pass

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
            assert(result.n_col = self.n_col)
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
        assert(self.lu_decomposeed):
