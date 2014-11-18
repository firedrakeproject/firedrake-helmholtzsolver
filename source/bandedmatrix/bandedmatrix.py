import fractions
import math

class BGMatric(object):
    def __init__(self,n_to,n_from,ind_n_to,ind_n_from,gamma_m,gamma_p,alpha,beta):
        '''Generalised block banded matrix.

        :math:`n_{to}\times n_{from}` matrix over the field of dense :math:`n_h \times n_h` 
        matrices with entries only for row-indices :math:`k` and column 
        index :math:`\ell` for which

            :math::
                -\gamma_- \le \alpha k - \beta \ell \le \gamma_+
        
            :arg n_to: Number of rows
            :arg n_from: Number of columns
            :arg gamma_m: band-parameter :math:`\gamma_-`
            :arg gamma_p: band-parameter :math:`\gamma_+`
            :arg alpha: band-parameter :math:`\alpha`
            :arg beta: band-parameter :math:`\beta`
        '''
        self.n_to = n_to
        self.n_from = n_from
        self.gamma_m = gamma_m
        self.gamma_p = gamma_p
        self.beta = beta
        self.alpha = alpha
        # Bandwidth
        self.bandwidth = 1+int(math.ceil((self.gamma_h+self.g)/float(self.beta)))
        self._divide_by_gcd()
        self.is_square = (self.n_to == self.n_from)
        self.is_sparsity_symmetric = ( (self.is_square) and \
                                       (self.alpha == self.beta) and \
                                       (self.gamma_m == self.gamma_p) )
        self.lu_decomposed = False
        self.param_dict = {'SELF_n_to':self.n_to,
                           'SELF_n_from':self.n_from,
                           'SELF_gamma_m':self.gamma_m,
                           'SELF_gamma_p':self.gamma_p,
                           'SELF_alpha':self.alpha,
                           'SELF_beta':self.beta,
                           'SELF_bandwidth':self.bandwidth}
    
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
        ind_dict = {'IND_FROM_map':self.ind_from.map('ell','i','perm'),
                    'IND_TO_map':self.ind_from.map('k','i','perm')}
        kernel_code = '''void axpy(double **data,
                                   double **u,
                                   double **v) {
          const int alpha = %(SELF_alpha)d;
          const double beta = %(SELF_beta)d;
          const int gamma_m = %(SELF_gamma_m)d;
          const int gamma_p = %(SELF_gamma_p)d;
          const int ndof_h_from = %(IND_FROM_ndof_h)d;
          const int ndof_h_to = %(IND_TO_ndof_h)d;
          const int bandwidth = %(SELF_bandwidth)d;
          for (int k=0;k<%(SELF_n_to)d;++k) {
            double s[ndof_h_to];
            int ell_m = int(ceil(alpha*k-gamma_p)/beta);
            int ell_p = int(ceil(alpha*k+gamma_m)/beta);
            for (int i=0;i<ndof_h_to;++i) {
              s[i]=0.0;
            }
            for (int ell=max(0,ell_m);ell<min(%(SELF_n_from)d,ell_p);++ell) {
              for (int i=0;i<ndof_h_to;++i) {
                for (int j=0;j<ndof_h_from;++j) {
                  s[i] += data[0][ndof_h_from*ndof_h_to*(bandwidth*k+(ell-ell_-))+j]
                        * u[0][%(IND_FROM_map)s];
                }
              }
            }
            for (int i=0;i<ndof_h_to;++i) {
              v[0][%(IND_TO_map)s] += s[i];
            }
          }
        }'''

        
    

    def multiply(self,other,result=None):
        '''Calculate matrix product self*other.

        If result is None, allocate a new matrix, otherwise write data to
        already allocated matrix.

            :arg other: matrix to multiply
            :arg result: resulting matrix
        '''
        # Check that matrices can be multiplied
        assert (self.n_from == other.n_to)
        if (result):
            assert(result.n_to == self.n_to)
            assert(result.n_from = other.n_from)
        pass

    def add(self,other,result=None):
        '''Calculate matrix sum self*other.

        If result is None, allocate a new matrix, otherwise write data to
        already allocated matrix.

            :arg other: matrix to mass
            :arg result: resulting matrix
        '''
        assert(self.n_to == other.n_to)
        assert(self.n_from == other.n_from)
        if (result):
            assert(result.n_to == self.n_to)
            assert(result.n_from = self.n_from)
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
