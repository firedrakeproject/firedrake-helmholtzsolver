import fractions
import math
import numpy as np
from ufl import HDiv
from firedrake import *
from firedrake.ffc_interface import compile_form
from firedrake.fiat_utils import *

class BandedMatrix(object):
    def __init__(self,fs_row,fs_col,alpha=None,beta=None,gamma_m=None,gamma_p=None):
        '''Generalised block banded matrix.

        :math:`n_{row}\\times n_{col}` matrix with entries only for 
        row-indices :math:`i` and column indices :math:`j` for which satisfy

            :math::
                -\gamma_- \le \\alpha i - \\beta j \le \gamma_+

        Internally the matrix is stored in a sparse format as an array 
        :math:`\overline{A}` of length :math:`n_{row}BW` with the bandwidth BW defined 
        as :math:`BW=1+\lceil((\gamma_++\gamma_-)/\\beta)\\rceil`. Element :math:`A_{ij}`
        can be accessed as :math:`A_{ij}=\overline{A}_{BW\cdot i+(j-j_-(i))}` where
        :math:`j_-(i) = \lceil((\\alpha i-\gamma_+)/\\beta)\`rceil`.
        
            :arg fs_row: Row function space
            :arg fs_col: Column function space
            :arg alpha: Parameter :math:`alpha`
            :arg beta: Parameter :math:`beta`
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
        
        if (gamma_m):
            self._gamma_m = gamma_m
        else:
            self._gamma_m = (self._ndof_col-1)*(self._ndof_cell_row+self._ndof_bottom_facet_row)
        if (gamma_p):
            self._gamma_p = gamma_p
        else:
            self._gamma_p = (self._ndof_row-1)*(self._ndof_cell_col+self._ndof_bottom_facet_col)
        if (alpha):
            self._alpha = alpha
        else:
            self._alpha = self._ndof_cell_col+self._ndof_bottom_facet_col
        if (beta):
            self._beta = beta
        else:
            self._beta  = self._ndof_cell_row+self._ndof_bottom_facet_row
        self._divide_by_gcd()
        self._Vcell = FunctionSpace(self._hostmesh,'DG',0)
        # Data array
        self._data = op2.Dat(self._Vcell.node_set**(self.bandwidth * self._n_row))
        self._data.zero()
        self._lu_decomposed = False
        self._nodemap_row = self._get_nodemap(self._fs_row)
        self._nodemap_col = self._get_nodemap(self._fs_col)
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
                            'ndof_facet_col':self._ndof_bottom_facet_col,
                            'nodemap_row':'{%s}' % ', '.join('%d' % o for o in self._nodemap_row),
                            'nodemap_col':'{%s}' % ', '.join('%d' % o for o in self._nodemap_col),
                            'n_nodemap_row':len(self._nodemap_row),
                            'n_nodemap_col':len(self._nodemap_col)}
        self._lu = None
        self._ipiv = None

    def _get_nodemap(self,fs):
        '''Return node map of first base cell in the extruded mesh.'''
        return fs.cell_node_map().values[0]

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

        In each cell of the extruded mesh, build the local matrix stencil associated
        with the UFL form. Then call _assemble_lma() to loop over all cells and assemble 
        into the banded matrix.

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
        '''Assemble the matrix from the LMA storage format.'''
        param_dict = {'A_'+x:y for (x,y) in self._param_dict.iteritems()}
        kernel_code = ''' void assemble_lma(double **lma,
                                            double **A) {
          const int ndof_cell_row = %(A_ndof_cell_row)d;
          const int ndof_facet_row = %(A_ndof_facet_row)d;
          const int ndof_cell_col = %(A_ndof_cell_col)d;
          const int ndof_facet_col = %(A_ndof_facet_col)d;
          const int ndof_row = ndof_cell_row + 2*ndof_facet_row;
          const int ndof_col = ndof_cell_col + 2*ndof_facet_col;
          const int nodemap_row[%(A_n_nodemap_row)d] = %(A_nodemap_row)s;
          const int nodemap_col[%(A_n_nodemap_col)d] = %(A_nodemap_col)s;
          double *layer_lma = lma[0];
          for (int celllayer=0;celllayer<%(A_ncelllayers)d;++celllayer) {
            // Loop over local vertical dofs in row space
            for (int i_local=0;i_local<ndof_row;++i_local) {
              // Loop over local vertical dofs in column space
              for (int j_local=0;j_local<ndof_col;++j_local) {
                // Work out global vertical indices (for accessing A)
                int i = celllayer*(ndof_cell_row+ndof_facet_row)+nodemap_row[i_local];
                int j = celllayer*(ndof_cell_col+ndof_facet_col)+nodemap_col[j_local];
                int j_m = (int) ceil((%(A_alpha)d*i-%(A_gamma_p)d)/%(A_beta)f);
                A[0][%(A_bandwidth)d*i+(j-j_m)] += layer_lma[i_local * ndof_col + j_local];
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
          // Loop over matrix rows
          for (int i=0;i<%(A_n_row)d;++i) {
            double s = 0;
            // Work out column loop bounds
            int j_m = (int) ceil((%(A_alpha)d*i-%(A_gamma_p)d)/%(A_beta)f);
            int j_p = (int) floor((%(A_alpha)d*i+%(A_gamma_m)d)/%(A_beta)f);
            // Loop over columns
            for (int j=std::max(0,j_m);j<std::min(%(A_n_col)d,j_p+1);++j) {
               s += A[0][%(A_bandwidth)d*i+(j-j_m)] * u[0][j];
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

    def matmul(self,other,result=None):
        '''Calculate matrix product :math:`C=AB`.

        Multiply this matrix by the banded matrix :math:`B` from the right and store the
        result in the matrix :math:`C`. If result is None, allocate a new matrix, otherwise 
        write data to already allocated matrix. 

            :arg other: matrix :math:`B` to multiply with
            :arg result: resulting matrix :math:`C`
        '''
        # Check that matrices can be multiplied
        assert (self._n_col == other._n_row)
        if (result):
            assert(result._n_row == self._n_row)
            assert(result._n_col == other._n_col)
        else:
            alpha = self.alpha * other.alpha
            beta = self.beta * other.beta
            gamma_m = other.alpha * self.gamma_m + self.beta*other.gamma_m
            gamma_p = other.alpha * self.gamma_p + self.beta*other.gamma_p
            result = BandedMatrix(self._fs_row,other._fs_col,
                                  alpha=alpha,beta=beta,
                                  gamma_m=gamma_m,gamma_p=gamma_p)

        param_dict = {}
        for label, matrix in zip(('A','B','C'),(self,other,result)):
            param_dict.update({label+'_'+x:y for (x,y) in matrix._param_dict.iteritems()})
        kernel_code = '''void matmul(double **A,
                                     double **B,
                                     double **C) {
          for (int i=0;i<%(C_n_row)d;++i) {
            int j_m = (int) ceil((%(C_alpha)d*i-%(C_gamma_p)d)/(1.0*%(C_beta)f));
            int j_p = (int) floor((%(C_alpha)d*i+%(C_gamma_m)d)/(1.0*%(C_beta)f));
            int k_m = (int) ceil((%(A_alpha)d*i-%(A_gamma_p)d)/(1.0*%(A_beta)f));
            int k_p = (int) floor((%(A_alpha)d*i+%(A_gamma_m)d)/(1.0*%(A_beta)f));
            for (int j=std::max(0,j_m);j<std::min(%(C_n_col)d,j_p+1);++j) {
              double s = 0.0;
              for (int k=std::max(0,k_m);k<std::min(%(A_n_col)d,k_p+1);++k) {
                if ( (ceil((%(B_alpha)d*k-%(B_gamma_p)d)/%(B_beta)f) <= j) &&
                     (j <= floor((%(B_alpha)d*k+%(B_gamma_m)d)/(1.0*%(B_beta)f))) ) {
                  int j_m_B = (int) ceil((%(B_alpha)d*k-%(B_gamma_p)d)/(1.0*%(B_beta)f));
                  s += A[0][%(A_bandwidth)d*i+(k-k_m)]
                     * B[0][%(B_bandwidth)d*k+(j-j_m_B)];
                }
              }
              C[0][%(C_bandwidth)d*i+(j-j_m)] = s;
            }
          }
        }'''
        kernel = op2.Kernel(kernel_code % param_dict, 'matmul',cpp=True)
        op2.par_loop(kernel,
                     self._hostmesh.cell_set,
                     self._data(op2.READ,self._Vcell.cell_node_map()),
                     other._data(op2.READ,self._Vcell.cell_node_map()),
                     result._data(op2.WRITE,self._Vcell.cell_node_map()))
        return result

    def matadd(self,other,omega=1.0,result=None):
        '''Calculate matrix sum :math:A+\omega B.

        Add the banded matrix :math:`\omega B` to this matrix and store the result in the 
        banded matrix :math:`C`. If result is None, allocate a new matrix, otherwise write 
        data to already allocated matrix.

            :arg other: matrix :math:`B` to add
            :arg omega: real scaling factor :math:`\omega`
            :arg result: resulting matrix :math:`C`
        '''
        assert(self._n_row == other._n_row)
        assert(self._n_col == other._n_col)
        assert(self.alpha == other.alpha)
        assert(self.beta == other.beta)
        gamma_m = max(self.gamma_m,other.gamma_m)
        gamma_p = max(self.gamma_p,other.gamma_p)
        if (result):
            assert(result._n_row == self._n_row)
            assert(result._n_col == self._n_col)
            assert(result.alpha == self.alpha)
            assert(result.beta == self.beta)
            assert(result.gamma_m >= gamma_m)
            assert(result.gamma_p >= gamma_p)
        else:
            result = BandedMatrix(self._fs_row,other._fs_col,
                                  alpha=self.alpha,beta=self.beta,
                                  gamma_m=gamma_m,gamma_p=gamma_p)
        param_dict = {}
        for label, matrix in zip(('A','B','C'),(self,other,result)):
            param_dict.update({label+'_'+x:y for (x,y) in matrix._param_dict.iteritems()})
        param_dict.update({'omega':omega})
        kernel_code = '''void matadd(double **A,
                                     double **B,
                                     double **C) {
          for (int i=0;i<%(C_n_row)d;++i) {
            int j_m_C = (int) ceil((%(C_alpha)d*i-%(C_gamma_p)d)/%(C_beta)f);
            int j_p_C = (int) floor((%(C_alpha)d*i+%(C_gamma_m)d)/%(C_beta)f);
            int j_m_A = (int) ceil((%(A_alpha)d*i-%(A_gamma_p)d)/%(A_beta)f);
            int j_p_A = (int) floor((%(A_alpha)d*i+%(A_gamma_m)d)/%(A_beta)f);
            int j_m_B = (int) ceil((%(B_alpha)d*i-%(B_gamma_p)d)/%(B_beta)f);
            int j_p_B = (int) floor((%(B_alpha)d*i+%(B_gamma_m)d)/%(B_beta)f);
            for (int j=std::max(0,j_m_A);j<std::min(%(A_n_col)d,j_p_A+1);++j) {
              C[0][%(C_bandwidth)d*i+(j-j_m_C)] += A[0][%(A_bandwidth)d*i+(j-j_m_A)];
            }
            for (int j=std::max(0,j_m_B);j<std::min(%(B_n_col)d,j_p_B+1);++j) {
              C[0][%(C_bandwidth)d*i+(j-j_m_C)] += %(omega)f*B[0][%(B_bandwidth)d*i+(j-j_m_B)];
            }
          }
        }'''
        kernel = op2.Kernel(kernel_code % param_dict, 'matadd',cpp=True)
        op2.par_loop(kernel,
                     self._hostmesh.cell_set,
                     self._data(op2.READ,self._Vcell.cell_node_map()),
                     other._data(op2.READ,self._Vcell.cell_node_map()),
                     result._data(op2.WRITE,self._Vcell.cell_node_map()))
        return result
            
    def lu_decompose(self):
        '''Construct LU decomposition :math:`A=LU` on the fly.

            Replace A by matrix which stores the lower (L) and
            upper (U) factors of the factorisation, where L is
            assumened to have ones on the diagonal.
        ''' 
        # Number of super-diagonals (ku): gamma_m
        # Number of sub-diagonals (kl): gamma_p
        # Storage for LU decomposition is n_{row} * (1+ku+kl)+kl
        # (see http://www.netlib.org/lapack/lug/node124.html and 
        # documentation of DGBTRF http://phase.hpcc.jp/mirrors/netlib/lapack/double/dgbtrf.f)
        # The LAPACKe C-interface to LAPACK is used, see
        # http://www.netlib.org/lapack/lapacke.html
        assert (self._n_row == self._n_col)
        lda = self.bandwidth+self.gamma_p
        if (not self._lu):
            self._lu = op2.Dat(self._Vcell.node_set**(lda * self._n_row))
            self._lu.zero()
        if (not self._ipiv):
            self._ipiv = op2.Dat(self._Vcell.node_set**(self._n_row),dtype=np.int32)
        # Copy data into array which will be LU decomposed.
        kernel_code = '''void lu_decompose(double **A,
                                           double **LU,
                                           int **ipiv) {
          // Step 1: write to column-major LU matrix
          for (int i=0;i<%(A_n_row)d;++i) {
            int j_m = (int) ceil((%(A_alpha)d*i-%(A_gamma_p)d)/%(A_beta)f);
            int j_p = (int) floor((%(A_alpha)d*i+%(A_gamma_m)d)/%(A_beta)f);
            for (int j=std::max(0,j_m);j<std::min(%(A_n_col)d,j_p+1);++j) {
              LU[0][%(A_bandwidth)d-1+(i-j)+(%(A_bandwidth)d+%(A_gamma_p)d)*j]
                = A[0][%(A_bandwidth)d*i+(j-j_m)];
            }
          }
          // Step 2: Call LAPACK's DGBTRF routine to LU decompose the matrix
          LAPACKE_dgbtrf_work(LAPACK_COL_MAJOR,
                              %(A_n_row)d,%(A_n_row)d,
                              %(A_gamma_p)d,%(A_gamma_m)d,
                              LU[0],%(A_bandwidth)d+%(A_gamma_p)d,ipiv[0]);
        }'''
        param_dict = {'A_'+x:y for (x,y) in self._param_dict.iteritems()}
        kernel = op2.Kernel(kernel_code % param_dict, 'lu_decompose',
                            cpp=True,
                            headers=['#include "lapacke.h"'])
        op2.par_loop(kernel,
                     self._hostmesh.cell_set,
                     self._data(op2.READ,self._Vcell.cell_node_map()),
                     self._lu(op2.WRITE,self._Vcell.cell_node_map()),
                     self._ipiv(op2.WRITE,self._Vcell.cell_node_map()))
        self._lu_decomposed = True

    def lu_solve(self,u):
        '''In-place LU solve for a field u.
        
        :arg u: Function to be solved for.
        '''
        assert(self._lu_decomposed)
        kernel_code = '''void lu_solve(double **LU,
                                       int **ipiv,
                                       double **u) {
          LAPACKE_dgbtrs_work(LAPACK_COL_MAJOR,'N',
                              %(A_n_row)d,%(A_gamma_p)d,%(A_gamma_m)d,1,
                              LU[0],%(A_bandwidth)d+%(A_gamma_p)d,ipiv[0],
                              u[0],%(A_n_row)d);

        }'''
        param_dict = {'A_'+x:y for (x,y) in self._param_dict.iteritems()}
        kernel = op2.Kernel(kernel_code % param_dict, 'lu_solve',
                            cpp=True,
                            headers=['#include "lapacke.h"'])
        op2.par_loop(kernel,
                     self._hostmesh.cell_set,
                     self._lu(op2.WRITE,self._Vcell.cell_node_map()),
                     self._ipiv(op2.WRITE,self._Vcell.cell_node_map()),
                     u.dat(op2.RW,u.cell_node_map()))
        return u
