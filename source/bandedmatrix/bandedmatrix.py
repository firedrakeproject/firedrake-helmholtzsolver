import fractions
import math
import numpy as np
from ufl import HDiv
from firedrake import *
from firedrake.ffc_interface import compile_form
from firedrake.fiat_utils import *
import socket
from pyop2.profiling import timed_region

class BandedMatrix(object):
    '''Generalised block banded matrix.

        :math:`n_{row}\\times n_{col}` matrix with entries only for 
        row-indices :math:`i` and column indices :math:`j` for which satisfy

        .. math::
                -\gamma_- \le \\alpha i - \\beta j \le \gamma_+

        Internally the matrix is stored in a sparse format as an array 
        :math:`\overline{A}` of length :math:`n_{row}\cdot BW` with the bandwidth defined 
        as :math:`BW=1+\lceil((\gamma_++\gamma_-)/\\beta)\\rceil`. Element :math:`A_{ij}`
        can be accessed as :math:`A_{ij}=\overline{A}_{BW\cdot i+(j-j_-(i))}` where
        :math:`j_-(i) = \lceil((\\alpha i-\gamma_+)/\\beta)\\rceil`.

        If the parameters :math:`\\alpha`, :math:`\\beta`, :math:`\\gamma_-` and
        :math:`\\gamma_+` are not given explicitly, they are deduced from the function spaces
        fs_row and fs_col.

        For more details see also section 3.2 in `notes in LaTeX <./GravityWaves.pdf>`_.
        
        :arg fs_row: Row function space
        :arg fs_col: Column function space
        :arg alpha: Parameter :math:`\\alpha`
        :arg beta: Parameter :math:`\\beta`
        :arg gamma_m: Lower bound :math:`\\gamma_-`
        :arg gamma_p: Upper bound :math:`\\gamma_+`
        :arg use_blas_for_axpy: Use BLAS routine dgbmv in axpy operation for matrices
            with :math:`\\alpha = \\beta`
        '''
    def __init__(self,fs_row,fs_col,alpha=None,beta=None,gamma_m=None,gamma_p=None,
                 use_blas_for_axpy=False,label=None):
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
        
        if (gamma_m != None):
            self._gamma_m = gamma_m
        else:
            self._gamma_m = (self._ndof_col-1)*(self._ndof_cell_row+self._ndof_bottom_facet_row)
        if (gamma_p != None):
            self._gamma_p = gamma_p
        else:
            self._gamma_p = (self._ndof_row-1)*(self._ndof_cell_col+self._ndof_bottom_facet_col)
        if (alpha != None):
            self._alpha = alpha
        else:
            self._alpha = self._ndof_cell_col+self._ndof_bottom_facet_col
        if (beta != None):
            self._beta = beta
        else:
            self._beta  = self._ndof_cell_row+self._ndof_bottom_facet_row
        self._divide_by_gcd()
        self._Vcell = FunctionSpace(self._hostmesh,'DG',0)
        # Data array
        self._data = op2.Dat(self._Vcell.node_set**(self.bandwidth * self._n_row))
        self._data.zero()
        self._lu_version = None
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
        hostname = socket.gethostname()
        self._libs = []
        if ( ('Eikes-MacBook-Pro.local' in hostname) or \
             ('eduroam.bath.ac.uk' in hostname) or \
             ('Eikes-MBP' in hostname) ):
            self._libs = ['lapack','lapacke','cblas','blas']
        self._use_blas_for_axpy = False
        if (label==None):
            self._label = '___'
        else:
            self._label = label

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
        '''Bandedness parameter :math:`\\alpha`'''
        return self._alpha
    
    @property
    def beta(self):
        '''Bandedness parameter :math:`\\beta`'''
        return self._beta
    
    @property
    def gamma_m(self):
        '''Bandedness parameter :math:`\\gamma_-`'''
        return self._gamma_m
    
    @property
    def gamma_p(self):
        '''Bandedness parameter :math:`\\gamma_+`'''
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
        '''Count the number of dofs associated with (tdim-1,n) in a function space

        :arg fs: the function space to inspect for the element dof
        ordering.
        '''
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

    def assemble_ufl_form(self,ufl_form,vertical_bcs=False):
        '''Assemble the matrix form a UFL form.

        In each cell of the extruded mesh, build the local matrix stencil associated
        with the UFL form. Then call _assemble_lma() to loop over all cells and assemble 
        into the banded matrix.

        If the flag ``vertical_bcs`` is set, then homogeneous boundary conditions on 
        the top and bottom surfaces are assumed (on the column space).

        :arg ufl_form: UFL form to assemble
        :arg vertical_bcs: Apply homogeneous Dirichlet boundary conditions on the
            top and bottom surfaces.
        '''

        with timed_region('bandedmatrix compile_ufl_form'):
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
        with timed_region('bandedmatrix assemble_ufl_form'):
            op2.par_loop(kernel,lma.cell_set, *args)
        self._assemble_lma(lma,vertical_bcs)
        
    def _assemble_lma(self,lma,vertical_bcs=False):
        '''Assemble the matrix from the LMA storage format.

        If the flag ``vertical_bcs`` is set, then homogeneous boundary conditions on 
        the top and bottom surfaces are assumed (on the column space).

        :arg lma: Matrix in LMA storage format.
        :arg vertical_bcs: Apply homogeneous Dirichlet boundary conditions on the
            top and bottom surfaces.
        '''
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
        with timed_region('bandedmatrix assemble_lma'):
            op2.par_loop(kernel,
                         self._hostmesh.cell_set,
                         lma.dat(op2.READ,lma.cell_node_map()),
                         self._data(op2.INC,self._Vcell.cell_node_map()))
        if (vertical_bcs):
            self.apply_vertical_bcs()
        
    def apply_vertical_bcs(self):
        '''Apply homogeneous boundary conditions on the top and bottom surfaces.

        Loop over all matrix rows i and columns j and set :math:`A_{ij}=\delta_{ij}` 
        if i or j is the index of a dof on the top or bottom surface of the domain.
        '''
        # identify local indices of dofs on top and bottom boundaries
        boundary_nodes = {}
        for label, bc_masks, ndof, nodemap in \
            zip(('col','row'), \
                 (self._fs_col.bt_masks['topological'],
                  self._fs_row.bt_masks['topological']), \
                 (self._ndof_cell_col+self._ndof_bottom_facet_col, \
                  self._ndof_cell_row+self._ndof_bottom_facet_row), \
                 (self._nodemap_col, self._nodemap_row)):
            offset = (self._ncelllayers-1)*ndof
            boundary_nodes[label] = np.concatenate(([-1],nodemap[bc_masks[0]], \
                                                    offset + nodemap[bc_masks[1]]))
        declare_boundary_nodes = ''
        for label,nodes in boundary_nodes.iteritems():
            declare_boundary_nodes += 'int boundary_nodes_'+label+'['+str(len(nodes))+'] = '
            declare_boundary_nodes += '{%s};\n' % ', '.join('%d' % o for o in nodes)
        param_dict = {'A_'+x:y for (x,y) in self._param_dict.iteritems()}
        param_dict.update({'DECLARE_BOUNDARY_NODES':declare_boundary_nodes,
                           'n_boundary_nodes_row':len(boundary_nodes['row']),
                           'n_boundary_nodes_col':len(boundary_nodes['col'])})
        kernel_code ='''void apply_bcs(double **A) {
          #include <stdio.h>
          %(DECLARE_BOUNDARY_NODES)s
          // Loop over matrix rows  
          // Skip the first entry which is always -1
          // Zero out rows
          for (int i_bd=1;i_bd<%(n_boundary_nodes_row)d;++i_bd) {
            int i = boundary_nodes_row[i_bd];
            // Work out column loop bounds
            int j_m = (int) ceil((%(A_alpha)d*i-%(A_gamma_p)d)/%(A_beta)f);
            int j_p = (int) floor((%(A_alpha)d*i+%(A_gamma_m)d)/%(A_beta)f);
            for (int j=std::max(0,j_m);j<std::min(%(A_n_col)d,j_p+1);++j) {
               A[0][%(A_bandwidth)d*i+(j-j_m)] = 0;
            }
          }
          // Zero out columns
          for (int i=0;i<%(A_n_row)d;++i) {
            int j_m = (int) ceil((%(A_alpha)d*i-%(A_gamma_p)d)/%(A_beta)f);
            int j_p = (int) floor((%(A_alpha)d*i+%(A_gamma_m)d)/%(A_beta)f);
            for (int j_bd=1;j_bd<%(n_boundary_nodes_col)d;++j_bd) {
              int j = boundary_nodes_col[j_bd];
              if ( (j_m <= j) && (j <= j_p) ) {
                A[0][%(A_bandwidth)d*i+(j-j_m)] = 0;
              }
            }
          }
          // Set diagonal entries to 1.
          for (int i_bd=1;i_bd<%(n_boundary_nodes_row)d;++i_bd) {
            int i = boundary_nodes_row[i_bd];
            // Work out column loop bounds
            int j_m = (int) ceil((%(A_alpha)d*i-%(A_gamma_p)d)/%(A_beta)f);
            int j_p = (int) floor((%(A_alpha)d*i+%(A_gamma_m)d)/%(A_beta)f);
            // Loop over all columns
            for (int j_bd=1;j_bd<%(n_boundary_nodes_col)d;++j_bd) {
              int j = boundary_nodes_col[j_bd];
              if ( (j_m <= j) && (j <= j_p) ) {
                A[0][%(A_bandwidth)d*i+(j-j_m)] = (i==j);
              }
            }
          }
        }
        '''
        kernel = op2.Kernel(kernel_code % param_dict,'apply_bcs',cpp=True)
        with timed_region('bandedmatrix apply_bcs'):
            op2.par_loop(kernel,
                         self._hostmesh.cell_set,
                         self._data(op2.WRITE,self._Vcell.cell_node_map()))

    def ax(self,u):
        '''In-place Matrix-vector mutiplication :math:`u\mapsto Au`

        :arg u: Function to multiply, on exit this contains result :math:`Au`
        '''
        assert(u.function_space() == self._fs_col)
        assert(u.function_space() == self._fs_row)
        param_dict = {'A_'+x:y for (x,y) in self._param_dict.iteritems()}
        kernel_code = '''void ax(double **A,
                                 double **u) {
          // Copy vector into temporary array
          double u_tmp[%(A_n_row)d];
          for (int i=0;i<%(A_n_row)d;++i) {
            u_tmp[i] = u[0][i];
          }
        '''
        if (self._alpha == self._beta) and self._use_blas_for_axpy:
            kernel_code +='''
            cblas_dgbmv(CblasColMajor,CblasTrans,
                        %(A_n_col)d,%(A_n_row)d,
                        %(A_gamma_m)d,%(A_gamma_p)d,
                        1.0,A[0],%(A_bandwidth)d,
                        u_tmp,1,0.0,u[0],1);
            '''
        else:
            kernel_code += '''
            // Loop over matrix rows
            for (int i=0;i<%(A_n_row)d;++i) {
              double s = 0;
              // Work out column loop bounds
              int j_m = (int) ceil((%(A_alpha)d*i-%(A_gamma_p)d)/%(A_beta)f);
              int j_p = (int) floor((%(A_alpha)d*i+%(A_gamma_m)d)/%(A_beta)f);
              // Loop over columns
              for (int j=std::max(0,j_m);j<std::min(%(A_n_col)d,j_p+1);++j) {
                 s += A[0][%(A_bandwidth)d*i+(j-j_m)] * u_tmp[j];
              }
              u[0][i] = s;
            }
            '''
        kernel_code +='''}'''
        kernel = op2.Kernel(kernel_code % param_dict,'ax',cpp=True,
                            headers=['#include "cblas.h"'],
                            libs=self._libs)
        with timed_region('bandedmatrix ax'):
            op2.par_loop(kernel,
                         self._hostmesh.cell_set,
                         self._data(op2.READ,self._Vcell.cell_node_map()),
                         u.dat(op2.RW,u.cell_node_map()),
                         name='bandedmatrix ax['+self._label+']',
                         measure_flops=(not self._use_blas_for_axpy))

    def axpy(self,u,v):
        '''axpy Matrix-vector mutiplication :math:`v\mapsto v+Au`

        :arg u: Vector to multiply
        :arg v: Resulting vector
        '''
        assert(u.function_space() == self._fs_col)
        assert(v.function_space() == self._fs_row)
        param_dict = {'A_'+x:y for (x,y) in self._param_dict.iteritems()}
        if (self._alpha == self._beta) and self._use_blas_for_axpy:
            kernel_code = '''void axpy(double **A,
                                       double **u,
                                       double **v) {
              cblas_dgbmv(CblasColMajor,CblasTrans,
                          %(A_n_col)d,%(A_n_row)d,
                          %(A_gamma_m)d,%(A_gamma_p)d,
                          1.0,A[0],%(A_bandwidth)d,
                          u[0],1,1.0,v[0],1);
            }'''
        else:
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
        kernel = op2.Kernel(kernel_code % param_dict,'axpy',cpp=True,
                            headers=['#include "cblas.h"'],
                            libs=self._libs)
        with timed_region('bandedmatrix axpy'):
            op2.par_loop(kernel,
                         self._hostmesh.cell_set,
                         self._data(op2.READ,self._Vcell.cell_node_map()),
                         u.dat(op2.READ,u.cell_node_map()),
                         v.dat(op2.INC,v.cell_node_map()))

    def fraction_of_nnz(self,tolerance=1.E-12):
        '''Return the fraction of non-zero entries 

            Count the number of entries whose absolute value is larger than a 
            given tolerance.

            :arg tolerance: Tolerance to determine if an entry is deemed to be zero
        '''
        kernel_code = '''void count_zeros(double **A,
                                          double *ntotal,
                                          double *nnz) {
          for (int i=0;i<%(A_n_row)d*%(A_bandwidth)d;++i) {
            nnz[0] += (fabs(A[0][i]) > %(TOLERANCE)e);
          }
          ntotal[0] += %(A_n_row)d*%(A_bandwidth)d;
        }'''
        
        param_dict = {'A_'+x:y for (x,y) in self._param_dict.iteritems()}
        param_dict.update({'TOLERANCE':tolerance})
        kernel = op2.Kernel(kernel_code % param_dict, 'count_zeros',cpp=True)
        ntotal = op2.Global(1,data=0.0,dtype=float)
        nnz = op2.Global(1,data=0.0,dtype=float)
        op2.par_loop(kernel,
                     self._hostmesh.cell_set,
                     self._data(op2.READ,self._Vcell.cell_node_map()),
                     ntotal(op2.INC),nnz(op2.INC))
        return float(nnz.data[0])/float(ntotal.data[0])

    def dense(self):
        '''Convert to a dense matrix format.

        Return the matrix in a dense format, i.e. a n_col x n_row matrix in every 
        vertical column. This should mainly be used for debugging since the matrix
        is sparse and the routine will return huge dense matrices for larger meshes.
        '''
        A_dense = Function(self._Vcell,
                           val=op2.Dat(self._Vcell.node_set**(self._n_row,self._n_col)))
        kernel_code = '''void convert_to_dense(double **A,
                                               double **B) {
          for (int i=0;i<%(A_n_row)d;++i) {
            int j_m = (int) ceil((%(A_alpha)d*i-%(A_gamma_p)d)/(1.0*%(A_beta)f));
            int j_p = (int) floor((%(A_alpha)d*i+%(A_gamma_m)d)/(1.0*%(A_beta)f));
            for (int j=std::max(0,j_m);j<std::min(%(A_n_col)d,j_p+1);++j) {
               B[0][%(A_n_col)d*i+j] = A[0][%(A_bandwidth)d*i+(j-j_m)];
            }
          }
        }'''
        param_dict = {'A_'+x:y for (x,y) in self._param_dict.iteritems()}
        kernel = op2.Kernel(kernel_code % param_dict, 'convert_to_dense',cpp=True)
        with timed_region('bandedmatrix dense'):
            op2.par_loop(kernel,
                         self._hostmesh.cell_set,
                         self._data(op2.READ,self._Vcell.cell_node_map()),
                         A_dense.dat(op2.WRITE,self._Vcell.cell_node_map()))
        return A_dense

    def transpose(self,result=None):
        '''Calculate transpose of matrix :math:`B=A^T`.

        Transpose the matrix and return the result. If the parameter result is passed,
        this matrix is used, otherwise new storage space is allocated.

        :arg result: resulting matrix :math:`B`
        '''
        if (result != None):
            assert(result._n_row == self._n_col)
            assert(result._n_col == self._n_row)
            assert(result.alpha == self.beta)
            assert(result.beta == self.alpha)
            assert(result.gamma_p == self.gamma_m)
            assert(result.gamma_m == self.gamma_p)
        else:
            result = BandedMatrix(self._fs_col,self._fs_row,
                                  alpha=self.beta,beta=self.alpha,
                                  gamma_m=self.gamma_p,gamma_p=self.gamma_m)
        param_dict = {}
        for label, matrix in zip(('A','B'),(self,result)):
            param_dict.update({label+'_'+x:y for (x,y) in matrix._param_dict.iteritems()})
        kernel_code = '''void transpose(double **A,
                                        double **B) {
          for (int i=0;i<%(A_n_row)d;++i) {
            int j_m = (int) ceil((%(A_alpha)d*i-%(A_gamma_p)d)/(1.0*%(A_beta)f));
            int j_p = (int) floor((%(A_alpha)d*i+%(A_gamma_m)d)/(1.0*%(A_beta)f));
            for (int j=std::max(0,j_m);j<std::min(%(A_n_col)d,j_p+1);++j) {
               int i_m = (int) ceil((%(B_alpha)d*j-%(B_gamma_p)d)/(1.0*%(B_beta)f));
               B[0][%(B_bandwidth)d*j+(i-i_m)] = A[0][%(A_bandwidth)d*i+(j-j_m)];
            }
          }
        }'''
        kernel = op2.Kernel(kernel_code % param_dict, 'transpose',cpp=True)
        with timed_region('bandedmatrix transpose'):
            op2.par_loop(kernel,
                         self._hostmesh.cell_set,
                         self._data(op2.READ,self._Vcell.cell_node_map()),
                         result._data(op2.WRITE,self._Vcell.cell_node_map()))
        return result

    def matmul(self,other,result=None):
        '''Calculate matrix product :math:`C=AB`.

        Multiply this matrix by the banded matrix :math:`B` from the right and store the
        result in the matrix :math:`C`, which is returned on exit.
        If result is None, allocate a new matrix, otherwise 
        write data to already allocated matrix ``result``. 

        :arg other: matrix :math:`B` to multiply with
        :arg result: resulting matrix :math:`C`
        '''
        # Check that matrices can be multiplied
        assert (self._n_col == other._n_row)
        if (result != None):
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
        with timed_region('bandedmatrix matmul'):
            op2.par_loop(kernel,
                         self._hostmesh.cell_set,
                         self._data(op2.READ,self._Vcell.cell_node_map()),
                         other._data(op2.READ,self._Vcell.cell_node_map()),
                         result._data(op2.WRITE,self._Vcell.cell_node_map()))
        return result

    def transpose_matmul(self,other,result=None):
        '''Calculate matrix product :math:`C=A^TB`.

        Multiply the transpose of this matrix by the banded matrix :math:`B` 
        from the right and store the
        result in the matrix :math:`C`, which is returned on exit.
        If result is None, allocate a new matrix, otherwise 
        write data to already allocated matrix ``result``. 

        :arg other: matrix :math:`B` to multiply with
        :arg result: resulting matrix :math:`C`
        '''
        # Check that matrices can be multiplied
        assert (self._n_row == other._n_row)
        if (result != None):
            assert(result._n_row == self._n_col)
            assert(result._n_col == other._n_col)
        else:
            alpha = self.beta * other.alpha
            beta = self.alpha * other.beta
            gamma_m = other.alpha * self.gamma_p + self.alpha*other.gamma_m
            gamma_p = other.alpha * self.gamma_m + self.alpha*other.gamma_p
            result = BandedMatrix(self._fs_col,other._fs_col,
                                  alpha=alpha,beta=beta,
                                  gamma_m=gamma_m,gamma_p=gamma_p)

        param_dict = {}
        for label, matrix in zip(('A','B','C'),(self,other,result)):
            param_dict.update({label+'_'+x:y for (x,y) in matrix._param_dict.iteritems()})
        kernel_code = '''void transpose_matmul(double **A,
                                               double **B,
                                               double **C) {
          for (int i=0;i<%(C_n_row)d;++i) {
            int j_m = (int) ceil((%(C_alpha)d*i-%(C_gamma_p)d)/(1.0*%(C_beta)f));
            int j_p = (int) floor((%(C_alpha)d*i+%(C_gamma_m)d)/(1.0*%(C_beta)f));
            int k_m = (int) ceil((%(A_beta)d*i-%(A_gamma_m)d)/(1.0*%(A_alpha)f));
            int k_p = (int) floor((%(A_beta)d*i+%(A_gamma_p)d)/(1.0*%(A_alpha)f));
            for (int j=std::max(0,j_m);j<std::min(%(C_n_col)d,j_p+1);++j) {
              double s = 0.0;
              for (int k=std::max(0,k_m);k<std::min(%(A_n_row)d,k_p+1);++k) {
                if ( (ceil((%(B_alpha)d*k-%(B_gamma_p)d)/%(B_beta)f) <= j) &&
                     (j <= floor((%(B_alpha)d*k+%(B_gamma_m)d)/(1.0*%(B_beta)f))) &&
                     (ceil((%(A_alpha)d*k-%(A_gamma_p)d)/%(B_beta)f) <= i) &&
                     (i <= floor((%(A_alpha)d*k+%(A_gamma_m)d)/(1.0*%(A_beta)f))) ) {
                  int i_m_A = (int) ceil((%(A_alpha)d*k-%(A_gamma_p)d)/(1.0*%(A_beta)f));
                  int j_m_B = (int) ceil((%(B_alpha)d*k-%(B_gamma_p)d)/(1.0*%(B_beta)f));
                  s += A[0][%(A_bandwidth)d*k+(i-i_m_A)]
                     * B[0][%(B_bandwidth)d*k+(j-j_m_B)];
                }
              }
              C[0][%(C_bandwidth)d*i+(j-j_m)] = s;
            }
          }
        }'''
        kernel = op2.Kernel(kernel_code % param_dict, 'transpose_matmul',cpp=True)
        with timed_region('bandedmatrix transpose_matmul'):
            op2.par_loop(kernel,
                         self._hostmesh.cell_set,
                         self._data(op2.READ,self._Vcell.cell_node_map()),
                         other._data(op2.READ,self._Vcell.cell_node_map()),
                         result._data(op2.WRITE,self._Vcell.cell_node_map()))
        return result

    def scale(self,omega=1.0):
        '''Scale matrix :math:`A` in place by a given factor.

        Scale this matrix :math:`A\mapsto \omega A`

        :arg omega: factor :math:`\omega` to scale by
        '''
        self._data *= omega

    def matadd(self,other,omega=1.0,result=None):
        '''Calculate matrix sum :math:`C=A+\omega B`.

        Add the banded matrix :math:`\omega B` to this matrix and store the result in the 
        banded matrix :math:`C`, which is returned on exit.
        If result is None, allocate a new matrix, otherwise write 
        data to already allocated matrix ``result``.

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
        if (result != None):
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
        kernel_code = '''void matadd(double *c_omega,
                                     double **A,
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
              C[0][%(C_bandwidth)d*i+(j-j_m_C)] += c_omega[0]*B[0][%(B_bandwidth)d*i+(j-j_m_B)];
            }
          }
        }'''
        kernel = op2.Kernel(kernel_code % param_dict, 'matadd',cpp=True)
        c_omega = Constant(omega)
        with timed_region('bandedmatrix matadd'):
            op2.par_loop(kernel,
                         self._hostmesh.cell_set,
                         c_omega.dat(op2.READ),
                         self._data(op2.READ,self._Vcell.cell_node_map()),
                         other._data(op2.READ,self._Vcell.cell_node_map()),
                         result._data(op2.WRITE,self._Vcell.cell_node_map()))
        return result

    def solve(self,u):
        '''Solve in place.

        Solve the equation :math:`Au=b` and replace function by result.

        :arg u: RHS b and resulting function u
        ''' 
        # Check whether data has changed since last LU decomposition
        if not (self._lu_version == self._data._version):
            self._lu_decompose()
        return self._lu_solve(u)
            
    def _lu_decompose(self):
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
                            headers=['#include "lapacke.h"'],
                            libs=self._libs)
        with timed_region('bandedmatrix lu_decompose'):
            op2.par_loop(kernel,
                         self._hostmesh.cell_set,
                         self._data(op2.READ,self._Vcell.cell_node_map()),
                         self._lu(op2.WRITE,self._Vcell.cell_node_map()),
                         self._ipiv(op2.WRITE,self._Vcell.cell_node_map()))
        self._lu_version = self._data._version

    def _lu_solve(self,u):
        '''In-place LU solve for a field u.
        
        :arg u: Function to be solved for.
        '''
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
                            headers=['#include "lapacke.h"'],
                            libs=self._libs)
        with timed_region('bandedmatrix lu_solve'):
            op2.par_loop(kernel,
                         self._hostmesh.cell_set,
                         self._lu(op2.WRITE,self._Vcell.cell_node_map()),
                         self._ipiv(op2.WRITE,self._Vcell.cell_node_map()),
                         u.dat(op2.RW,u.cell_node_map()),
                         name='bandedmatrix lu_solve['+self._label+']')
        return u

    def diagonal(self):
        '''Extract diagonal entries.

        For a banded matrix with alpha=beta, create a new banded matrix
        which contains the diagonal entries only.
        '''
        assert(self._alpha == self._beta)
        assert(self._gamma_m>=0)
        assert(self._gamma_p>=0)
        result = BandedMatrix(self._fs_row,self._fs_col,
                              alpha=1,beta=1,gamma_m=0,gamma_p=0)
        kernel_code = '''void diagonal(double **A,
                                       double **Adiag) {
          for (int i=0;i<%(A_n_row)d;++i) {
            int j_m = (int) ceil((%(A_alpha)d*i-%(A_gamma_p)d)/%(A_beta)f);
            Adiag[0][i] = A[0][%(A_bandwidth)d*i+(i-j_m)];
          }
        }'''
        param_dict = {'A_'+x:y for (x,y) in self._param_dict.iteritems()}
        kernel = op2.Kernel(kernel_code % param_dict, 'diagonal')
        with timed_region('bandedmatrix diagonal'):
            op2.par_loop(kernel,
                         self._hostmesh.cell_set,
                         self._data(op2.READ,self._Vcell.cell_node_map()),
                         result._data(op2.WRITE,self._Vcell.cell_node_map()))
        return result

    def inv_diagonal(self):
        '''Extract inverse diagonal entries.

        For a banded matrix with alpha=beta, create a new banded matrix
        which contains the inverse diagonal entries only.
        '''
        assert(self._alpha == self._beta)
        assert(self._gamma_m>=0)
        assert(self._gamma_p>=0)
        result = BandedMatrix(self._fs_row,self._fs_col,
                              alpha=1,beta=1,gamma_m=0,gamma_p=0)
        kernel_code = '''void diagonal(double **A,
                                       double **Adiag) {
          for (int i=0;i<%(A_n_row)d;++i) {
            int j_m = (int) ceil((%(A_alpha)d*i-%(A_gamma_p)d)/%(A_beta)f);
            Adiag[0][i] = 1.0/A[0][%(A_bandwidth)d*i+(i-j_m)];
          }
        }'''
        param_dict = {'A_'+x:y for (x,y) in self._param_dict.iteritems()}
        kernel = op2.Kernel(kernel_code % param_dict, 'diagonal')
        with timed_region('bandedmatrix inv_diagonal'):
            op2.par_loop(kernel,
                         self._hostmesh.cell_set,
                         self._data(op2.READ,self._Vcell.cell_node_map()),
                         result._data(op2.WRITE,self._Vcell.cell_node_map()))
        return result        

    def spai(self,gamma=None):
        '''Calculate Sparse approximate inverse based on a fixed
        sparsity pattern.

        The matrix has to be square and symmetrically banded 
        (:math:`\gamma=\gamma_+=\gamma_-`, :math:`\\alpha=\\beta=1`), i.e. the sparsity 
        pattern is

        .. math::
            -\gamma \le i-j \le \gamma

        If no argument is passed, the sparsity pattern of the resulting matrix
        is the same as that of the current matrix, otherwise it is the one
        obtained by using :math:`\gamma \mapsto \gamma^{(M)}`.

        For the description of the SPAI preconditioner, see Grote, Marcus J., and 
        Thomas Huckle:
        "Parallel preconditioning with sparse approximate inverses."
        SIAM Journal on Scientific Computing 18.3 (1997): 838-853.

        Documentation of LAPACK DGELS routine see e.g. 
        http://www.netlib.no/netlib/lapack/double/dgels.f
        
        :arg gamma: Parameter :math:`\gamma^{(M)}` of the resulting matrix.
        '''
        assert self.is_sparsity_symmetric
        gamma_M = gamma
        if (gamma == None):
            gamma_M = self.gamma_m
        result = BandedMatrix(self._fs_row,self._fs_col,
                              alpha=self.alpha,beta=self.beta,
                              gamma_m=gamma_M,gamma_p=gamma_M)

        kernel_code = '''void spai(double **A,
                                   double **M) {
          double Ahat[%(A_n_row)d*%(M_bandwidth)d];
          double mhat[%(A_n_row)d];
          for(int k=0;k<%(A_n_row)d;++k) {
            int j_m = std::max(0,k-%(M_gamma_p)d);
            int j_p = std::min(%(A_n_row)d-1,k+%(M_gamma_m)d);
            int i_m = std::max(0,j_m-%(A_gamma_p)d);
            int i_p = std::min(%(A_n_row)d-1,j_p+%(A_gamma_m)d);
            // Number of rows and columns in local matrix Ahat
            int n_row_hat = i_p-i_m+1;
            int n_col_hat = j_p-j_m+1;
            for (int i_hat=0;i_hat<n_row_hat;++i_hat) {
              int i = i_hat+i_m;
              for (int j_hat=0;j_hat<n_col_hat;++j_hat) {
                int j = j_hat+j_m;
                double tmp = 0.0;
                if ( ((i-%(A_gamma_p)d)<=j) && (j<=(i+%(A_gamma_m)d)) ) {
                  tmp = A[0][%(A_bandwidth)d*i+(j-(i-%(A_gamma_p)d))];
                }
                Ahat[i_hat+%(A_n_row)d*j_hat] = tmp;
              }
            }
            for (int i_hat=0;i_hat<n_row_hat;++i_hat) {
              mhat[i_hat] = ((i_hat+i_m)==k);
            }
            LAPACKE_dgels(LAPACK_COL_MAJOR,'N',n_row_hat,n_col_hat,1,
                          Ahat,%(A_n_row)d,
                          mhat,%(A_n_row)d);
            for (int j_hat=0;j_hat<n_col_hat;++j_hat) {
              int j = j_hat+j_m;
              M[0][%(M_bandwidth)d*j + (k-(j-%(M_gamma_p)d))] = mhat[j_hat];
            }
          }
        }'''
        param_dict = {}
        for label, matrix in zip(('A','M'),(self,result)):
            param_dict.update({label+'_'+x:y for (x,y) in matrix._param_dict.iteritems()})
        kernel = op2.Kernel(kernel_code % param_dict, 'spai',
                            cpp=True,
                            headers=['#include "lapacke.h"'],
                            libs=self._libs)
        with timed_region('bandedmatrix spai'):
            op2.par_loop(kernel,
                         self._hostmesh.cell_set,
                         self._data(op2.READ,self._Vcell.cell_node_map()),
                         result._data(op2.WRITE,self._Vcell.cell_node_map()))
        return result
        
