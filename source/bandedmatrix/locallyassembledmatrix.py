import fractions
import math
import numpy as np
import socket
from firedrake import *
from firedrake.ffc_interface import compile_form

class LocallyAssembledMatrix(object):
    '''Explicit representation of locally assembled matrices.

        This class can be used to assemble a UFL form into a local matrix on each
        cell. Note that this is different from a globally assembled matrix.
        The UFL form is assumed to map from the column function space to the row function
        space, and the local matrix on each grid cell has the size
        :math:`n_{row}\\times n_{col}`. This is represented as a matrix-valued DG0 field.

        :arg fs_row: Row function space
        :arg fs_col: Column function space
        :arg ufl_form: UFL form to assemble from
    '''
    def __init__(self,fs_row,fs_col,ufl_form=None):
        self._fs_row = fs_row
        self._fs_col = fs_col
        self._ndof_row = len(self._fs_row.cell_node_map().values[0])
        self._ndof_col = len(self._fs_col.cell_node_map().values[0])
        self._mesh = self._fs_row.mesh()
        self._Vcell = FunctionSpace(self._mesh,'DG',0)
        self._data = op2.Dat(self._Vcell.node_set**(self._ndof_row*self._ndof_col))

        self._param_dict={'n_row':self._ndof_row,
                          'n_col':self._ndof_col}

        if (ufl_form != None):
            self.assemble_ufl_form(ufl_form)
        else:
            self._data.zero()
        hostname = socket.gethostname()
        self._libs = []
        if ( ('Eikes-MacBook-Pro.local' in hostname) or \
             ('eduroam.bath.ac.uk' in hostname) or \
             ('Eikes-MBP' in hostname) ):
            self._libs = ['lapack','lapacke','cblas','blas']

    def assemble_ufl_form(self,ufl_form):
        '''Assemble the local matrix from a UFL form.

        In each cell, build the local matrix stencil associated
        with the UFL form.

        :arg ufl_form: UFL form to assemble
        '''
        param_coffee_old = parameters["coffee"]["O2"]
        parameters["coffee"]["O2"] = False
        compiled_form = compile_form(ufl_form, 'ufl_form')[0]
        kernel = compiled_form[6]
        coords = compiled_form[3]
        coefficients = compiled_form[4]
        arguments = ufl_form.arguments()
        assert len(arguments) == 2, 'Not a bilinear form'
        nrow = arguments[0].cell_node_map().arity
        assert (nrow == self._ndof_row), 'Dimension of row space not correct'
        ncol = arguments[1].cell_node_map().arity
        assert (ncol == self._ndof_col), 'Dimension of column space not correct'
        args = [self._data(op2.INC, self._Vcell.cell_node_map()[op2.i[0]]), 
                coords.dat(op2.READ, coords.cell_node_map(), flatten=True)]
        for c in coefficients:
            args.append(c.dat(op2.READ, c.cell_node_map(), flatten=True))
        op2.par_loop(kernel,self._mesh.cell_set, *args)
        parameters["coffee"]["O2"] = param_coffee_old

    def ax(self,u,v=None):
        '''Matrix-vector multiplication :math:`v=\\alpha Au`

            :arg u: Function to multiply
            :arg v: Function to add to
        '''
        if (v != None):
            assert(v.function_space() == self._fs_row)
        else:
            v = Function(self._fs_row)
        assert(u.function_space() == self._fs_col)
        param_dict = {'A_'+x:y for (x,y) in self._param_dict.iteritems()}
        kernel_code = '''void ax(double **A,
                                 double **u,
                                 double **v) {
          // Loop over matrix rows
          for (int i=0;i<%(A_n_row)d;++i) {
            double s = 0;
            // Loop over columns
            for (int j=0;j<%(A_n_col)d;++j) {
               s += A[0][%(A_n_col)d*i+j] * u[j][0];
            }
            v[i][0] += s;
          }
        }
        '''
        v.dat.zero()
        kernel = op2.Kernel(kernel_code % param_dict,'ax')
        op2.par_loop(kernel,
                     self._mesh.cell_set,
                     self._data(op2.READ,self._Vcell.cell_node_map()),
                     u.dat(op2.READ,u.cell_node_map()),
                     v.dat(op2.INC,v.cell_node_map()))
        return v

    def transpose(self,result=None):
        '''Locally transpose matrix :math:`B=A^T`

            :arg result: Resulting matrix :`B=A^T`
        '''
        if (result != None):
            assert(result._ndof_row == self._ndof_col)
            assert(result._ndof_col == other._ndof_row)
        else:
            result = LocallyAssembledMatrix(self._fs_col,self._fs_row)
        param_dict = {}
        for label, matrix in zip(('A','B'),(self,result)):
            param_dict.update({label+'_'+x:y for (x,y) in matrix._param_dict.iteritems()})
        kernel_code = '''void transpose(double **A,
                                        double **B) {
          for(int i=0;i<%(A_n_row)d;++i) {
            for(int j=0;j<%(A_n_col)d;++j) {
              B[0][%(B_n_col)d*j+i] = A[0][%(A_n_col)d*i+j];
            }
          }
        }'''
        kernel = op2.Kernel(kernel_code % param_dict, 'transpose')
        op2.par_loop(kernel,
                     self._mesh.cell_set,
                     self._data(op2.READ,self._Vcell.cell_node_map()),
                     result._data(op2.WRITE,self._Vcell.cell_node_map()))

        return result
    def matmul(self,other,result=None):
        '''Locally multiply matrices :math:`C=AB`

            :arg other: Matrix :math:`B`
            :arg result: Resulting matrix :`C`
        '''
        assert (self._ndof_col == other._ndof_row)
        if (result != None):
            assert(result._ndof_row == self._ndof_row)
            assert(result._ndof_col == other._ndof_col)
        else:
            result = LocallyAssembledMatrix(self._fs_row,other._fs_col)
        param_dict = {}
        for label, matrix in zip(('A','B','C'),(self,other,result)):
            param_dict.update({label+'_'+x:y for (x,y) in matrix._param_dict.iteritems()})
        kernel_code = '''void matmul(double **A,
                                double **B,
                                double **C) {
          for(int i=0;i<%(C_n_row)d;++i) {
            for(int j=0;j<%(C_n_col)d;++j) {
              double s = 0.0;
              for (int k=0;k<%(A_n_col)d;++k) {
                s += A[0][i*%(A_n_col)d+k] * B[0][k*%(B_n_col)d+j];
              }
              C[0][%(C_n_col)d*i+j] = s;
            }
          }
        }'''
        kernel = op2.Kernel(kernel_code % param_dict, 'matmul')
        op2.par_loop(kernel,
                     self._mesh.cell_set,
                     self._data(op2.READ,self._Vcell.cell_node_map()),
                     other._data(op2.READ,self._Vcell.cell_node_map()),
                     result._data(op2.WRITE,self._Vcell.cell_node_map()))

        return result

    def matadd(self,other,omega=1.0,result=None):
        '''Locally add matrices :math:`C=A+\omega B`

            :arg other: Matrix :math:`B`
            :arg alpha: Scaling factor :math:`\omega`
            :arg result: Resulting matrix :`C`
        '''
        assert (self._ndof_row == other._ndof_row)
        assert (self._ndof_col == other._ndof_col)
        if (result != None):
            assert(result._ndof_row == self._ndof_row)
            assert(result._ndof_col == self._ndof_col)
        else:
            result = LocallyAssembledMatrix(self._fs_row,other._fs_col)
        result._data = self._data + omega*other._data
        return result

    def scale(self,omega=1.0):
        '''Multiply by a factor in place
    
            :arg omega: Scaling factor    
        '''
        self._data *= omega

    def inverse(self,result=None): 
        '''Calculate local inverse :math:`C=A^{-1}`

            Note that this only works for square matrices.

            :arg result: Resulting matrix :math:`C`
        '''
        assert (self._ndof_row == self._ndof_col)
        if (result != None):
            assert(result._ndof_row == self._ndof_row)
            assert(result._ndof_col == self._ndof_col)
        else:
            result = LocallyAssembledMatrix(self._fs_row,self._fs_col)
        param_dict = {}
        for label, matrix in zip(('A','B'),(self,result)):
            param_dict.update({label+'_'+x:y for (x,y) in matrix._param_dict.iteritems()})
        kernel_code = '''void inverse(double **A,
                                      double **B) {
            double rhs[%(A_n_row)d*%(A_n_col)d];
            for (int i=0;i<%(A_n_row)d*%(A_n_col)d;++i) {
              rhs[i] = 0.0;
              B[0][i] = A[0][i];
            }
            for (int i=0;i<%(A_n_row)d;++i) rhs[(%(A_n_row)d+1)*i]= 1.0;
            LAPACKE_dgels(LAPACK_COL_MAJOR,'N',%(A_n_row)d,%(A_n_row)d,%(A_n_row)d,
                          B[0],%(A_n_row)d,
                          rhs,%(A_n_row)d);
            for (int i=0;i<%(A_n_row)d*%(A_n_col)d;++i) {
              B[0][i] = rhs[i];
            }

        }'''
        kernel = op2.Kernel(kernel_code % param_dict,
                            'inverse',
                            cpp=True,
                            headers=['#include "lapacke.h"'],
                            libs=self._libs)
        op2.par_loop(kernel,
                     self._mesh.cell_set,
                     self._data(op2.READ,self._Vcell.cell_node_map()),
                     result._data(op2.WRITE,self._Vcell.cell_node_map()))
        return result

