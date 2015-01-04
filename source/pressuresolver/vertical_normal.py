from firedrake import *
import sys, petsc4py
import numpy as np

petsc4py.init(sys.argv)

from petsc4py import PETSc


class VerticalNormal(object):
    '''Class for constructing a vertical normal field on a given extruded mesh`.

    This class can be used to construct a constant vertical normal field 
    :math:`\hat{z}`, which is, for example, used to express terms of the form
    :math:`\hat{\\vec{z}}\cdot\\vec{u}`.

    :arg mesh: extruded mesh
    '''
    def __init__(self,mesh):
        self._mesh = mesh
        self._build_zhat()

    @property
    def zhat(self):
        '''Return field :math:`\hat{\\vec{z}}`'''
        return self._zhat

    def _build_zhat(self):
        '''Calculate vector field representing the unit normal in each cell.
        '''
        normal_fs = VectorFunctionSpace(self._mesh,'DG',0)
        self._zhat = Function(normal_fs)
        host_mesh = self._mesh._old_mesh
        # Topological dimension of the underlying grid, can be 1 or 2
        host_dimension = host_mesh._ufl_cell.topological_dimension()
        if (host_dimension == 1):
            kernel_code = '''build_normal(double **base_coords,
                                          double **normals) {
              const int ndim=2;
              const int nvert=2;
              double dx[ndim];
              double xavg[ndim];
              double n[ndim];
              // Calculate vector between the two points
              for (int i=0;i<ndim;++i) { // Loop over dimensions
                dx[i] = base_coords[1][i] - base_coords[0][i];
              }
              // Rotate by 90 degrees to get normal
              n[0] = -dx[1];
              n[1] = +dx[0];
              // Calculate vector at centre of edge
              for (int i=0;i<ndim;++i) { // Loop over dimensions
                xavg[i] = 0.0;
                for (int j=0;j<nvert;++j) { // Loop over vertices
                  xavg[i] += base_coords[j][i];
                }
              }
              // Calculate ||n|| and n.x_avg
              double nrm = 0.0;
              double n_dot_xavg = 0.0;
              for (int i=0;i<ndim;++i) {
                nrm += n[i]*n[i];
                n_dot_xavg += n[i]*xavg[i];
              }
              nrm = sqrt(nrm);
              // Orient correctly
              nrm *= (n_dot_xavg<0?-1:+1);
              for (int i=0;i<ndim;++i) {
                normals[0][i] = n[i]/nrm;
              }
            }'''
        else:
            kernel_code = '''build_normal(double **base_coords,
                                          double **normals) {
              const int ndim=3;
              const int nvert=3;
              double dx[2][ndim];
              double xavg[ndim];
              double n[ndim];
              // Calculate vector between the two points
              for (int i=0;i<ndim;++i) { // Loop over dimensions
                for (int j=0;j<2;++j) {
                  dx[j][i] = base_coords[1+j][i] - base_coords[0][i];
                }
              }
              // Calculate normal
              for (int i=0;i<ndim;++i) {
                n[i] = dx[0][(1+i)%3]*dx[1][(2+i)%3] 
                     - dx[0][(2+i)%3]*dx[1][(1+i)%3];
              }
              // Calculate vector at centre of edge
              for (int i=0;i<ndim;++i) { // Loop over dimensions
                xavg[i] = 0.0;
                for (int j=0;j<nvert;++j) { // Loop over vertices
                  xavg[i] += base_coords[j][i];
                }
              }
              // Calculate ||n|| and n.x_avg
              double nrm = 0.0;
              double n_dot_xavg = 0.0;
              for (int i=0;i<ndim;++i) {
                nrm += n[i]*n[i];
                n_dot_xavg += n[i]*xavg[i];
              }
              nrm = sqrt(nrm);
              // Orient correctly
              nrm *= (n_dot_xavg<0?-1:+1);
              for (int i=0;i<ndim;++i) {
                normals[0][i] = n[i]/nrm;
              }
            }'''
        kernel = op2.Kernel(kernel_code,'build_normal')
        base_coords = host_mesh.coordinates
        op2.par_loop(kernel,self._zhat.cell_set,
                     base_coords.dat(op2.READ,base_coords.cell_node_map()),
                     self._zhat.dat(op2.WRITE,self._zhat.cell_node_map()))
