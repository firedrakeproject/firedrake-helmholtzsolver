from firedrake import *
from pressuresolver.vertical_normal import *
import numpy as np
import pytest
import os
parameters["coffee"]["O2"] = False

@pytest.fixture
def mesh_2d():
    '''Create 1+1 dimensional mesh by extruding a circle.'''
    D = 1.0
    nlayers=4
    ncells=16

    host_mesh = CircleManifoldMesh(ncells)
    mesh_2d = ExtrudedMesh(host_mesh,
                           layers=nlayers,
                           extrusion_type='radial',
                           layer_height=D/nlayers)

    return mesh_2d

@pytest.fixture
def mesh_3d():
    '''Create 2+1 dimensional mesh by extruding an icosahedral mesh.'''
    D = 1.0
    nlayers=4
    refinement_level=1

    host_mesh = UnitIcosahedralSphereMesh(refinement_level)
    mesh_3d = ExtrudedMesh(host_mesh,
                           layers=nlayers,
                           extrusion_type='radial',
                           layer_height=D/nlayers)

    return mesh_3d

def test_zhat_2d_plot(mesh_2d):
    '''Create a vertical normal field on a 1+1 dimensional extruded grid and
    save to disk.

    :arg mesh_2d: 1+1 dimensional mesh 
    '''
    vertical_normal = VerticalNormal(mesh_2d)
    zhat = vertical_normal.zhat
    DFile = File(os.path.join('output','zhat_2d.pvd'))
    DFile << zhat 
    assert True

def test_zhat_2d_length(mesh_2d):
    '''Create a vertical normal field on a 1+1 dimensional extruded grid and
    check that the vectors have the correct length.

    :arg mesh_2d: 1+1 dimensional mesh 
    '''
    vertical_normal = VerticalNormal(mesh_2d)
    zhat = vertical_normal.zhat
    nrm = [x**2+y**2 for x,y in zhat.dat.data]
    assert np.allclose(nrm, 1.0)

def test_zhat_2d_orthogonality_and_orientation(mesh_2d):
    '''Create a vertical normal field on a 1+1 dimensional extruded grid and
    check that the vectors are orthogonal to the bottom facet.

    :arg mesh_2d: 1+1 dimensional mesh 
    '''
    vertical_normal = VerticalNormal(mesh_2d)
    zhat = vertical_normal.zhat
    host_mesh = mesh_2d._old_mesh
    base_coords = host_mesh.coordinates
    fs_ortho = FunctionSpace(mesh_2d,'DG',0)
    is_orthogonal = Function(fs_ortho,dtype=int)
    kernel_code = '''void check_orthogonality(double **base_coords,
                                              double **normals,
                                              int **is_orthogonal) {
                const int ndim=2;
                const int nvert=2;
                double dx[ndim];
                double xavg[ndim];
                // Calculate vector dx between the two points
                // and facet centre vector xavg
                for (int i=0;i<ndim;++i) { // Loop over dimensions
                  dx[i] = base_coords[1][i] - base_coords[0][i];
                  xavg[i] = 0.0;
                  for (int j=0;j<nvert;++j) { // Loop over vertices
                    xavg[i] += base_coords[j][i];
                  }
                }
                double n_dot_xavg = 0.0;
                double n_dot_dx = 0.0;
                for (int i=0;i<ndim;++i) {
                  n_dot_xavg += normals[0][i]*xavg[i];
                  n_dot_dx += normals[0][i]*dx[i];
                }
                is_orthogonal[0][0] = (n_dot_xavg > 0.0) && (abs(n_dot_dx) < 1.E-9);
              }'''
    kernel = op2.Kernel(kernel_code,'check_orthogonality')
    op2.par_loop(kernel,zhat.cell_set,
                 base_coords.dat(op2.READ,base_coords.cell_node_map()),
                 zhat.dat(op2.READ,zhat.cell_node_map()),
                 is_orthogonal.dat(op2.WRITE,is_orthogonal.cell_node_map()))
    assert np.allclose(is_orthogonal.dat.data, 1.0)


def test_zhat_3d_plot(mesh_3d):
    '''Create a vertical normal field on a 2+1 dimensional extruded grid and
    save to disk.

    :arg mesh_3d: 2+1 dimensional mesh 
    '''
    vertical_normal = VerticalNormal(mesh_3d)
    zhat = vertical_normal.zhat
    DFile = File(os.path.join('output','zhat_3d.pvd'))
    DFile << zhat 
    assert True

def test_zhat_3d_length(mesh_3d):
    '''Create a vertical normal field on a 2+1 dimensional extruded grid and
    check that the vectors have the correct length.

    :arg mesh_3d: 2+1 dimensional mesh 
    '''
    vertical_normal = VerticalNormal(mesh_3d)
    zhat = vertical_normal.zhat
    nrm = [x**2+y**2+z**2 for x,y,z in zhat.dat.data]
    assert np.allclose(nrm, 1.0)

def test_zhat_3d_orthogonality_and_orientation(mesh_3d):
    '''Create a vertical normal field on a 2+1 dimensional extruded grid and
    check that the vectors are orthogonal to the bottom facet.

    :arg mesh_2d: 2+1 dimensional mesh 
    '''
    vertical_normal = VerticalNormal(mesh_3d)
    zhat = vertical_normal.zhat
    host_mesh = mesh_3d._old_mesh
    base_coords = host_mesh.coordinates
    fs_ortho = FunctionSpace(mesh_3d,'DG',0)
    is_orthogonal = Function(fs_ortho,dtype=int)
    kernel_code = '''void check_orthogonality(double **base_coords,
                                              double **normals,
                                              int **is_orthogonal) {
                const int ndim=3;
                const int nvert=3;
                double dx[2][ndim];
                double xavg[ndim];
                // Calculate vector dx between the two points
                // and facet centre vector xavg
                for (int i=0;i<ndim;++i) { // Loop over dimensions
                  for (int j=0;j<2;++j) {
                    dx[j][i] = base_coords[1+j][i] - base_coords[0][i];
                  }
                  xavg[i] = 0.0;
                  for (int j=0;j<nvert;++j) { // Loop over vertices
                    xavg[i] += base_coords[j][i];
                  }
                }
                double n_dot_xavg = 0.0;
                double n_dot_dx[2];
                for (int j=0;j<2;++j) {
                  n_dot_dx[j] = 0.0;
                }
                for (int i=0;i<ndim;++i) {
                  n_dot_xavg += normals[0][i]*xavg[i];
                  for (int j=0;j<2;++j) {
                    n_dot_dx[j] += normals[0][i]*dx[j][i];
                  }
                }
                is_orthogonal[0][0] = (n_dot_xavg > 0.0) 
                                   && (abs(n_dot_dx[0]) < 1.E-9)
                                   && (abs(n_dot_dx[1]) < 1.E-9);
              }'''
    kernel = op2.Kernel(kernel_code,'check_orthogonality')
    op2.par_loop(kernel,zhat.cell_set,
                 base_coords.dat(op2.READ,base_coords.cell_node_map()),
                 zhat.dat(op2.READ,zhat.cell_node_map()),
                 is_orthogonal.dat(op2.WRITE,is_orthogonal.cell_node_map()))
    assert np.allclose(is_orthogonal.dat.data, 1.0)

##############################################################
# M A I N
##############################################################
if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
