import os
import numpy as np
from collections import Counter
from firedrake import *

class LumpedMass(object):
    ''' Base class for velocity lumped mass matrix.

    :arg V_velocity: Velocity space, has to be a :math:`RT1` space.
    :arg ignore_lumping: For debugging, this can be set to true to use the
        full mass matrix in the :class:`multiply()` and :class:`divide()`
        methods.
    '''
    def __init__(self,V_velocity,
                      ignore_lumping=False):
        self.V_velocity = V_velocity
        self.w = TestFunction(self.V_velocity)
        self.mesh = self.V_velocity.mesh()
        self.ignore_lumping = ignore_lumping
        self.dx = self.V_velocity.mesh()._dx

    def test_kinetic_energy(self):
        '''Check how exactly the kinetic energy can be represented with
        lumped mass matrix.

        Calculate :math:`E_{kin}= \int \\vec{u}\cdot \\vec{u}\;dx` both with
        the full- and with the lumped mass matrix and compare
        '''
        u_SBR = Function(self.V_velocity)
        u_SBR.project(Expression(('0','-x[2]','x[1]')))
        energy_full = assemble(dot(u_SBR,u_SBR)*self.dx)
        Mu_SBR = Function(self.V_velocity)
        Mu_SBR.assign(u_SBR)
        self.multiply(Mu_SBR)
        energy_lumped = 0.0
        for (x,y) in zip(u_SBR.dat.data,Mu_SBR.dat.data):
            energy_lumped += x*y
        energy_exact = 4.*pi/3.
        energy_lumped *= 0.5
        energy_full *= 0.5
        print 'kinetic energy = '+('%18.12f' % energy_exact)+' [exact]'
        print '                 '+('%18.12f' % energy_full)+' [full mass matrix]'
        print '                 '+('%18.12f' % energy_lumped)+' [lumped mass matrix]'
            
    def multiply(self,u):
        '''Multiply by lumped mass matrix

        In-place multiply a velocity field by the lumped mass matrix

        :arg u: velocity field to multiply (will be modified in-place)
        '''
        if (self.ignore_lumping):
            psi = TestFunction(self.V_velocity)
            w = assemble(dot(self.w,u)*self.dx)
            u.assign(w)
        else:
            self._matmul(self.data,u)
            

    def divide(self,u):
        '''Divide by lumped mass matrix

        In-place divide a velocity field by the lumped mass matrix

        :arg u: velocity field to divide (will be modified in-place)
        '''
        if (self.ignore_lumping):
            psi = TestFunction(self.V_velocity)
            phi = TrialFunction(self.V_velocity)
            a_mass = assemble(dot(psi,phi)*self.dx)
            w = Function(self.V_velocity)
            solve(a_mass, w, u)
            u.assign(w)
        else:
            self._matmul(self.data_inv,u)


class LumpedMassRT1(LumpedMass):
    '''Lumped velocity mass matrix.
    
    This class constructs a diagonal lumped velocity mass matrix :math:`M_u^*` 
    in the :math:`RT1` space and provides methods for multiplying and dividing 
    :math:`RT1` functions by this lumped mass matrix. Internally the mass matrix
    is represented as a :math:`RT1` field.

    Currently, two methods for mass lumping are supported and can be chosen by 
    the parameter :class:`use_SBR`:

    * Lumped mass matrix is exact when acting on constant fields:

        .. math::
        
            M_u^* C = M_u C

        where C is constant.

    * On each edge e the lumped mass matrix is exact when acting on a solid
        body rotation field that has maximal flux through this edge.
        Mathematically this means that

        .. math::

            (M_u^*)_{ee} = \\frac{\sum_{i=1}^3 V^{(i)}_e U^{(i)}_e }{\sum_{i=1}^3 (U^{(i)}_e)^2}

        where :math:`U^{(i)}` is a solid body rotation field around coordinate
        axis :math:`i` and :math:`V^{(i)} = M_u U^{(i)}`

    :arg V_velocity: Velocity space, has to be a :math:`RT1` space.
    :arg ignore_lumping: For debugging, this can be set to true to use the
        full mass matrix in the :class:`multiply()` and :class:`divide()`
        methods.
    :arg use_SBR: Use mass lumping based on solid body rotation fields.
    '''
    def __init__(self,V_velocity,ignore_lumping=False,use_SBR=True):
        super(LumpedMassRT1,self).__init__(V_velocity,ignore_lumping)
        self.use_SBR = use_SBR
        if (self.use_SBR):
            w = TestFunction(self.V_velocity)
            self.data = Function(self.V_velocity)
            SBR_x = Function(self.V_velocity).project(Expression(('0','-x[2]','x[1]')))
            SBR_y = Function(self.V_velocity).project(Expression(('x[2]','0','-x[0]')))
            SBR_z = Function(self.V_velocity).project(Expression(('-x[1]','x[0]','0')))
            M_SBR_x = assemble(dot(w,SBR_x)*self.dx)
            M_SBR_y = assemble(dot(w,SBR_y)*self.dx)
            M_SBR_z = assemble(dot(w,SBR_z)*self.dx)
            kernel = '''*data = (  (*SBR_x)*(*M_SBR_x) 
                                 + (*SBR_y)*(*M_SBR_y)
                                 + (*SBR_z)*(*M_SBR_z) ) / 
                                (  (*SBR_x)*(*SBR_x) 
                                 + (*SBR_y)*(*SBR_y)
                                 + (*SBR_z)*(*SBR_z) );
            '''
            par_loop(kernel,direct,
                         {'data':(self.data,WRITE),
                          'SBR_x':(SBR_x,READ),
                          'SBR_y':(SBR_y,READ),
                          'SBR_z':(SBR_z,READ),
                          'M_SBR_x':(M_SBR_x,READ),
                          'M_SBR_y':(M_SBR_y,READ),
                          'M_SBR_z':(M_SBR_z,READ)})
        else: 
            one_velocity = Function(self.V_velocity)
            one_velocity.assign(1.0)
            self.data = assemble(inner(TestFunction(self.V_velocity),one_velocity)*self.dx)
        self.data_inv = Function(self.V_velocity)
        kernel_inv = '*data_inv = 1./(*data);'
        par_loop(kernel_inv,direct,
                 {'data_inv':(self.data_inv,WRITE),
                  'data':(self.data,READ)})

    def get(self):
        '''Return :math:`RT1` representation of mass matrix.'''
        return self.data

    def _matmul(self,m,u):
        '''Multiply by diagonal matrix

        In-place multiply a RT1 field by a diagonal matrix, which 
        is either the lumped mass matrix or its inverse.

        :arg m: block-diagonal matrix to multiply with
        :arg u: RT1 field to multiply (will be modified in-place)
        '''
        kernel = '(*u) *= (*m);'
        par_loop(kernel,direct,
                 {'u':(u,RW),
                  'm':(m,READ)})

class LumpedMassBDFM1(LumpedMass):
    '''BDFM1 lumped mass matrix.

    Represents a lumped approximation of the BDFM1 mass matrix. The matrix
    is block-diagonal with each 4x4 block corresponding to the couplings
    between the dofs on one facet (2 continuous normal dofs, 2 discontinuous
    tangential dofs). It is constructed by requiring that on each facet the
    lumped mass matrix gives the same result as the full BDFM1 mass matrix 
    when applied to a set of solid body rotation fields.

    Internally each block of the lumped mass matrix is represented as a
    Dat of suitable shape located on the facets, i.e. the same dof-map as
    for the RT0 space can be used.

    :arg V_velocity: Velocity function space, has to be of type
        :math:`BDFM_1`
    :arg ignore_lumping: For debugging, this can be set to true to use the
        full mass matrix in the :class:`multiply()` and :class:`divide()`
        methods.
    :arg diagonal_matrix: Assume local blocks are diagonal. This allows for 
        some further optimisations. NB: Currently only this option is
        supported in :class:`Jacobi_HigherOrder`
    '''
    def __init__(self,V_velocity,ignore_lumping=False,
                 diagonal_matrix=True):
        super(LumpedMassBDFM1,self).__init__(V_velocity,ignore_lumping)
        self.coords = self.mesh.coordinates
        self.n_SBR=4
        self.diagonal_matrix = diagonal_matrix
        # Coordinate space
        self.V_coords = self.coords.function_space()
        # Set up map from facets to coordinate dofs
        self.facet2dof_map_coords = self._build_interiorfacet2dofmap_coords()
        # Space with one dof per facet (hijack RT0 space)
        self.V_facets = FunctionSpace(self.mesh,'RT',1)
        # Set up map from facets to dofs on facet
        self.facet2dof_map_facets = self._build_interiorfacet2dofmap_facets()
        # Set up map from facets to BDFM1 dofs on facet
        self.facet2dof_map_BDFM1 = self._build_interiorfacet2dofmap_BDFM1()
        # Build lumped mass matrix
        self._build_lumped_massmatrix()
        
    def _build_interiorfacet2dofmap_coords(self):
        '''Map to facet coordinates

        Build a map from the interior facets to the dofs of the 
        two coordinates on the adjacents vertices.
        The map is constructed by using the interior_facets.local_facet_dat
        structure.
        '''
        cell2dof_map = self.V_coords.cell_node_map()
        facet2celldof_map = self.V_coords.interior_facet_node_map()
        facet2dof_map_val = []
        # Loop over all facets and identify shared dofs
        for (x,y) in zip(facet2celldof_map.values,
                         self.mesh.interior_facets.local_facet_dat.data):
            local_map = {0:[1,2],1:[0,2],2:[0,1]}
            dofs = [x[local_map[y[0]][i]] for i in range(2)]
            facet2dof_map_val.append(dofs)
        toset = cell2dof_map.toset
        return op2.Map(self.mesh.interior_facets.set,toset,2,
                       values=facet2dof_map_val)

    def _build_interiorfacet2dofmap_facets(self):
        '''Map to facet dofs

        Build a map from the interior facets to the facet, i.e. the RT0
        dofs. The map is constructed by looping over all facets, finding the
        RT0 dofs of the adjacent cells and identifying the local index of the
        facet in the adjacent cells via interor_facets.local_facet_dat.

        This map is used to access the lumped 4x4 mass matrix.
        '''
        cell2dof_map = self.V_facets.cell_node_map()
        facet2celldof_map = self.V_facets.interior_facet_node_map()
        facet2dof_map_val = []
        for (x,idx) in zip(facet2celldof_map.values,
                           self.mesh.interior_facets.local_facet_dat.data):
            facet2dof_map_val.append(x[idx[0]])
        toset = cell2dof_map.toset
        return op2.Map(self.mesh.interior_facets.set,toset,1,
                       values=facet2dof_map_val)

    def _build_interiorfacet2dofmap_BDFM1(self):
        '''Map to BDFM1 dofs on a facet

        Build a map from the interior facets to the four BDFM1 dofs 
        associated with this facet. For this, loop over the BDFM1 dofs in the
        cells associated with this facet and use local_facet_dat to identify
        the local index of the facet in each of the two cells.
        On each cells are always ordered like this
        :math:`(a_1,a_2,b_1,b_2,c_1,c_2,a_3,b_3,c_3)`, where :math:`a_1` and
        :math:`a_2` are the normal dofs on edge 1 and :math:`a_3` is the
        tangential dof.
        '''
        cell2dof_map = self.V_velocity.cell_node_map()
        facet2celldof_map = self.V_velocity.interior_facet_node_map()
        facet2dof_map_val = []
        for (x,idx) in zip(facet2celldof_map.values,
                           self.mesh.interior_facets.local_facet_dat.data):
            dofs = [x[2*idx[0]],x[2*idx[0]+1],x[6+idx[0]],x[9+6+idx[1]]]
            facet2dof_map_val.append(dofs)
        toset = cell2dof_map.toset
        return op2.Map(self.mesh.interior_facets.set,toset,4,
                       values=facet2dof_map_val)

    def _construct_MU_U(self):
        '''Construct the solid body rotation fields.

        Construct BDFM1 fields from the projection of
        :math:`u_x=(0,-z,y)`, :math:`u_y=(z,0,-x)`, :math:`u_z=(-y,x,0)`,  
        :math:`\\tilde{u_x}=(0,-zx,yx)`, :math:`\\tilde{u_y}=(zy,0,-xy)`,  
        :math:`\\tilde{u_z}=(-yz,xz,0)` and the corresponding six fields
        which are obtained by applying the BDFM1 mass matrix to these.
        '''
        # Set columns of matrix to values of the vector functions
        kernel_filename = os.path.join(os.path.dirname(__file__),
                                       'kernel_bdfm1_lumpedmass.c')
        kernel_file = file(kernel_filename,'r')
        kernel_code = ''
        for line in kernel_file:
            kernel_code += line
        kernel_file.close()
        kernel = op2.Kernel(kernel_code,"set_matrix")

        toset = self.V_facets.cell_node_map().toset
        w = TestFunction(self.V_velocity)
        m_U = Function(self.V_facets,
                       val=op2.Dat(toset**(self.n_SBR,4),
                       dtype=float))
        m_MU = Function(self.V_facets,
                        val=op2.Dat(toset**(self.n_SBR,4),
                        dtype=float))

        U_x = Function(self.V_velocity).project(Expression(('0','-x[2]','x[1]')))
        U_y = Function(self.V_velocity).project(Expression(('x[2]','0','-x[0]')))
        U_z = Function(self.V_velocity).project(Expression(('-x[1]','x[0]','0')))
        U_tilde_x = Function(self.V_velocity).project(Expression(('0',
                                                               '-x[2]*x[0]',
                                                               'x[1]*x[0]')))
        U_tilde_y = Function(self.V_velocity).project(Expression(('x[2]*x[1]',
                                                               '0',
                                                               '-x[0]*x[1]')))
        U_tilde_z = Function(self.V_velocity).project(Expression(('-x[1]*x[2]',
                                                               'x[0]*x[2]',
                                                               '0')))
        MU_x = assemble(dot(w,U_x)*dx)
        MU_y = assemble(dot(w,U_y)*dx)
        MU_z = assemble(dot(w,U_z)*dx)
        MU_tilde_x = assemble(dot(w,U_tilde_x)*dx)
        MU_tilde_y = assemble(dot(w,U_tilde_y)*dx)
        MU_tilde_z = assemble(dot(w,U_tilde_z)*dx)

        op2.par_loop(kernel,self.mesh.interior_facets.set,
                     m_U.dat(op2.WRITE,self.facet2dof_map_facets),
                     self.coords.dat(op2.READ,self.facet2dof_map_coords),
                     U_x.dat(op2.READ,self.facet2dof_map_BDFM1),
                     U_y.dat(op2.READ,self.facet2dof_map_BDFM1),
                     U_z.dat(op2.READ,self.facet2dof_map_BDFM1),
                     U_tilde_x.dat(op2.READ,self.facet2dof_map_BDFM1),
                     U_tilde_y.dat(op2.READ,self.facet2dof_map_BDFM1),
                     U_tilde_z.dat(op2.READ,self.facet2dof_map_BDFM1))

        op2.par_loop(kernel,self.mesh.interior_facets.set,
                     m_MU.dat(op2.WRITE,self.facet2dof_map_facets),
                     self.coords.dat(op2.READ,self.facet2dof_map_coords),
                     MU_x.dat(op2.READ,self.facet2dof_map_BDFM1),
                     MU_y.dat(op2.READ,self.facet2dof_map_BDFM1),
                     MU_z.dat(op2.READ,self.facet2dof_map_BDFM1),
                     MU_tilde_x.dat(op2.READ,self.facet2dof_map_BDFM1),
                     MU_tilde_y.dat(op2.READ,self.facet2dof_map_BDFM1),
                     MU_tilde_z.dat(op2.READ,self.facet2dof_map_BDFM1))
        return (m_U,m_MU)

    def _build_lumped_massmatrix(self):
        '''Build the lumped mass matrix.

        Construct the lumped mass matrix by looping over all factes and
        calculating the suilably rotated solid rotation fields :math:`u^{(i)}` 
        which are tangential/normal to that facet and the fields
        :math:`v^{(i)}` which are obtained from these by a mass matrix 
        application. Store the results in a 4x4 matrix
        on that facet. The lumped mass matrix is obtained by requiring that
        the result of the full mass matrix application is close to the
        result of a lumped mass matrix application, i.e. minimise

        :math: \sum_{k=1}^{\\nu} ||(M_u^*)_{ee}u^{(i)}-v^{(i)}||_2^2
    
        on each edge with constraints on the local 4x4 lumped mass matrix
        :math:`(M_u^*)_{ee}`. It should be symmetric and positive definite.
        Currently we also enforce it to be diagonal if the flag
        diagonal_matrix has been set in the constructor.
        '''
        m_U, m_MU = self._construct_MU_U()
        toset = self.V_facets.cell_node_map().toset
        if (self.diagonal_matrix):
            self.mat_toset = toset**(4,1)
        else:
            self.mat_toset = toset**(4,4)
        self.data = Function(self.V_facets,
                             val=op2.Dat(self.mat_toset,
                             dtype=float))
        self.data_inv = Function(self.V_facets,
                                 val=op2.Dat(self.mat_toset,
                                 dtype=float))
        if (self.diagonal_matrix):
            kernel = '''{
              for (int mu=0;mu<4;++mu) {
                double b = 0;
                double r = 0;
                for (int k=0;k<4;++k) {
                  double u_tmp = u[4*k+mu];
                  double v_tmp = v[4*k+mu];
                  b += u_tmp*u_tmp;
                  r += u_tmp*v_tmp;
                }
                data[mu] = b/r;
                data_inv[mu] = r/b;
              }
            }'''
            par_loop(kernel,direct,
                     {'u':(m_U,READ),
                      'v':(m_MU,READ),
                      'data':(self.data,WRITE),
                      'data_inv':(self.data_inv,WRITE)})
        else:
            d_U = m_U.dat.data
            d_MU = m_MU.dat.data
            d_data = self.data.dat.data
            d_data_inv = self.data_inv.dat.data

            # Build matrix basis for local lumped mass matrix
            # NB: Currently basis is the diagonal basis, so will
            # give same results as with diagonal_matrix
            a_basis = []
            for i in range(0,4):
                a = np.matrix(np.zeros((4,4)))
                a[i,i] = 1
                a_basis.append(a)
            n_basis = len(a_basis)

            # Loop over all edges and construct the lumped matrix
            for (U,V,i) in zip(d_U,d_MU,range(len(d_data))):
                B = np.matrix(np.zeros((n_basis,n_basis),dtype=float))
                R = np.matrix(np.zeros((n_basis,1),dtype=float))
                for k in range(self.n_SBR):
                    u = np.matrix(U[k])
                    v = np.matrix(V[k])
                    for mu in range(n_basis):
                        for nu in range(n_basis):
                            m = u*a_basis[mu]*a_basis[nu]*u.transpose()
                            B[mu,nu] += m[0,0]
                        r = u*a_basis[mu]*v.transpose()
                        R[mu] += r[0,0]
                
                coeff = np.linalg.solve(B,R)
                d_data[i] = np.zeros((4,4))
                for j in range(n_basis):
                    d_data[i] += coeff[j,0]*a_basis[j]
                d_data_inv[i] = np.linalg.inv(d_data[i])

    def _matmul(self,m,u):
        '''Multiply by block-diagonal matrix

        In-place multiply a BDFM1 field by a block-diagonal matrix, which 
        is either the lumped mass matrix or its inverse.

        :arg m: block-diagonal matrix to multiply with
        :arg u: BDFM1 field to multiply (will be modified in-place)
        '''
        if (self.diagonal_matrix):
            kernel_code = '''void matmul(double **m,
                                         double **U) {
                               for (int i=0; i<4; ++i) {
                                 U[i][0] *= m[0][i];
                               }
                             }'''
        else:
            kernel_code = '''void matmul(double **m,
                                         double **U) {
                               double tmp[4];
                               for (int i=0; i<4; ++i) tmp[i] = U[i][0];
                               for (int i=0; i<4; ++i) {
                                 U[i][0] = 0.0;
                                 for (int j=0; j<4; ++j) {
                                   U[i][0] += m[0][4*i+j]*tmp[j];
                                 }
                               }
                             }'''
        kernel = op2.Kernel(kernel_code,"matmul")
        op2.par_loop(kernel,self.mesh.interior_facets.set,
                     m.dat(op2.READ,self.facet2dof_map_facets),
                     u.dat(op2.RW,self.facet2dof_map_BDFM1))
