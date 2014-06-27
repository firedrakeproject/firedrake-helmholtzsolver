import sys
from collections import Counter
from firedrake import *
import numpy as np
from matplotlib import pyplot as plt


#########################################################
# E X P E R I M E N T A L
# This code is currently tested and not part of the 
# solver yet.
#########################################################

class LumpedMassBDFM1(object):
    '''BDFM1 lumped mass
    '''
    def __init__(self,V_velocity):
        self.V_BDFM1 = V_velocity
        self.mesh = self.V_BDFM1.mesh()
        self.coords = self.mesh.coordinates
        self.n_SBR=4
        # Coordinate space
        self.V_coords = self.coords.function_space()
        # Set up map from facets to coordinate dofs
        self._build_interiorfacet2dofmap_coords()
        # Space with one dof per facet (hijack RT0 space)
        self.V_facets = FunctionSpace(mesh,'RT',1)
        self._build_interiorfacet2dofmap_facets()
        self._build_interiorfacet2dofmap_BDFM1()
        self._build_lumped_massmatrix()
        
    def _build_interiorfacet2dofmap_coords(self):
        '''Build a map from the interior facets to the dofs for the coordinates
        '''
        cell2dof_map = self.V_coords.cell_node_map()
        facet2celldof_map = self.V_coords.interior_facet_node_map()
        facet2dof_map_val = []
        # Loop over all facets and identify shared dofs
        for x in facet2celldof_map.values:
            # find duplicates
            c = Counter(x)
            vertex_idx=[]
            for k in c.keys():
                if c[k] == 2:
                    vertex_idx.append(k)
            dofs = [vertex_idx[0],vertex_idx[1]]
            facet2dof_map_val.append(dofs)
        toset = cell2dof_map.toset
        self.facet2dof_map_coords = op2.Map(self.mesh.interior_facets.set,
                                            toset,
                                            2,values=facet2dof_map_val)

    def _build_interiorfacet2dofmap_facets(self):
        '''Build a map from the interior facet's to the facet (i.e. RT0 dofs).
        '''
        cell2dof_map = self.V_facets.cell_node_map()
        facet2celldof_map = self.V_facets.interior_facet_node_map()
        facet2dof_map_val = []
        for x in facet2celldof_map.values:
            # find duplicates
            c = Counter(x)
            for k in c.keys():
                if c[k] == 2:
                    facet2dof_map_val.append(k)
        toset = cell2dof_map.toset
        self.facet2dof_map_facets = op2.Map(self.mesh.interior_facets.set,
                                            toset,
                                            1,values=facet2dof_map_val)

    def _build_interiorfacet2dofmap_BDFM1(self):
        '''Build a map from the interior facet's to the BDFM1 dofs
        '''
        cell2dof_map = self.V_BDFM1.cell_node_map()
        facet2celldof_map = self.V_BDFM1.interior_facet_node_map()
        facet2dof_map_val = []
        for x in facet2celldof_map.values:
            # find duplicated dofs
            c = Counter(x)
            for k in c.keys():
                if c[k] == 2:
                    duplicated_dof = k
                    break
            facet1_idx = np.argwhere(x==duplicated_dof)[0][0]/2
            facet2_idx = (np.argwhere(x==duplicated_dof)[1][0]-9)/2
            dofs = [x[2*facet1_idx],
                    x[2*facet1_idx+1],
                    x[6+facet1_idx],
                    x[9+6+facet2_idx]]
            facet2dof_map_val.append(dofs)
        toset = cell2dof_map.toset
        self.facet2dof_map_BDFM1 = op2.Map(self.mesh.interior_facets.set,
                                           toset,
                                           4,values=facet2dof_map_val)

    def _construct_MU_U(self):
        '''Construct and return the matrices U and MU.
        '''

        # Set columns of matrix to values of the vector functions
        kernel_file = file('kernel_bdfm1_lumpedmass.c','r')
        kernel_code = ''
        for line in kernel_file:
            kernel_code += line
        kernel_file.close()
        kernel = op2.Kernel(kernel_code,"set_matrix")

        toset = self.V_facets.cell_node_map().toset
        w = TestFunction(self.V_BDFM1)
        m_U = Function(self.V_facets,
                       val=op2.Dat(toset**(self.n_SBR,4),
                       dtype=float))
        m_MU = Function(self.V_facets,
                        val=op2.Dat(toset**(self.n_SBR,4),
                        dtype=float))

        U_x = Function(self.V_BDFM1).project(Expression(('0','-x[2]','x[1]')))
        U_y = Function(self.V_BDFM1).project(Expression(('x[2]','0','-x[0]')))
        U_z = Function(self.V_BDFM1).project(Expression(('-x[1]','x[0]','0')))
        U_tilde_x = Function(self.V_BDFM1).project(Expression(('0',
                                                               '-x[2]*x[0]',
                                                               'x[1]*x[0]')))
        U_tilde_y = Function(self.V_BDFM1).project(Expression(('x[2]*x[1]',
                                                               '0',
                                                               '-x[0]*x[1]')))
        U_tilde_z = Function(self.V_BDFM1).project(Expression(('-x[1]*x[2]',
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
        m_U, m_MU = self._construct_MU_U()
        # Build matrix basis for local lumped mass matrix
        a_basis = []
        for i in range(0,4):
            a = np.matrix(np.zeros((4,4)))
            a[i,i] = 1
            a_basis.append(a)
        n_basis = len(a_basis)

        toset = self.V_facets.cell_node_map().toset
        self.Mu_lumped = Function(self.V_facets,
                                  val=op2.Dat(toset**(4,4),
                                  dtype=float))
        self.Mu_lumped_inv = Function(self.V_facets,
                                      val=op2.Dat(toset**(4,4),
                                      dtype=float))

        d_U = m_U.dat.data
        d_MU = m_MU.dat.data
        d_Mu_lumped = self.Mu_lumped.dat.data
        d_Mu_lumped_inv = self.Mu_lumped_inv.dat.data

        # Loop over all edges and construct the lumped matrix
        for (U,V,i) in zip(d_U,d_MU,range(len(d_Mu_lumped))):
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
            d_Mu_lumped[i] = np.zeros((4,4))
            for j in range(n_basis):
                d_Mu_lumped[i] += coeff[j,0]*a_basis[j]
            d_Mu_lumped_inv[i] = np.linalg.inv(d_Mu_lumped[i])

    def _matmul(self,m,u):
        kernel_code = '''void matmul(double **m,
                                     double **U, 
                                     double **mU) {
                           for (int i=0; i<4; ++i) {
                             mU[i][0] = 0.0;
                             for (int j=0; j<4; ++j) {
                               mU[i][0] += m[0][4*i+j]*U[j][0];
                             }
                           }
                         }'''
        v = Function(self.V_BDFM1)
        kernel = op2.Kernel(kernel_code,"matmul")
        op2.par_loop(kernel,self.mesh.interior_facets.set,
                     m.dat(op2.READ,self.facet2dof_map_facets),
                     u.dat(op2.READ,self.facet2dof_map_BDFM1),
                     v.dat(op2.WRITE,self.facet2dof_map_BDFM1))
        return v

    def multiply(self,u):
        '''Multiply a BDFM1 field with the lumped mass matrix.
        '''
        return self._matmul(self.Mu_lumped,u)
            

    def divide(self,u):
        '''Divide a BDFM1 field with the lumped mass matrix.
        '''
        return self._matmul(self.Mu_lumped_inv,u)    

# TESTING

if (__name__ == '__main__'):
    ref_count = 1
    mesh = UnitIcosahedralSphereMesh(refinement_level=ref_count)
    global_normal = Expression(("x[0]","x[1]","x[2]"))
    mesh.init_cell_orientations(global_normal)

    V_BDFM1 = FunctionSpace(mesh,'BDFM',2)
    Mu = LumpedMassBDFM1(V_BDFM1)

    U_z = Function(V_BDFM1).project(Expression(('-x[1]','x[0]','0')))

    E_kin = assemble(dot(U_z,U_z)*dx)

    MU_z = Mu.multiply(U_z)

    E_kin_lumped = 0.0
    for (u,v) in zip(U_z.dat.data,MU_z.dat.data):
        E_kin_lumped += u*v

    print '(Twice) kinetic energy: '
    print '   full   mass : '+ ('%8.4e' % E_kin)
    print '   lumped mass : '+ ('%8.4e' % E_kin_lumped)

    # Calculate eigenvalues
    f = Function(V_BDFM1)
    g = Function(V_BDFM1)
    ndofs = len(f.dat.data)
    m_eigen_full = np.zeros((ndofs,ndofs))
    m_eigen_lumped = np.zeros((ndofs,ndofs))
    w = TestFunction(V_BDFM1)
    for i in range(ndofs):
        f.dat.data[:] = 0 
        f.dat.data[i] = 1
        g = assemble(dot(w,f)*dx)
        m_eigen_full[i,:] = g.dat.data[:]
        g = Mu.multiply(f)
        m_eigen_lumped[i,:] = g.dat.data[:]
    evals_full = np.linalg.eigvalsh(m_eigen_full)
    evals_lumped = np.linalg.eigvalsh(m_eigen_lumped)

    max_eval = max(np.max(evals_full),max(evals_lumped))
    min_eval = min(np.min(evals_full),min(evals_lumped))
    y = np.zeros(ndofs)
    plt.clf()
    ax = plt.gca()
    ax.set_ylim(-0.3,0.3)
    ax.set_xlim(min_eval,max_eval)
    p1 = plt.plot(evals_full,y-0.2,color='red',markeredgecolor='red',linewidth=0,markersize=6,marker='o')[0]
    p2 = plt.plot(evals_lumped,y+0.2,color='blue',markeredgecolor='blue',linewidth=0,markersize=6,marker='o')[0]
    plt.legend((p1,p2),('full','lumped'),'upper left')
    plt.savefig('eigenvalues.pdf',bbox_inches='tight')


