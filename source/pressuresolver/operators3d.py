from firedrake import *
from lumpedmass_diagonal import *
import xml.etree.cElementTree as ET
from pyop2.profiling import timed_function, timed_region
from mpi4py import MPI
from bandedmatrix import *

class Operator_Hhat(object):
    '''Schur complement operator :math:`\hat{H}`.

    The class provides methods for applying the linear operator which arises in the
    preconditioner for the Schur complement pressure system in a matrix-free way.
    In the operator couplings between the vertical and horizontal verlocity spaces are
    dropped and the velocity mass matrices are lumped to be able to compute the inverse
    mass matrices.
    The vertical-diagonal, which is required by the smoother, can be extracted.

    Explicity the operator is given by

    .. math::
        
        
        \hat{H} = M_\phi + \omega_c^2(B_h M_{u,h,inv} B_h^T+1/(1+\omega_N^2)B_v M_{u,v,inv} B_v^T)

    where :math:`B_h` and :math:`B_v` arise from the finite element
    representation of the divergence and gradient operator in the horizontal- and vertical
    direction. The horizontal mass matrix is obtained by diagonal lumping, and the vertical
    mass matrix by a sparse approximate inverse (SPAI).
    
    :arg W_pressure: Function space for pressure fields
    :arg W_velocity_h: Function space for horizontal component of velocity fields
    :arg W_velocity_v: Function space for vertical component of velocity fields
    :arg omega_c: Positive real constant arising from acoustic frequency
    :arg omega_N: Positive real constant arising from buoyancy frequency
    '''
    def __init__(self,
                 W_pressure,
                 W_velocity_h,
                 W_velocity_v,
                 omega_c,
                 omega_N):
        self._W_pressure = W_pressure
        self._W_velocity_h = W_velocity_h
        self._W_velocity_v = W_velocity_v
        self._omega_c = omega_c
        self._omega_N = omega_N
        self._omega_c2 = Constant(omega_c**2)
        self._const2 = Constant(omega_c**2/(1.+omega_N**2))
        ncells = MPI.COMM_WORLD.allreduce(self._W_pressure.mesh().cell_set.size)
        self._timer_label = str(ncells)
        w_h = TestFunction(self._W_velocity_h)
        w_v = TestFunction(self._W_velocity_v)
        self._psi = TestFunction(self._W_pressure)
        self._phi_tmp = Function(self._W_pressure)
        self._res_tmp = Function(self._W_pressure)
        self._mesh = self._W_pressure.mesh()
        self._dx = self._mesh._dx

        # Forms for operator applications
        self._B_h_phi_form = div(w_h)*self._phi_tmp*self._dx
        self._B_v_phi_form = div(w_v)*self._phi_tmp*self._dx
        self._B_h_phi = Function(self._W_velocity_h)
        self._B_v_phi = Function(self._W_velocity_v)
        self._BT_B_h_phi_form = self._psi*div(self._B_h_phi)*self._dx
        self._BT_B_v_phi_form = self._psi*div(self._B_v_phi)*self._dx
        self._M_phi_form = self._psi*self._phi_tmp*self._dx
        self._M_phi = Function(self._W_pressure)
        self._BT_B_h_phi = Function(self._W_pressure)
        self._BT_B_v_phi = Function(self._W_pressure)

        # Lumped mass matrices.
        self._Mu_h = LumpedMass(self._W_velocity_h)
        Mu_v = BandedMatrix(self._W_velocity_v,self._W_velocity_v)
        self._Mu_vinv = Mu_v.spai()

    def add_to_xml(self,parent,function):
        '''Add to existing xml tree.

        :arg parent: Parent node to be added to
        :arg function: Function of object
        '''
        e = ET.SubElement(parent,function)
        e.set("type",type(self).__name__)
        v_str = self.W_pressure.ufl_element()._short_name
        v_str += str(self._W_pressure.ufl_element().degree())
        e.set("pressure_space",v_str)
        v_str = self._W_velocity_h.ufl_element()._short_name
        v_str += str(self._W_velocity_h.ufl_element().degree())
        e.set("velocity_space [horizontal]",v_str)
        v_str = self._W_velocity_v.ufl_element()._short_name
        v_str += str(self._W_velocity_v.ufl_element().degree())
        e.set("velocity_space [vertical]",v_str)

    @timed_function("apply_pressure_operator")
    def apply(self,phi):
        '''Apply operator.

        Apply the operator :math:`H` to a field :math:`phi` in a matrix free
        way by applying the individual components in turn and return the
        result :math:`H\phi`.

        :arg phi: Pressure field :math:`phi` to apply the operator to
        '''
        with timed_region('apply_pressure_operator_'+self._timer_label):
            # Calculate action of B_h
            self._phi_tmp.assign(phi)
            assemble(self._B_h_phi_form, tensor=self._B_h_phi)
            # divide by horizontal velocity mass matrix
            self._Mu_h.divide(self._B_h_phi)
            # Calculate action of B_h^T
            assemble(self._BT_B_h_phi_form, tensor=self._BT_B_h_phi)

            # Calculate action of B_v
            self._phi_tmp.assign(phi)
            assemble(self._B_v_phi_form, tensor=self._B_v_phi)
            # divide by vertical velocity mass matrix
            self._Mu_vinv.ax(self._B_v_phi)
            # Calculate action of B_v^T
            assemble(self._BT_B_v_phi_form, tensor=self._BT_B_v_phi)

            # Calculate action of pressure mass matrix
            assemble(self._M_phi_form, tensor=self._M_phi)
        return assemble(self._M_phi + \
                        self._omega_c2*self._BT_B_h_phi + \
                        self._const2*self._BT_B_v_phi)

    def vertical_diagonal(self):
        '''Construct the block-diagonal matrix :math:`\hat{H}_z` which only 
        contains vertical couplings.'''

        phi_test = TestFunction(self._W_pressure)
        phi_trial = TrialFunction(self._W_pressure)
        w_h_test = TestFunction(self._W_velocity_h)
        w_h_trial = TrialFunction(self._W_velocity_h)
        w_v_test = TestFunction(self._W_velocity_v)
        w_v_trial = TrialFunction(self._W_velocity_v)

        # Pressure mass matrix
        M_phi = BandedMatrix(self._W_pressure,self._W_pressure)
        M_phi.assemble_ufl_form(phi_test*phi_trial*self._dx)

        # B_v M_{u,v,inv} B_v^T
        B_v_T = BandedMatrix(self._W_velocity_v,self._W_pressure)
        B_v_T.assemble_ufl_form(div(w_v_test)*phi_trial*self._dx)
        B_v = BandedMatrix(self._W_pressure,self._W_velocity_v)
        B_v.assemble_ufl_form(phi_test*div(w_v_trial)*self._dx)
        B_v_Mu_vinv_B_v_T = B_v.matmul(self._Mu_vinv.matmul(B_v_T))

        # Build LMA for B_h and for delta_h = diag_h(B_h*M_{u,h,inv}*B_h^T)
        ufl_form = phi_test*div(w_h_trial)*self._dx
        compiled_form = compile_form(ufl_form, 'ufl_form')[0]
        kernel = compiled_form[6]
        coords = compiled_form[3]
        coefficients = compiled_form[4]
        arguments = ufl_form.arguments()
        ndof_pressure = arguments[0].cell_node_map().arity
        ndof_velocity_h = arguments[1].cell_node_map().arity

        # Build LMA for B_h
        V_lma = FunctionSpace(self._mesh,'DG',0)
        lma_B_h = Function(V_lma, val=op2.Dat(V_lma.node_set**(ndof_pressure*ndof_velocity_h)))
        args = [lma_B_h.dat(op2.INC, lma_B_h.cell_node_map()[op2.i[0]]), 
                coords.dat(op2.READ, coords.cell_node_map(), flatten=True)]
        for c in coefficients:
            args.append(c.dat(op2.READ, c.cell_node_map(), flatten=True))
        op2.par_loop(kernel,lma_B_h.cell_set, *args)

        # Build LMA representation for delta_h = diag_h(B_h*M_{u,h,inv}*B_h^T)
        lma_delta_h = Function(V_lma, val=op2.Dat(V_lma.node_set**(ndof_pressure**2)))
        kernel_code = '''void build_delta_h(double **lma_B_h,
                                            double **Mu_hinv,
                                            double **lma_delta_h) {
          for (int i=0;i<%(ndof_pressure)d;++i) {
            for (int j=0;j<%(ndof_pressure)d;++j) {
              double tmp = 0.0;
              for (int k=0;k<%(ndof_velocity_h)d;++k) {
                 tmp += lma_B_h[0][%(ndof_velocity_h)d*i+k]
                      * Mu_hinv[k][0]
                      * lma_B_h[0][%(ndof_velocity_h)d*j+k];
              }
              lma_delta_h[0][%(ndof_pressure)d*i+j] = tmp;
            }
          }
        }'''
        kernel = op2.Kernel(kernel_code % {'ndof_pressure':ndof_pressure,
                                           'ndof_velocity_h':ndof_velocity_h},
                            'build_delta_h')
        op2.par_loop(kernel,self._mesh.cell_set,
                     lma_B_h.dat(op2.READ,lma_B_h.cell_node_map()),
                     self._Mu_h._data_inv.dat(op2.READ,self._Mu_h._data_inv.cell_node_map()),
                     lma_delta_h.dat(op2.WRITE,lma_delta_h.cell_node_map()))

        delta_h = BandedMatrix(self._W_pressure,self._W_pressure)
        delta_h._assemble_lma(lma_delta_h)

        # Add everything up       
        return M_phi.matadd(delta_h.matadd(B_v_Mu_vinv_B_v_T,
                                           omega=1./(1.+self._omega_N**2)),
                            omega=self._omega_c**2)

    def mult(self,mat,x,y):
        '''PETSc interface for operator application.

        PETSc interface wrapper for the :func:`apply` method.

        :arg x: PETSc vector representing the field to be multiplied.
        :arg y: PETSc vector representing the result.
        '''
        with self.phi_tmp.dat.vec as v:
            v.array[:] = x.array[:]
        self.res_tmp = self.apply(self.phi_tmp)
        with self.res_tmp.dat.vec_ro as v:
            y.array[:] = v.array[:]
    
    def residual(self,b,phi):
        '''Calculate the residual.

        Evaluate the residual :math:`r=b-H\phi` for a given RHS :math:`b` and
        return the result.

        :arg b: Right hand side pressure field
        :arg phi: Pressure field to apply the operator to.
        '''
        return assemble(b - self.apply(phi))
