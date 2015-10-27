from firedrake import *
from firedrake.petsc import PETSc
from lumpedmass import *
import xml.etree.cElementTree as ET
from firedrake.ffc_interface import compile_form
from pyop2.profiling import timed_function, timed_region
from mpi4py import MPI
from bandedmatrix import *

class Operator_H(object):
    '''Schur complement operator :math:`H`.

        The class provides methods for applying the linear operator which arises in the
        preconditioner for the Schur complement pressure system in a matrix-free way.

        Explicity the operator is given by

        .. math::
        
            H = M_p + \omega_c^2 D (\\tilde{M}_u)^{-1} D^T

        where :math:`D` represents the weak derivative.

        :arg W3: L2 function space for pressure fields
        :arg W2: HDiv function space for velocity fields
        :arg omega_c: Positive real constant arising from sound wave speed,
            :math:`\omega_c=\\frac{\Delta t}{2}c`
        :arg mutilde: Mass matrix :math:`\\tilde{M}_u`, see :class:`.Mutilde`
    '''
    def __init__(self,W3,W2,mutilde,omega_c):
        self._W3 = W3
        self._W2 = W2
        self._mutilde = mutilde
        self._omega_c = omega_c
        self._omega_c2 = Constant(self._omega_c**2)
        self._dx = self._W3.mesh()._dx
        self._phi_test = TestFunction(self._W3)
        self._u_test = TestFunction(self._W2)
        self._u_trial = TrialFunction(self._W2)
        self._phi_tmp = Function(self._W3)
        self._res_tmp = Function(self._W3)
        self._bcs = [DirichletBC(self._W2, 0.0, "bottom"),
                     DirichletBC(self._W2, 0.0, "top")]
        with self._phi_tmp.dat.vec as v:
            self._iset = PETSc.IS().createStride(self._W3.dof_dset.size,
                                                 first=v.owner_range[0],
                                                 step=1,
                                                 comm=v.comm)
            

    def _apply_bcs(self,u):
        '''Apply boundary conditions to velocity function.

            :arg u: Function in velocity space
        '''
        for bc in self._bcs:
            bc.apply(u)

    @timed_function("apply_H")
    def apply(self,phi):
        '''Apply operator to pressure field and return result.

        :arg phi: pressure field
        '''
        BT_phi = assemble(div(self._u_test)*phi*self._dx)
        #self._apply_bcs(BT_phi)
        Mutildeinv_BT_phi = Function(self._W2)
        self._mutilde.divide(BT_phi,Mutildeinv_BT_phi)
        B_Mutildeinv_BT_phi = self._phi_test*div(Mutildeinv_BT_phi)*self._dx
        M_phi_phi = self._phi_test*phi*self._dx
        return assemble(M_phi_phi + self._omega_c2*B_Mutildeinv_BT_phi)

    def mult(self,mat,x,y):
        '''PETSc interface for operator application.

        PETSc interface wrapper for the :func:`apply` method.

        :arg x: PETSc vector representing the field to be multiplied.
        :arg y: PETSc vector representing the result.
        '''
        with self._phi_tmp.dat.vec as v:
            tmp = x.getSubVector(self._iset)
            x.copy(v)
            x.restoreSubVector(self._iset, tmp)
        self._res_tmp = self.apply(self._phi_tmp)
        with self._res_tmp.dat.vec_ro as v:
            y.array[:] = v.array[:]
    
    def add_to_xml(self,parent,function):
        '''Add to existing xml tree.

        :arg parent: Parent node to be added to
        :arg function: Function of object
        '''
        e = ET.SubElement(parent,function)
        e.set("type",type(self).__name__)
        v_str = str(self._W3.ufl_element().shortstr())
        e.set("pressure_space",v_str)
        v_str = str(self._W2.ufl_element().shortstr())
        e.set("velocity_space",v_str)


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
        
        
        \hat{H} = M_\phi + \omega_c^2(D_h M_{u,h,inv} D_h^T+1/(1+\omega_N^2)D_v M_{u,v,inv} D_v^T)

    where :math:`D_h` and :math:`D_v` arise from the finite element
    representation of the divergence and gradient operator in the horizontal- and vertical
    direction. Both the horizontal and vertical mass matrices are obtained by
    diagonal lumping.
    
    :arg W3: Function space for pressure fields
    :arg W2_h: Function space for horizontal component of velocity fields
    :arg W2_v: Function space for vertical component of velocity fields
    :arg omega_c: Positive real constant, related to sound wave speed,
        :math:`\omega_c=\\frac{\Delta t}{2}c`
    :arg omega_N: Positive real constant, related to buoyancy frequency
        :math:`\omega_c=\\frac{\Delta t}{2}N`
    :arg preassemble_horizontal: Pre-assemble horizontal part of the operator
    '''
    def __init__(self,
                 W3,
                 W2_h,
                 W2_v,
                 omega_c,
                 omega_N,
                 preassemble_horizontal=True,
                 level=-1):
        self._W3 = W3
        self._W2_h = W2_h
        self._W2_v = W2_v
        self._omega_c = omega_c
        self._omega_N = omega_N
        self._omega = self._omega_c**2/(1.+self._omega_N**2)
        self._omega_c2 = Constant(omega_c**2)
        self._const2 = Constant(omega_c**2/(1.+self._omega_N**2))
        self._preassemble_horizontal = preassemble_horizontal
        ncells = MPI.COMM_WORLD.allreduce(self._W3.mesh().cell_set.size)
        w_h = TestFunction(self._W2_h)
        w_v = TestFunction(self._W2_v)
        self._psi = TestFunction(self._W3)
        self._phi_tmp = Function(self._W3)
        self._res_tmp = Function(self._W3)
        self._mesh = self._W3.mesh()
        self._dx = self._mesh._dx
        self._level=level
        # Forms for operator applications
        self._B_v_phi_form = div(w_v)*self._phi_tmp*self._dx
        self._B_h_phi = Function(self._W2_h)
        self._B_v_phi = Function(self._W2_v)
        self._BT_B_v_phi_form = self._psi*div(self._B_v_phi)*self._dx
        self._M_phi_form = self._psi*self._phi_tmp*self._dx
        self._M_phi = Function(self._W3)
        self._BT_B_h_phi = Function(self._W3)
        self._BT_B_v_phi = Function(self._W3)
        self._Mu_h = LumpedMass(dot(w_h,TrialFunction(self._W2_h))*self._dx,
                                label='h')
        if (self._preassemble_horizontal):
            with timed_region('assemble B_h'):
                mat_B_h = \
                    assemble(div(TestFunction(self._W2_h))*TrialFunction(self._W3)*self._dx).M.handle
            tmp_h = mat_B_h.duplicate(copy=True)
            with timed_region('diagonal_scale'):
                with self._Mu_h._data_inv.dat.vec_ro as inv_diag:
                    tmp_h.diagonalScale(L=inv_diag,R=None)
            with timed_region('transposeMatMult'):
                self._mat_Hhat_h = mat_B_h.transposeMatMult(tmp_h)
        else:
            self._B_h_phi_form = div(w_h)*self._phi_tmp*self._dx
            self._BT_B_h_phi_form = self._psi*div(self._B_h_phi)*self._dx

        # Lumped mass matrices.
        Mu_v = BandedMatrix(self._W2_v,self._W2_v,label='Mu_v_level_'+str(self._level))
        Mu_v.assemble_ufl_form(dot(w_v,TrialFunction(self._W2_v))*self._dx,
                               vertical_bcs=True)
        self._Mu_vinv = Mu_v.inv_diagonal()
        B_v = BandedMatrix(self._W2_v,self._W3,label='B_v_level_'+str(self._level))
        B_v.assemble_ufl_form(div(w_v)*TrialFunction(self._W3)*self._dx,
                              vertical_bcs=True)
        M_phi = BandedMatrix(self._W3,self._W3,label='M_phi_level_'+str(self._level))
        M_phi.assemble_ufl_form(TestFunction(self._W3)*TrialFunction(self._W3)*self._dx,
                                vertical_bcs=True)
        self._Hhat_v = M_phi.matadd(B_v.transpose_matmul(self._Mu_vinv.matmul(B_v)),
                                    omega=self._omega)
        self._bcs = [DirichletBC(self._W2_v, 0.0, "bottom"),
                     DirichletBC(self._W2_v, 0.0, "top")]
        with self._phi_tmp.dat.vec as v:
            self._iset = PETSc.IS().createStride(self._W3.dof_dset.size,
                                                 first=v.owner_range[0],
                                                 step=1,
                                                 comm=PETSc.COMM_SELF)
        with timed_region('vertical_diagonal'):
            self._vertical_diagonal = self.vertical_diagonal()

    def _apply_bcs(self,u):
        '''Apply boundary conditions to velocity function.

            :arg u: Function in velocity space
        '''
        for bc in self._bcs:
            bc.apply(u)

    def add_to_xml(self,parent,function):
        '''Add to existing xml tree.

        :arg parent: Parent node to be added to
        :arg function: Function of object
        '''
        e = ET.SubElement(parent,function)
        e.set("type",type(self).__name__)
        v_str = str(self._W3.ufl_element().shortstr())
        e.set("pressure_space",v_str)
        v_str = str(self._W2_h.ufl_element().shortstr())
        e.set("velocity_space_horizontal",v_str)
        v_str = str(self._W2_v.ufl_element().shortstr())
        e.set("velocity_space_vertical",v_str)

    def apply(self,phi):
        '''Apply operator.

        Apply the operator :math:`\hat{H}` to a field :math:`\phi` in a matrix free
        way by applying the individual components in turn and return the
        result :math:`\hat{H}\phi`.

        :arg phi: Pressure field :math:`\phi` to apply the operator to
        '''
        with timed_region('apply_Hhat_level_'+str(self._level)):
            self._phi_tmp.assign(phi)
            with timed_region('apply_Hhat_h_level_'+str(self._level)):
                if (self._preassemble_horizontal):
                    with self._BT_B_h_phi.dat.vec as v:
                        with phi.dat.vec_ro as x:
                            self._mat_Hhat_h.mult(x,v)
                else:
                    # Calculate action of B_h
                    assemble(self._B_h_phi_form, tensor=self._B_h_phi)
                    # divide by horizontal velocity mass matrix
                    self._Mu_h.divide(self._B_h_phi)
                    # Calculate action of B_h^T
                    assemble(self._BT_B_h_phi_form, tensor=self._BT_B_h_phi)
            with timed_region('apply_Hhat_z_level_'+str(self._level)):
                self._Hhat_v._label='apply_Hhat_z_level_'+str(self._level)
                self._Hhat_v.ax(self._phi_tmp)
        return assemble(self._phi_tmp + self._omega_c2*self._BT_B_h_phi)

    def apply_blockinverse(self,r):
        '''In-place multiply with inverse of block-diagonal

        Apply :math:`r\mapsto \hat{H}_z^{-1} r`
        
        :arg r: Vector to be multiplied
        '''
        with timed_region('apply_Hhat_z_inv_level_'+str(self._level)):
            self._vertical_diagonal._label='Hhat_z_level_'+str(self._level)
            self._vertical_diagonal._lu_solve(r)

    def vertical_diagonal(self):
        '''Construct the block-diagonal matrix :math:`\hat{H}_z` which only 
        contains vertical couplings.'''

        phi_test = TestFunction(self._W3)
        phi_trial = TrialFunction(self._W3)
        w_h_test = TestFunction(self._W2_h)
        w_h_trial = TrialFunction(self._W2_h)

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

        delta_h = BandedMatrix(self._W3,self._W3,label='delta_h_level_'+str(self._level))
        delta_h._assemble_lma(lma_delta_h)

        # Add everything up       
        delta_h.scale(self._omega_c**2)
        result = self._Hhat_v.matadd(delta_h)
        result._lu_decompose()
        return result

    def mult(self,mat,x,y):
        '''PETSc interface for operator application.

        PETSc interface wrapper for the :func:`apply` method.

        :arg x: PETSc vector representing the field to be multiplied.
        :arg y: PETSc vector representing the result.
        '''
        with self._phi_tmp.dat.vec as v:
            tmp = x.getSubVector(self._iset)
            x.copy(v)
            x.restoreSubVector(self._iset, tmp)
        self._res_tmp = self.apply(self._phi_tmp)
        with self._res_tmp.dat.vec_ro as v:
            y.array[:] = v.array[:]
    
    def residual(self,b,phi):
        '''Calculate the residual.

        Evaluate the residual :math:`r=b-H\phi` for a given RHS :math:`b` and
        return the result.

        :arg b: Right hand side pressure field
        :arg phi: Pressure field to apply the operator to.
        '''
        return assemble(b - self.apply(phi))
