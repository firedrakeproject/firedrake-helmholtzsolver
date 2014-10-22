from firedrake import *
from lumpedmass import *
import xml.etree.cElementTree as ET
from pyop2.profiling import timed_function, timed_region
from mpi4py import MPI

class Operator(object):
    '''Schur complement operator with lumped velocity mass matrix.

    The class provides methods for applying the linear operator for the Schur
    complement pressure system in a matrix-free way. Explicity the operator is
    given by

    .. math::
        
        
        H = M_\phi + \omega^2 B^T (M_{u}^*)^{-1} B

    where :math:`B` and :math:`B^T` arise from the finite element
    representation of the divergence and gradient operator. The 
    (possibly lumped) velocity mass matrix :math:`M_{u}^*` is represented by
    a :class:`.FullMass`, :class:`.LumpedMassRT0` or :class:`.LumpedMassBDFM1`
    instance.
    
    :arg V_pressure: Function space for pressure fields
    :arg V_velocity: Function space for velocity fields
    :arg velocity_mass_matrix: Velocity mass matrix
    :arg omega: Positive real constant
    '''
    def __init__(self,
                 V_pressure,
                 V_velocity,
                 velocity_mass_matrix,
                 omega):
        self.V_velocity = V_velocity
        self.V_pressure = V_pressure
        self.w = TestFunction(self.V_velocity)
        self.psi = TestFunction(self.V_pressure)
        self.phi_tmp = Function(self.V_pressure)
        self.res_tmp = Function(self.V_pressure)
        self.dx = self.V_pressure.mesh()._dx
        self.omega = omega
        self.omega2 = Constant(omega**2)
        self.velocity_mass_matrix = velocity_mass_matrix
        ncells = MPI.COMM_WORLD.allreduce(self.V_pressure.mesh().cell_set.size)
        if (type(self.velocity_mass_matrix) is LumpedMassRT0):
            self.timer_label = 'DG0_'
        elif (type(self.velocity_mass_matrix) is LumpedMassBDFM1):
            self.timer_label = 'DG1_'
        else:
            self.timer_label = ''
        self.timer_label += str(ncells)

        self.B_phi_form = div(self.w)*self.phi_tmp*self.dx
        self.B_phi = Function(self.V_velocity)
        self.BT_B_phi_form = self.psi*div(self.B_phi)*self.dx
        self.M_phi_form = self.psi*self.phi_tmp*self.dx
        self.M_phi = Function(self.V_pressure)
        self.BT_B_phi = Function(self.V_pressure)

    def add_to_xml(self,parent,function):
        '''Add to existing xml tree.

        :arg parent: Parent node to be added to
        :arg function: Function of object
        '''
        e = ET.SubElement(parent,function)
        e.set("type",type(self).__name__)
        v_str = self.V_pressure.ufl_element()._short_name
        v_str += str(self.V_pressure.ufl_element().degree())
        e.set("pressure_space",v_str)
        v_str = self.V_velocity.ufl_element()._short_name
        v_str += str(self.V_velocity.ufl_element().degree())
        e.set("velocity_space",v_str)
        self.velocity_mass_matrix.add_to_xml(e,'velocity_mass_matrix')

    @timed_function("apply_pressure_operator")
    def apply(self,phi):
        '''Apply operator.

        Apply the operator :math:`H` to a field :math:`phi` in a matrix free
        way by applying the individual components in turn and return the
        result :math:`H\phi`.

        :arg phi: Pressure field :math:`phi` to apply the operator to
        '''
        with timed_region('apply_pressure_operator_'+self.timer_label):
            # Calculate action of B
            self.phi_tmp.assign(phi)
            assemble(self.B_phi_form, tensor=self.B_phi)
            # divide by lumped velocity mass matrix
            self.velocity_mass_matrix.divide(self.B_phi)
            # Calculate action of B^T
            assemble(self.BT_B_phi_form, tensor=self.BT_B_phi)
            # Calculate action of pressure mass matrix
            assemble(self.M_phi_form, tensor=self.M_phi)
        return assemble(self.M_phi + self.omega2*self.BT_B_phi)

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
