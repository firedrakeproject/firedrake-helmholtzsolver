from firedrake import *
from lumpedmass import *

class Operator(object):
    '''Schur complement operator with lumped velocity mass matrix.

    The class provides methods for applying the linear operator for the Schur
    complement pressure system in a matrix-free way. Explicity the operator is
    given by

    .. math::
        
        
        H = M_\phi + \omega^2 B^T (M_{u}^*)^{-1} B

    where :math:`B` and :math:`B^T` arise from the finite element
    representation of the divergence and gradient operator. The lumped mass
    matrix is represented by a :class:`.LumpedMassRT0` or
    :class:`.LumpedMassBDFM1` object.
    
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
        self.velocity_mass_matrix = velocity_mass_matrix

    def apply(self,phi):
        '''Apply operator.

        Apply the operator :math:`H` to a field :math:`phi` in a matrix free
        way by applying the individual components in turn and return the
        result :math:`H\phi`.

        :arg phi: Pressure field :math:`phi` to apply the operator to
        '''
        # Calculate action of B
        B_phi = assemble(div(self.w)*phi*self.dx)
        # divide by lumped velocity mass matrix
        self.velocity_mass_matrix.divide(B_phi)
        # Calculate action of B^T
        BT_B_phi = assemble(self.psi*div(B_phi)*self.dx)
        # Calculate action of pressure mass matrix
        M_phi = assemble(self.psi*phi*self.dx)
        return assemble(M_phi + self.omega**2*BT_B_phi)

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

class OperatorHierarchy(object):
    '''Hierarchy of :class:`.Operator` s on function space hierarchy.

    Collection of operators on different levels of a function space hierarchy
    which represents fields on different multigrid levels.
    
    :arg V_pressure_hierarchy: Hierarchical function space for pressure fields
    :arg V_velocity_hierarchy: Hierarchical function space for velocity fields
    :arg velocity_mass_matrix_hierarchy: Hierarchy of lumped mass matrices
    :arg omega: Positive real constant
    '''
    def __init__(self,
                 V_pressure_hierarchy,
                 V_velocity_hierarchy,
                 velocity_mass_matrix_hierarchy,
                 omega):
        self.omega = omega
        self.V_pressure_hierarchy = V_pressure_hierarchy
        self.V_velocity_hierarchy = V_velocity_hierarchy
        self.velocity_mass_matrix_hierarchy = velocity_mass_matrix_hierarchy
        self._hierarchy = [Operator(V_pressure,
                                    V_velocity,
                                    velocity_mass_matrix,
                                    self.omega)
                           for (V_pressure,
                                V_velocity,
                                velocity_mass_matrix) in 
                              zip(self.V_pressure_hierarchy,
                                  self.V_velocity_hierarchy,
                                  self.velocity_mass_matrix_hierarchy)]

    def __getitem__(self,level):
        '''Return operator on given level in the functionspace hierarchy.

        :arg level: level in hierarchy
        '''
        return self._hierarchy[level]

    def __len__(self):
        '''Return number of levels in operator hierarchy.'''
        return len(self._hierarchy)

