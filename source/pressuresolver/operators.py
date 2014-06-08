from lumpedmass import *

class Operator(object):
    '''Schur complement operator with lumped velocity mass matrix.

    The class provides methods for applying the linear operator for the Schur complement
    pressure system in a matrix-free way. Explicity the operator is given by

    .. math::
        
        
        H = M_\phi + \omega^2 B^T (M_{u}^*)^{-1} B

    where :math:`B` and :math:`B^T` arise from the finite element representation of the
    divergence and gradient operator. The lumped mass matrix is represented by a 
    :class:`.LumpedMass` object.
    
    :arg V_pressure: Function space for pressure fields
    :arg V_velocity: Function space for velocity fields
    :arg omega: Positive real constant
    :arg ignore_mass_lumping: For debugging this can be set to true to use the
        full mass matrix
    '''
    def __init__(self,V_pressure,V_velocity,omega,
                 ignore_mass_lumping=False):
        self.omega = omega
        self.ignore_mass_lumping = ignore_mass_lumping
        self.V_velocity = V_velocity
        self.V_pressure = V_pressure
        self.w = TestFunction(self.V_velocity)
        self.psi = TestFunction(self.V_pressure)
        self.dx = self.V_pressure.mesh()._dx
        self.lumped_mass = LumpedMass(self.V_velocity,self.ignore_mass_lumping)
    
    def apply(self,phi):
        '''Apply operator.

        Apply the operator :math:`H` to a field :math:`phi` in a matrix free way by 
        applying the individual components in turn and return the result :math:`H\phi`.
        :arg phi: Pressure field :math:`phi` to apply the operator to
        '''
        # Calculate action of B
        B_phi = assemble(div(self.w)*phi*self.dx)
        # divide by lumped velocity mass matrix
        self.lumped_mass.divide(B_phi)
        # Calculate action of B^T
        BT_B_phi = assemble(self.psi*div(B_phi)*self.dx)
        # Calculate action of pressure mass matrix
        M_phi = assemble(self.psi*phi*self.dx)
        return assemble(M_phi + self.omega**2*BT_B_phi)

    
    def residual(self,b,phi):
        '''Calculate the residual.

        Evaluate the residual :math:`r=b-H\phi` for a given RHS :math:`b` and return 
        the result.
        :arg b: Right hand side pressure field
        :arg phi: Pressure field to apply the operator to.
        '''
        return assemble(b - self.apply(phi))

class OperatorHierarchy(object):
    '''Hierarchy of :class:`.Operator` s on function space hierarchy.

    Collection of operators on different levels of a function space hierarchy which
    represents fields on different multigrid levels.
    
    :arg V_pressure_hierarchy: Hierarchical function space for pressure fields
    :arg V_velocity_hierarchy: Hierarchical function space for velocity fields
    :arg omega: Positive real constant
    :arg ignore_mass_lumping: For debugging this can be set to true to use the
        full mass matrix
    '''
    def __init__(self,V_pressure_hierarchy,V_velocity_hierarchy,omega,
                 ignore_mass_lumping=False):
        self.ignore_mass_lumping = ignore_mass_lumping
        self.omega = omega
        self.V_pressure_hierarchy = V_pressure_hierarchy
        self.V_velocity_hierarchy = V_velocity_hierarchy
        self._hierarchy = [Operator(V_pressure,V_velocity,
                           self.omega,self.ignore_mass_lumping)
                           for (V_pressure,V_velocity) in zip(self.V_pressure_hierarchy,
                                                              self.V_velocity_hierarchy)]

    def __getitem__(self,level):
        '''Return operator on given level in the functionspace hierarchy.

        :arg level: level in hierarchy
        '''
        return self._hierarchy[level]

    def __len__(self):
        '''Return number of levels in operator hierarchy.'''
        return len(self._hierarchy)

