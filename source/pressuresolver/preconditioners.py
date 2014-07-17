from operators import *

class hMultigrid(object):
    '''Geometric Multigrid preconditioner with h-coarsening only.

    Solve approximately using a multigrid V-cycle. The operator, pre-, post-
    smoother and coarse grid operator are passed as arguments, this allows
    tuning of the number of smoothing steps etc.

    :arg operator_hierarchy: Schur complement :class:`Operator` s on the
        different multigrid levels.

    :arg presmoother_hierarchy: Presmoother on different multigrid levels
    :arg postsmoother_hierarchy: Postsmoother on different multigrid levels
    :arg coarsegrid_solver: Solver object for coarse grid equation
    ''' 
    def __init__(self,operator_hierarchy,
                 presmoother_hierarchy,
                 postsmoother_hierarchy,
                 coarsegrid_solver):
        self.operator_hierarchy = operator_hierarchy
        self.presmoother_hierarchy = presmoother_hierarchy
        self.postsmoother_hierarchy = postsmoother_hierarchy
        self.coarsegrid_solver = coarsegrid_solver
        self.V_pressure_hierarchy = self.operator_hierarchy.V_pressure_hierarchy
        self.residual = FunctionHierarchy(self.V_pressure_hierarchy)
        self.rhs = FunctionHierarchy(self.V_pressure_hierarchy)
        self.phi = FunctionHierarchy(self.V_pressure_hierarchy)
        self.fine_level = len(self.V_pressure_hierarchy)-1
        self.coarsest_level = 0
        self.dx = [self.V_pressure_hierarchy[level].mesh()._dx
                   for level in range(len(self.V_pressure_hierarchy))]
        self.operator = operator_hierarchy[self.fine_level] 

    def vcycle(self,level=None):
        '''Recursive multigrid V-cycle.
    
        :arg level: multigrid level, if None, start on finest level.
        '''
        if (level == None):
            level = self.fine_level
        # Solve exactly on coarsest level
        if (level == self.coarsest_level):
            # presmooth
            self.coarsegrid_solver.solve(self.rhs[level],self.phi[level])
        else:
        # Recursion on all other levels
            # Only initialise solution to zero on the coarser levels
            initial_phi_is_zero = not (level == self.fine_level)
            # Presmoother
            self.presmoother_hierarchy[level].smooth(self.rhs[level],
                                                     self.phi[level],
                                                     initial_phi_is_zero=initial_phi_is_zero)
            self.residual[level].assign(self.operator_hierarchy[level].residual(self.rhs[level],
                                                                                self.phi[level]))
            # Restrict residual to RHS on coarser level
            self.residual.restrict(level)
            self.rhs[level-1].assign(4.*self.residual[level-1])
            # Recursive call
            self.vcycle(level-1)
            # Prolongate and add coarse grid correction
            self.residual[level-1].assign(self.phi[level-1])
            self.residual.prolong(level-1)
            self.phi[level].assign(self.residual[level]+self.phi[level])
            # Postsmooth
            self.postsmoother_hierarchy[level].smooth(self.rhs[level],
                                                      self.phi[level],
                                                      initial_phi_is_zero=False)

    def solve(self,b,phi):
        '''Solve approximately.

        Solve the pressure correction equation approximately for a given right
        hand side :math:`b` with a V-cycle. Note that the state vector is
        updated in place.

        :arg b: right hand side in pressure space
        :arg phi: State :math:`\phi` in pressure space.
        '''
        self.phi[self.fine_level].assign(0.0)
        self.rhs[self.fine_level].assign(b)
        self.vcycle()
        phi.assign(self.phi[self.fine_level])

class hpMultigrid(object):
    '''Geometric Multigrid preconditioner with hp- coarsening.

    Solve approximately using a multigrid V-cycle. The first coarsening
    step is p-coarsening, i.e. coarsen from the higher order finite
    element space to the lowest order space. An instance of :class:`hMultigrid`
    is passed as an argument.

    :arg hmultigrid: Instance of :class:`hMultigrid`, which is used on the
        lowest order pressure space hierarchy. 
    :arg operator: Helmholtz operator on higher order space
    :arg presmoother: Presmoother on higher order space
    :arg postsmoother: Postsmoother on higher order space
    ''' 
    def __init__(self,
                 hmultigrid,
                 operator,
                 presmoother,
                 postsmoother):
        self.hmultigrid = hmultigrid
        self.operator = operator
        self.presmoother = presmoother
        self.postsmoother = postsmoother
        self.V_pressure = self.operator.V_pressure
        self.V_pressure_low = self.hmultigrid.V_pressure_hierarchy[-1]
        self.rhs_low = Function(self.V_pressure_low)
        self.dphi = Function(self.V_pressure)
        self.dphi_low = Function(self.V_pressure_low)
        self.dx = self.V_pressure.mesh()._dx
        self.psi = TestFunction(self.V_pressure)
        self.psi_low = TestFunction(self.V_pressure_low)
        self.a_mass = TrialFunction(self.V_pressure)*self.psi*self.dx
        self.a_mass_low = TrialFunction(self.V_pressure_low)*self.psi_low*self.dx

    def solve(self,b,phi):
        '''Solve approximately.

        Solve the pressure correction equation approximately for a given right
        hand side :math:`b` with a V-cycle. Note that the state vector is
        updated in place.

        :arg b: right hand side in pressure space
        :arg phi: State :math:`\phi` in pressure space.
        '''
        phi.assign(0.0)
        # Presmooth
        self.presmoother.smooth(b,phi,initial_phi_is_zero=True)
        # Calculuate residual...
        self.residual = self.operator.residual(b,phi)
        # ... and restrict to RHS in lowest order space
        self.restrict(self.residual,self.rhs_low)
        # h-multigrid in lower order space
        self.hmultigrid.solve(self.rhs_low,self.dphi_low)
        # Prolongate correction back to higher order space...
        self.prolong(self.dphi_low,self.dphi)
        # ... and add to solution in higher order space
        phi.assign(phi+self.dphi)
        # Postsmooth
        self.postsmoother.smooth(b,phi,initial_phi_is_zero=False)

    def restrict(self,phi_high, phi_low):
        '''Restrict to lower order space.
    
        Project a function in the higher order space onto the lower order space
    
        :arg phi_high: Function in higher order space
        :arg phi_low: Resulting function in lower order space
        '''
        L = self.psi_low*phi_high*self.dx
        solve(self.a_mass_low == L,phi_low)

    def prolong(self,phi_low, phi_high):
        '''Prolongate to higher order space.
    
        Project a function in the lower order space onto the higher order space
    
        :arg phi_low: Function in lower order space
        :arg phi_hight: Resulting function in lower order space
        '''
        L = self.psi*phi_low*self.dx
        solve(self.a_mass == L,phi_high)

