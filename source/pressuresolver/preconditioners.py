from operators import *

class Multigrid(object):
    '''Geometric Multigrid preconditioner for linear Schur complement system.

    Solve approximately using a multigrid V-cycle. The operator, pre-, post- smoother and
    coarse grid operator are passed as arguments, this allows tuning of the number of
    smoothing steps etc.

    :arg operator_hierarchy: Schur complement :class:`Operator` s on the different multigrid 
        levels
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
            self.rhs[level-1].assign(self.residual[level-1])
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

        Solve the pressure correction equation approximately for a given right hand side
        :math:`b` with a V-cycle. Note that the state vector is updated in place.

        :arg b: right hand side in pressure space
        :arg phi: State :math:`\phi` in pressure space.
        '''
        self.phi[self.fine_level].assign(phi)
        self.rhs[self.fine_level].assign(b)
        self.vcycle()
        phi.assign(self.phi[self.fine_level])

