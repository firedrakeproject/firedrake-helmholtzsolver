from operators import *

##########################################################
# Multigrid solver
##########################################################

class Multigrid(object):
    
##########################################################
# Constructor
##########################################################
    def __init__(self,operator_hierarchy,
                 presmoother_hierarchy,
                 postsmoother_hierarchy,
                 coarsegrid_solver,
                 maxiter=100,
                 tolerance=1.E-6):
        self.operator_hierarchy = operator_hierarchy
        self.presmoother_hierarchy = presmoother_hierarchy
        self.postsmoother_hierarchy = postsmoother_hierarchy
        self.coarsegrid_solver = coarsegrid_solver
        self.V_pressure_hierarchy = self.operator_hierarchy.V_pressure_hierarchy
        self.maxiter = maxiter
        self.tolerance = tolerance
        self.residual = FunctionHierarchy(self.V_pressure_hierarchy)
        self.rhs = FunctionHierarchy(self.V_pressure_hierarchy)
        self.phi = FunctionHierarchy(self.V_pressure_hierarchy)
        self.fine_level = len(self.V_pressure_hierarchy)-1
        self.coarsest_level = 0
        self.dx = [self.V_pressure_hierarchy[level].mesh()._dx
                   for level in range(len(self.V_pressure_hierarchy))]
        self.operator = operator_hierarchy[self.fine_level] 

##########################################################
# VCycle
##########################################################
    def vcycle(self,level=None):
        if (level == None):
            level = self.fine_level
        # Solve exactly on coarsest level
        if (level == self.coarsest_level):
            # presmooth
            self.coarsegrid_solver.solve(self.rhs[level],self.phi[level])
        else:
        # Recursion on all other levels
            # Initialise solution to zero on the coarser levels
            if (level != self.fine_level):
                self.phi[level].assign(0.0)
            # Presmoother
            self.presmoother_hierarchy[level].smooth(self.rhs[level],
                                                     self.phi[level])
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
                                                      self.phi[level])

##########################################################
# Solve
##########################################################
    def solveApprox(self,b,phi):
        self.phi[self.fine_level].assign(phi)
        self.rhs[self.fine_level].assign(b)
        self.vcycle()
        phi.assign(self.phi[self.fine_level])

