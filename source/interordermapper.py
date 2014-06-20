from firedrake import *

class Mapper(object):
    '''Map between different finite element spaces.
    
    Provide methods for projecting between high- and low order finite element spaces.
    Currently only supports some special spaces
    
    :arg V_high: High order space (can be DG1 or BDFM1)
    :arg V_low: Lower order space (can be DG0 or RT1)
    '''
    def __init__(self,V_high,V_low):
        self.dx = V_high.mesh()._dx
        self.V_high = V_high
        self.V_low = V_low
        self.psi_high = TestFunction(self.V_high)
        self.psi_low = TestFunction(self.V_low)
        self.phi_high = TrialFunction(self.V_high)
        self.phi_low = TrialFunction(self.V_low)
        self.a_mass_high = dot(self.phi_high,self.psi_high)*self.dx
        self.a_mass_low = dot(self.phi_low,self.psi_low)*self.dx

    def restrict(self,f):
        '''Restrict to lower order space.
    
        Project a function in the higher order space onto the lower order space
    
        :arg f: Function in higher order space
        '''
        f_low = Function(self.V_low)
        L = dot(self.psi_low,f)*self.dx
        solve(self.a_mass_low == L,f_low)
        return f_low

    def prolong(self,f):
        '''Prolongate to higher order space.
    
        Project a function in the lower order space onto the higher order space
    
        :arg f: Function in lower order space
        '''
        f_high = Function(self.V_high)
        L = dot(self.psi_high,f)*self.dx
        solve(self.a_mass_high == L,f_high)
        return f_high
