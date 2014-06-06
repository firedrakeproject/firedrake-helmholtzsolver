from operators import *

##########################################################
# Jacobi smoother
##########################################################
class Jacobi(object):

##########################################################
# Constructor
##########################################################
    def __init__(self,operator,
                 mu_relax=2./3.,
                 n_smooth=1):
        self.operator = operator
        self.V_pressure = self.operator.V_pressure
        self.mu_relax = mu_relax
        self.n_smooth = n_smooth
        self.dx = self.operator.V_pressure.mesh()._dx
        # Construct lumped mass matrix
        self.lumped_mass = self.operator.lumped_mass
        self._build_D_diag()

##########################################################
# Build diagonal matrix for smoother
##########################################################
    def _build_D_diag(self):
        # Construct inverse matrix for smoother
        one_pressure = Function(self.V_pressure)
        one_pressure.assign(1.0)
        D_diag = assemble(TestFunction(self.V_pressure)*one_pressure*self.dx)
        kernel_add_vterm = 'for(int i=0; i<M_u_lumped.dofs; i++) {D_diag[0][0] += 2./M_u_lumped[i][0];}'
        M_u_lumped = self.lumped_mass.get()
        par_loop(kernel_add_vterm,self.dx,{'D_diag':(D_diag,INC),'M_u_lumped':(M_u_lumped,READ)})
        kernel_inv = '{ D_diag_inv[0][0] = 1./D_diag[0][0]; }'
        self.D_diag_inv = Function(self.V_pressure)
        par_loop(kernel_inv,self.dx,{'D_diag_inv':(self.D_diag_inv,WRITE),
                                'D_diag':(D_diag,READ)})
       
##########################################################
# Solve approximately
##########################################################
    def solve(self,b,phi):
        phi.assign(0.0)
        self.smooth(b,phi)

##########################################################
# Solve approximately
##########################################################
    def solveApprox(self,b,phi):
        phi.assign(0.0)
        self.smooth(b,phi)

##########################################################
# Apply smoother according to
# 
# phi -> phi + 2*mu*D^{-1}*residual(b,phi)
#
##########################################################
    def smooth(self,b,phi):
        r = Function(self.V_pressure)
        for i in range(self.n_smooth):
            if (i==0):
                r.assign(b)
            else:
                r.assign(self.operator.residual(b,phi))
            # Apply inverse diagonal r_i -> D^{-1}_ii *r_i
            kernel_inv_diag = '{ r[0][0] *= D_diag_inv[0][0]; }'
            par_loop(kernel_inv_diag,self.dx,{'r':(r,RW),'D_diag_inv':(self.D_diag_inv,READ)})
            # Update phi 
            phi += 2.*self.mu_relax*r

##########################################################
# Jacobi hierarchy
##########################################################
class JacobiHierarchy(object):

##########################################################
# Constructor
##########################################################
    def __init__(self,operator_hierarchy,
                 mu_relax=2./3.,
                 n_smooth=1):
        self.operator_hierarchy = operator_hierarchy
        self.mu_relax = mu_relax
        self.n_smooth = n_smooth
        self._hierarchy = [Jacobi(operator,
                                  self.mu_relax,
                                  self.n_smooth)
                           for operator in self.operator_hierarchy]

##########################################################
# Get item
##########################################################
    def __getitem__(self,index):
        return self._hierarchy[index]

##########################################################
# Number of levels
##########################################################
    def __len__(self):
        return len(self._hierarchy)

