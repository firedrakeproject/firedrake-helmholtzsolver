from firedrake import *

##########################################################
# Class for lumped mass matrix
##########################################################
class LumpedMass(object):

##########################################################
# Constructor
#
# set ignore_lumping = True to override mass lumping for 
# testing.
##########################################################
    def __init__(self,V_velocity,ignore_lumping=False,use_SBR=True):
        self.ignore_lumping = ignore_lumping
        self.V_velocity = V_velocity
        self.use_SBR = use_SBR
        if (self.use_SBR):
            w = TestFunction(self.V_velocity)
            self.data = Function(self.V_velocity)
            SBR_x = Function(self.V_velocity).project(Expression(('0','-x[2]','x[1]')))
            SBR_y = Function(self.V_velocity).project(Expression(('x[2]','0','-x[0]')))
            SBR_z = Function(self.V_velocity).project(Expression(('-x[1]','x[0]','0')))
            M_SBR_x = assemble(dot(w,SBR_x)*dx)
            M_SBR_y = assemble(dot(w,SBR_y)*dx)
            M_SBR_z = assemble(dot(w,SBR_z)*dx)
            kernel_code = '''void lump_SBR(double *data,
                                           double *SBR_x,
                                           double *SBR_y,
                                           double *SBR_z,
                                           double *M_SBR_x,
                                           double *M_SBR_y,
                                           double *M_SBR_z) {
                              *data = (  (*SBR_x)*(*M_SBR_x) 
                                       + (*SBR_y)*(*M_SBR_y)
                                       + (*SBR_z)*(*M_SBR_z) ) / 
                                      (  (*SBR_x)*(*SBR_x) 
                                       + (*SBR_y)*(*SBR_y)
                                       + (*SBR_z)*(*SBR_z) );
                            }
            '''
            kernel = op2.Kernel(kernel_code,"lump_SBR")
            op2.par_loop(kernel,
                         self.data.dof_dset.set,
                         self.data.dat(op2.WRITE),
                         SBR_x.dat(op2.READ),
                         SBR_y.dat(op2.READ),
                         SBR_z.dat(op2.READ),
                         M_SBR_x.dat(op2.READ),
                         M_SBR_y.dat(op2.READ),
                         M_SBR_z.dat(op2.READ))
        else: 
            one_velocity = Function(self.V_velocity)
            one_velocity.assign(1.0)
            self.data = assemble(inner(TestFunction(self.V_velocity),one_velocity)*dx)
        self.data_inv = Function(self.V_velocity)
        kernel_inv_code = '''void invert(double *data_inv, double *data) {
                               *data_inv = 1./(*data); 
                             }
        '''
        kernel_inv = op2.Kernel(kernel_inv_code,'invert')
        op2.par_loop(kernel_inv,
                     self.data_inv.dof_dset.set,
                     self.data_inv.dat(op2.WRITE),
                     self.data.dat(op2.READ))

##########################################################
# Extract field vector
##########################################################
    def get(self):
        return self.data

##########################################################
# Multiply a field by the lumped mass matrix
##########################################################
    def multiply(self,u):
        if (self.ignore_lumping):
            psi = TestFunction(self.V_velocity)
            w = assemble(dot(self.w,u)*dx)
            u.assign(w)
        else:
            kernel_code = '''void multiply(double *u, double *data) {
                               (*u) *= (*data);
                             }
            '''
            kernel = op2.Kernel(kernel_code,'multiply')
            op2.par_loop(kernel,
                         u.dof_dset.set,
                         u.dat(op2.RW),
                         self.data.dat(op2.READ))

##########################################################
# Divide a field by the lumped mass matrix
##########################################################
    def divide(self,u):
        if (self.ignore_lumping):
            psi = TestFunction(self.V_velocity)
            phi = TrialFunction(self.V_velocity)
            a_mass = assemble(dot(psi,phi)*dx)
            w = Function(self.V_velocity)
            solve(a_mass, w, u)
            u.assign(w)
        else:
            kernel_code = '''void divide(double *u, double *data_inv) {
                               (*u) *= (*data_inv);
                             }
            '''
            kernel = op2.Kernel(kernel_code,'divide')
            op2.par_loop(kernel,
                         u.dof_dset.set,
                         u.dat(op2.RW),
                         self.data_inv.dat(op2.READ))
