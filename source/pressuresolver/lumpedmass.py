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
            kernel = '''*data = (  (*SBR_x)*(*M_SBR_x) 
                                 + (*SBR_y)*(*M_SBR_y)
                                 + (*SBR_z)*(*M_SBR_z) ) / 
                                (  (*SBR_x)*(*SBR_x) 
                                 + (*SBR_y)*(*SBR_y)
                                 + (*SBR_z)*(*SBR_z) );
            '''
            par_loop(kernel,direct,
                         {'data':(self.data,WRITE),
                          'SBR_x':(SBR_x,READ),
                          'SBR_y':(SBR_y,READ),
                          'SBR_z':(SBR_z,READ),
                          'M_SBR_x':(M_SBR_x,READ),
                          'M_SBR_y':(M_SBR_y,READ),
                          'M_SBR_z':(M_SBR_z,READ)})
        else: 
            one_velocity = Function(self.V_velocity)
            one_velocity.assign(1.0)
            self.data = assemble(inner(TestFunction(self.V_velocity),one_velocity)*dx)
        self.data_inv = Function(self.V_velocity)
        kernel_inv = '*data_inv = 1./(*data);'
        par_loop(kernel_inv,direct,
                 {'data_inv':(self.data_inv,WRITE),
                  'data':(self.data,READ)})

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
            kernel = '(*u) *= (*data);'
            par_loop(kernel,direct,
                     {'u':(u,RW),
                      'data':(self.data,READ)})

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
            kernel = '(*u) *= (*data_inv);'
            par_loop(kernel,direct,
                     {'u':(u,RW),
                      'data_inv':(self.data_inv,READ)})

