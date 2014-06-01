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
    def __init__(self,V_velocity,ignore_lumping=False):
        self.ignore_lumping = ignore_lumping
        self.V_velocity = V_velocity
        one_velocity = Function(self.V_velocity)
        one_velocity.assign(1.0)
        self.data = assemble(inner(TestFunction(self.V_velocity),one_velocity)*dx)
        self.data_inv = Function(self.V_velocity)
        kernel_inv = '{ data_inv[0][0] = 1./data[0][0]; }'
        for facet in (dS,ds):
            par_loop(kernel_inv,facet,{'data_inv':(self.data_inv,WRITE),
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
            kernel_inv = '{ u[0][0] *= data[0][0]; }'
            for facet in (dS,ds):
                par_loop(kernel,facet,{'u':(u,RW),
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
            kernel_inv = '{ u[0][0] *= data_inv[0][0]; }'
            for facet in (dS,ds):
                par_loop(kernel_inv,facet,{'u':(u,RW),
                                           'data_inv':(self.data_inv,READ)}) 
