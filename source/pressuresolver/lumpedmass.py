from firedrake import *

class LumpedMass(object):
    '''Lumped velocity mass matrix.
    
    This class constructs a diagonal lumped velocity mass matrix :math:`M_u^*` 
    in the :math:`RT1` space and provides methods for multiplying and dividing 
    :math:`RT1` functions by this lumped mass matrix. Internally the mass matrix
    is represented as a :math:`RT1` field.

    Currently, two methods for mass lumping are supported and can be chosen by 
    the parameter :class:`use_SBR`:

    * Lumped mass matrix is exact when acting on constant fields:

        .. math::
        
            M_u^* C = M_u C

        where C is constant.

    * On each edge e the lumped mass matrix is exact when acting on a solid body rotation
        field that has maximal flux through this edge. Mathematically this means that

        .. math::

            (M_u^*)_{ee} = \\frac{\sum_{i=1}^3 V^{(i)}_e U^{(i)}_e }{\sum_{i=1}^3 (U^{(i)}_e)^2}

        where :math:`U^{(i)}` is a solid body rotation field around coordinate axis :math:`i`
        and :math:`V^{(i)} = M_u U^{(i)}`

    :arg V_velocity: Velocity space, currently only :math:`RT1` is supported.
    :arg ignore_lumping: For debugging, this can be set to true to use the full mass
        matrix in the :class:`multiply()` and :class:`divide()` methods.
    :arg use_SBR: Use mass lumping based on solid body rotation fields.
    '''
    def __init__(self,V_velocity,ignore_lumping=False,use_SBR=True):
        self.ignore_lumping = ignore_lumping
        self.V_velocity = V_velocity
        self.dx = self.V_velocity.mesh()._dx
        self.use_SBR = use_SBR
        if (self.use_SBR):
            w = TestFunction(self.V_velocity)
            self.data = Function(self.V_velocity)
            SBR_x = Function(self.V_velocity).project(Expression(('0','-x[2]','x[1]')))
            SBR_y = Function(self.V_velocity).project(Expression(('x[2]','0','-x[0]')))
            SBR_z = Function(self.V_velocity).project(Expression(('-x[1]','x[0]','0')))
            M_SBR_x = assemble(dot(w,SBR_x)*self.dx)
            M_SBR_y = assemble(dot(w,SBR_y)*self.dx)
            M_SBR_z = assemble(dot(w,SBR_z)*self.dx)
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
            self.data = assemble(inner(TestFunction(self.V_velocity),one_velocity)*self.dx)
        self.data_inv = Function(self.V_velocity)
        kernel_inv = '*data_inv = 1./(*data);'
        par_loop(kernel_inv,direct,
                 {'data_inv':(self.data_inv,WRITE),
                  'data':(self.data,READ)})

    def get(self):
        '''Return :math:`RT1` representation of mass matrix.'''
        return self.data

    def multiply(self,u):
        '''Multiply velocity field by lumped mass matrix.

        Use a direct loop to multiply a function in velocity space by the lumped mass matrix.
        Note that this is an in-place operation on the input data.

        :arg u: Velocity field to be multiplied
        '''
        if (self.ignore_lumping):
            psi = TestFunction(self.V_velocity)
            w = assemble(dot(self.w,u)*self.dx)
            u.assign(w)
        else:
            kernel = '(*u) *= (*data);'
            par_loop(kernel,direct,
                     {'u':(u,RW),
                      'data':(self.data,READ)})

    def divide(self,u):
        '''Divide velocity field by lumped mass matrix.

        Use a direct loop to divide a function in velocity space by the lumped mass matrix.
        Note that this is an in-place operation on the input data.
        If the lumping is ignored, the division is implemented by a mass matrix solve.

        :arg u: Velocity field to be divided
        '''
        if (self.ignore_lumping):
            psi = TestFunction(self.V_velocity)
            phi = TrialFunction(self.V_velocity)
            a_mass = assemble(dot(psi,phi)*self.dx)
            w = Function(self.V_velocity)
            solve(a_mass, w, u)
            u.assign(w)
        else:
            kernel = '(*u) *= (*data_inv);'
            par_loop(kernel,direct,
                     {'u':(u,RW),
                      'data_inv':(self.data_inv,READ)})

