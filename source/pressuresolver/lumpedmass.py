import os
import numpy as np
from mpi4py import MPI
from firedrake import *
from firedrake.ffc_interface import compile_form
import xml.etree.cElementTree as ET

class FullMass(object):
    '''Class for full velocity mass matrix implemented in UFL.

    Note that this implementation is not very efficient as the mass matrix is
    assembled in the constructor and inverted explicity using the built-in
    PETSc solver when the :func:`divide` method is called.

    :arg V_velocity: Velocity space the mass matrix is built on
    '''
    def __init__(self,V_velocity):
        self.V_velocity = V_velocity
        self.w = TestFunction(self.V_velocity)
        self._dx = self.V_velocity.mesh()._dx
        v = TrialFunction(self.V_velocity)
        self.a_mass = assemble(dot(self.w,v)*self._dx)

    def add_to_xml(self,parent,function):
        '''Add to existing xml tree.

        :arg parent: Parent node to be added to
        :arg function: Function of object
        '''
        e = ET.SubElement(parent,function)
        e.set("type",type(self).__name__)
        v_str = self.V_velocity.ufl_element()._short_name
        v_str += str(self.V_velocity.ufl_element().degree())
        e.set("velocity_space",v_str)
            
    def multiply(self,u):
        '''Multiply by mass matrix

        In-place multiply a velocity field by the mass matrix.

        :arg u: velocity field to multiply (will be modified in-place)
        '''
        u_out = assemble(dot(self.w,u)*self._dx)
        u.assign(u_out)
            
    def divide(self,u):
        '''Divide by mass matrix

        In-place divide a velocity field by the mass matrix. Note that this
        requires a global solve.

        :arg u: velocity field to divide (will be modified in-place)
        '''
        u_out = Function(self.V_velocity)
        solve(self.a_mass, u_out, u,
              solver_parameters={'ksp_type': 'cg',
                                 'pc_type':'jacobi'})
        u.assign(u_out)

class LumpedMass(object):
    '''Class for lumped velocity mass matrix.

    The lumped mass matrix provides some approximation to the full mass
    matrix implemented in :class:`FullMass` which is cheaper to invert
    (in particular this does not require a global solve), but is less
    accurate. Here we use the diagonal elements of the full mass matrix

    :arg V_velocity: Velocity space
    '''
    def __init__(self,V_velocity):
        self.V_velocity = V_velocity
        self.mesh = self.V_velocity.mesh()
        self._dx = self.V_velocity.mesh()._dx
        self.project_solver_param = {'ksp_type':'cg',
                                     'pc_type':'jacobi'}
        nlocaldof = self.V_velocity.cell_node_map().arity 

        V_cells = FunctionSpace(self.mesh,'DG',0)

        u = TestFunction(self.V_velocity)
        v = TrialFunction(self.V_velocity)

        # Build local stencil of full mass matrix
        mass = dot(u,v)*self._dx
        compiled_form = compile_form(mass, 'mass')[0]
        mass_kernel = compiled_form[6]
        coords = compiled_form[3]
        coefficients = compiled_form[4]
        arguments = mass.arguments()
        mass_matrix = Function(V_cells, val=op2.Dat(V_cells.node_set**(nlocaldof**2)))
        args = [mass_matrix.dat(op2.INC, mass_matrix.cell_node_map()[op2.i[0]]),
                coords.dat(op2.READ,coords.cell_node_map(),flatten=True)]
        for c in coefficients:
            args.append(c.dat(op2.READ, c.cell_node_map(), flatten=True))
        op2.par_loop(mass_kernel,mass_matrix.cell_set,*args)

        self._data = Function(self.V_velocity)

        assemble_diag_kernel = '''void assemble_diag(double **mass_matrix,
                                                     double **lumped_mass_matrix) {
          for (int i=0; i<%(nlocaldof)d; ++i) {
            lumped_mass_matrix[i][0] += mass_matrix[0][(%(nlocaldof)d+1)*i];
          }
        }'''

        assemble_diag_kernel = op2.Kernel(assemble_diag_kernel % {'nlocaldof':nlocaldof},
                                          'assemble_diag')
        op2.par_loop(assemble_diag_kernel,
                     mass_matrix.cell_set,
                     mass_matrix.dat(op2.READ, mass_matrix.cell_node_map()),
                     self._data.dat(op2.INC, self._data.cell_node_map()))
        # Construct pointwise inverse
        self._data_inv = Function(self.V_velocity)
        kernel_inv = '*data_inv = 1./(*data);'
        par_loop(kernel_inv,direct,
                 {'data_inv':(self._data_inv,WRITE),
                  'data':(self._data,READ)})

    def add_to_xml(self,parent,function):
        '''Add to existing xml tree.

        :arg parent: Parent node to be added to
        :arg function: Function of object
        '''
        e = ET.SubElement(parent,function)
        e.set("type",type(self).__name__)
        v_str = self.V_velocity.ufl_element()._short_name
        v_str += str(self.V_velocity.ufl_element().degree())
        e.set("velocity_space",v_str)

    def test_kinetic_energy(self):
        '''Check how exactly the kinetic energy can be represented with
        lumped mass matrix.

        Calculate :math:`E_{kin}= \\frac{1}{2} \int \\vec{u}\cdot \\vec{u}\;dx`
        for a solid body rotation field :math:`(0,-z,y)` both with
        the full- and with the lumped mass matrix and compare the results.
        '''
        u_SBR = Function(self.V_velocity)
        u_SBR.project(Expression(('0','-x[2]','x[1]')),
                      solver_parameters=self.project_solver_param)
        energy_full = assemble(dot(u_SBR,u_SBR)*self._dx)
        Mu_SBR = Function(self.V_velocity)
        Mu_SBR.assign(u_SBR)
        self.multiply(Mu_SBR)
        energy_lumped = u_SBR.dat.inner(Mu_SBR.dat)
        energy_exact = 4.*pi/3.
        energy_lumped *= 0.5
        energy_full *= 0.5
        comm = MPI.COMM_WORLD
        if (comm.Get_rank() == 0):
            print 'kinetic energy = '+('%18.12f' % energy_exact)+' [exact]'
            print '                 '+('%18.12f' % energy_full)+ \
                  ' (error = '+('%6.4f' % (100.*abs(energy_full/energy_exact-1.)))+'%)' \
                  ' [full mass matrix]'
            print '                 '+('%18.12f' % energy_lumped)+ \
                  ' (error = '+('%6.4f' % (100.*abs(energy_lumped/energy_exact-1.)))+'%)' \
                  ' [lumped mass matrix]'
            
    def multiply(self,u):
        '''Multiply by lumped mass matrix

        In-place multiply a velocity field by the lumped mass matrix

        :arg u: velocity field to multiply (will be modified in-place)
        '''
        self._matmul(self._data,u)
            

    def divide(self,u):
        '''Divide by lumped mass matrix

        In-place divide a velocity field by the lumped mass matrix

        :arg u: velocity field to divide (will be modified in-place)
        '''
        self._matmul(self._data_inv,u)

    def get(self):
        '''Return field representation of mass matrix.'''
        return self._data

    def _matmul(self,m,u):
        '''Multiply by diagonal matrix

        In-place multiply a velocity field by a diagonal matrix, which 
        is either the lumped mass matrix or its inverse.

        :arg m: block-diagonal matrix to multiply with
        :arg u: velocity field to multiply (will be modified in-place)
        '''
        kernel = '(*u) *= (*m);'
        par_loop(kernel,direct,
                 {'u':(u,RW),
                  'm':(m,READ)})
