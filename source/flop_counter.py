from firedrake import *
import firedrake
from firedrake.ffc_interface import compile_form
from pyop2 import Global
from firedrake.fiat_utils import *
import os
import tempfile

class FlopCounter1Form(object):
    '''FLOP counter for 1-forms of the form <u_{test},....,u>'''
    def __init__(self,ufl_form):
        # Ensure this really is a 1-form
        rank = len(ufl_form.arguments())
        assert(rank==1)
        self._ufl_form = ufl_form
        # Extract function space
        self._fs_to = self._ufl_form.arguments()[0].function_space()

    def _get_ndofcell_to(self):
        '''Count the number of dofs per cell'''
        ufl_ele = self._fs_to.ufl_element()
        # Unwrap non HDiv'd element if necessary
        if isinstance(ufl_ele, HDiv):
            ufl_element = ufl_ele._element
        else:
            ufl_element = ufl_ele
        tdim = ufl_element.cell().topological_dimension()
        element = fiat_from_ufl_element(ufl_element)
        ndof_cell = 0
        for entity_maps in element.entity_dofs().values():
            for entity in entity_maps.values():
                ndof_cell += len(entity)
        return ndof_cell

    @property
    def flops(self):
        '''Count the number of FLOPs'''
        compiled_form = compile_form(self._ufl_form, 'ufl_form')[0]
        kernel = compiled_form[6]
        # Construct a new firedrake_geometry.h file in which
        # double is replaced by LoggedDouble and store in temporary directory
        firedrake_geometry_file = open(os.path.dirname(firedrake.__file__)+'/firedrake_geometry.h')
        contents = firedrake_geometry_file.read()
        firedrake_geometry_file.close()
        tmp_dir = tempfile.mkdtemp()
        firedrake_geometry_file_logged = open(tmp_dir+'/firedrake_geometry_LOGGED.h','w')
        print >> firedrake_geometry_file_logged, contents.replace('double','LoggedDouble')
        firedrake_geometry_file_logged.close()
        # Add temporary directory to kernel include directories
        kernel._include_dirs.append(tmp_dir)
        coords = compiled_form[3]
        coefficients = compiled_form[4]
        # Function
        u = coefficients[0]
        # Construct dummy global with local unknowns
        local_dofs = op2.Global(self._get_ndofcell_to())

        # Construct first two arguments
        # 1st one is dummy with local unknowns, 2nd is the coordinates
        args = [local_dofs(op2.INC),
                coords.dat(op2.READ, coords.cell_node_map(), flatten=True)]
        # Append all other arguments
        for c in coefficients:
            args.append(c.dat(op2.READ, c.cell_node_map(), flatten=True))

        # Build ParLoop object and extract number of FLOPs
        par_loop = op2.par_loop(kernel,u.cell_set, *args,measure_flops=True)
        nflop = par_loop.total_flops
        # Delete temporary directory with firedrake_geomtry.h
        os.remove(tmp_dir+'/firedrake_geometry_LOGGED.h')
        os.rmdir(tmp_dir)
        return nflop
