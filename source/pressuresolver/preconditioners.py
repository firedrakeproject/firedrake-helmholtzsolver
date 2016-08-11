import xml.etree.cElementTree as ET
from firedrake import *
from firedrake.petsc import PETSc
from pyop2.profiling import timed_function, timed_region

class hMultigrid(object):
    '''Geometric Multigrid preconditioner with h-coarsening only.

    Solve approximately using a multigrid V-cycle. The operator, pre-, post-
    smoother and coarse grid solver are passed as arguments, this allows
    tuning of the number of smoothing steps etc.
    The smoothers can for example be a :class:`.Jacobi` instance.

    :arg W_3_hierarchy: Hierarchy of pressure spaces to solve on
    :arg operator_hierarchy: Schur complement :class:`.Operator_Hhat` s on the
        different multigrid levels.
    :arg presmoother_hierarchy: Presmoother on different multigrid levels
    :arg postsmoother_hierarchy: Postsmoother on different multigrid levels
    :arg coarsegrid_solver: Solver object for coarse grid equation
    ''' 
    def __init__(self,
                 W3_hierarchy,
                 operator_hierarchy,
                 presmoother_hierarchy,
                 postsmoother_hierarchy,
                 coarsegrid_solver):
        self._operator_hierarchy = operator_hierarchy
        self._presmoother_hierarchy = presmoother_hierarchy
        self._postsmoother_hierarchy = postsmoother_hierarchy
        self._coarsegrid_solver = coarsegrid_solver
        self._W3_hierarchy = W3_hierarchy 
        self._residual = FunctionHierarchy(self._W3_hierarchy)
        self._rhs = FunctionHierarchy(self._W3_hierarchy)
        self._phi = FunctionHierarchy(self._W3_hierarchy)
        self._fine_level = len(self._W3_hierarchy)-1
        self._coarsest_level = 0
        self._dx = [dx(domain=self._W3_hierarchy[level].mesh())
                    for level in range(len(self._W3_hierarchy))]
        self._operator = operator_hierarchy[self._fine_level] 
        with self._rhs[self._fine_level].dat.vec as v:
            ndof = self._W3_hierarchy[self._fine_level].dof_dset.size
            self._iset = PETSc.IS().createStride(ndof,
                                                 first=v.owner_range[0],
                                                 step=1,
                                                 comm=v.comm)

    def add_to_xml(self,parent,function):
        '''Add to existing xml tree.

        :arg parent: Parent node to be added to
        :arg function: Function of object
        '''
        e = ET.SubElement(parent,function)
        e.set("type",type(self).__name__)
        self._operator_hierarchy.add_to_xml(e,'operator_hierarchy')
        self._presmoother_hierarchy.add_to_xml(e,'presmoother_hierarchy')
        self._postsmoother_hierarchy.add_to_xml(e,'postsmoother_hierarchy')
        self._coarsegrid_solver.add_to_xml(e,'coarse_grid_solver')

    def vcycle(self,level=None):
        '''Recursive implementation of multigrid V-cycle.
    
        :arg level: multigrid level, if None, start on finest level.
        '''
        if (level == None):
            level = self._fine_level
        with timed_region("vcycle_level_"+str(level)):
            # Solve exactly on coarsest level
            if (level == self._coarsest_level):
                # presmooth
                self._coarsegrid_solver.solve(self._rhs[level],self._phi[level])
            else:
                # Recursion on all other levels
                # Only initialise solution to zero on the coarser levels
                # Presmoother
                self._presmoother_hierarchy[level].smooth(self._rhs[level],
                  self._phi[level],
                  initial_phi_is_zero=True)
                self._residual[level].assign(self._operator_hierarchy[level].residual(
                  self._rhs[level],
                  self._phi[level]))
                # Restrict residual to RHS on coarser level
                self._residual.restrict(level)
                self._rhs[level-1].assign(self._residual[level-1])
                # Recursive call
                self.vcycle(level-1)
                # Prolongate and add coarse grid correction
                self._residual[level-1].assign(self._phi[level-1])
                self._residual.prolong(level-1)
                self._phi[level].assign(self._residual[level]+self._phi[level])
                # Postsmooth
                self._postsmoother_hierarchy[level].smooth(self._rhs[level],
                                                           self._phi[level],
                                                           initial_phi_is_zero=False)

    def solve(self,b,phi):
        '''Solve approximately.

        Solve the pressure correction equation approximately for a given right
        hand side :math:`b` with a V-cycle. Note that the state vector is
        updated in place.

        :arg b: right hand side in pressure space
        :arg phi: State :math:`\phi` in pressure space.
        '''
        self._phi[self._fine_level].assign(0.0)
        self._rhs[self._fine_level].assign(b)
        self.vcycle()
        phi.assign(self._phi[self._fine_level])

    def apply(self,pc,x,y):
        '''PETSc interface for preconditioner solve.

        PETSc interface wrapper for the :func:`solve` method.

        :arg x: PETSc vector representing the right hand side in pressure
            space
        :arg y: PETSc vector representing the solution pressure space.
        '''
        with self._rhs[self._fine_level].dat.vec as v:
            tmp = x.getSubVector(self._iset)
            x.copy(v)
            x.restoreSubVector(self._iset, tmp)
        self._phi[self._fine_level].assign(0.0)
        self.vcycle()
        with self._phi[self._fine_level].dat.vec_ro as v:
            y.array[:] = v.array[:]

class hpMultigrid(object):
    '''Geometric Multigrid preconditioner with hp- coarsening.

    Solve approximately using a multigrid V-cycle. The first coarsening
    step is p-coarsening, i.e. coarsen from the higher order finite
    element space to the lowest order space. An instance of :class:`hMultigrid`
    is passed as an argument.

    :arg hmultigrid: Instance of :class:`hMultigrid`, which is used on the
        lowest order pressure space hierarchy. 
    :arg operator: Helmholtz operator (instance of :class:`.Operator_Hhat`) on higher
        order space
    :arg presmoother: Presmoother on higher order space
    :arg postsmoother: Postsmoother on higher order space
    ''' 
    def __init__(self,
                 hmultigrid,
                 operator,
                 presmoother,
                 postsmoother):
        self._hmultigrid = hmultigrid
        self._operator = operator
        self._presmoother = presmoother
        self._postsmoother = postsmoother
        self._W3 = self._operator._W3
        self._W3_low = self._hmultigrid._W3_hierarchy[-1]
        self._rhs_low = Function(self._W3_low)
        self._dphi_low = Function(self._W3_low)
        self._dx = dx(domain=self._W3.mesh())
        self._phi_tmp = Function(self._W3)
        self._rhs_tmp = Function(self._W3)
        with self._rhs_tmp.dat.vec as v:
            self._iset = PETSc.IS().createStride(self._W3.dof_dset.size,
                                                 first=v.owner_range[0],
                                                 step=1,
                                                 comm=v.comm)
        ndof = len(self._W3.cell_node_map().values[0])
        d = {'NDOF':ndof,'INV_NDOF':1./float(ndof)}
        self._kernel_restrict = '''
          double tmp = 0.0;
          for(int i=0;i<%(NDOF)d;++i) {
            tmp += phi_high[0][i];
          }
          phi_low[0][0] = tmp;
        ''' % d
        self._kernel_prolongadd = '''
          double tmp = phi_low[0][0];
          for(int i=0;i<%(NDOF)d;++i) {
            phi_high[0][i] += tmp;
          }
        ''' % d

    def add_to_xml(self,parent,function):
        '''Add to existing xml tree.

        :arg parent: Parent node to be added to
        :arg function: Function of object
        '''
        e = ET.SubElement(parent,function)
        e.set("type",type(self).__name__)
        self._operator.add_to_xml(e,'high_order_operator')
        self._presmoother.add_to_xml(e,'high_order_presmoother')
        self._postsmoother.add_to_xml(e,'high_order_postsmoother')
        self._hmultigrid.add_to_xml(e,'hmultigrid')

    def solve(self,b,phi):
        '''Solve approximately.

        Solve the pressure correction equation approximately for a given right
        hand side :math:`b` with a V-cycle. Note that the state vector is
        updated in place.

        :arg b: right hand side in pressure space
        :arg phi: State :math:`\phi` in pressure space.
        '''
        with timed_region("vcycle_level_"+str(self._hmultigrid._fine_level+1)):
            phi.assign(0.0)
            # Presmooth
            self._presmoother.smooth(b,phi,initial_phi_is_zero=True)
            # Calculuate residual...
            self._residual = self._operator.residual(b,phi)
            # ... and restrict to RHS in lowest order space
            self.restrict(self._residual,self._rhs_low)
            # h-multigrid in lower order space
            self._hmultigrid.solve(self._rhs_low,self._dphi_low)
            # Prolongate correction back to higher order space...
            # ... and add to solution in higher order space
            self.prolongadd(self._dphi_low,phi)
            # Postsmooth
            self._postsmoother.smooth(b,phi,initial_phi_is_zero=False)

    def apply(self,pc,x,y):
        '''PETSc interface for preconditioner solve.

        PETSc interface wrapper for the :func:`solve` method.

        :arg x: PETSc vector representing the right hand side in pressure
            space
        :arg y: PETSc vector representing the solution pressure space.
        '''
        
        with self._rhs_tmp.dat.vec as v:
            tmp = x.getSubVector(self._iset)
            x.copy(v)
            x.restoreSubVector(self._iset, tmp)
        self.solve(self._rhs_tmp,self._phi_tmp)
        with self._phi_tmp.dat.vec_ro as v:
            y.array[:] = v.array[:]

    def restrict(self,phi_high, phi_low):
        '''Restrict to lower order space.
    
        Project a function in the higher order space onto the lower order space
    
        :arg phi_high: Function in higher order space
        :arg phi_low: Resulting function in lower order space
        '''
        par_loop(self._kernel_restrict,self._dx,
                 {'phi_high':(phi_high,READ),
                  'phi_low':(phi_low,WRITE)})

    def prolongadd(self,phi_low, phi_high):
        '''Prolongate to higher order space.
    
        Project a function in the lower order space onto the higher order space
    
        :arg phi_low: Function in lower order space
        :arg phi_high: Resulting function in higher order space
        '''
        par_loop(self._kernel_prolongadd,self._dx,
                 {'phi_low':(phi_low,READ),
                  'phi_high':(phi_high,INC)})
