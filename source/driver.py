import sys
import os
import math
import xml.etree.cElementTree as ET
from firedrake import * 
op2.init(log_level="ERROR")
from ffc import log as ffc_log
ffc_log.set_level(ffc_log.ERROR)
from ufl import log as ufl_log
ufl_log.set_level(ufl_log.ERROR)
import gravitywaves
from mixedoperators import *
import pressuresolver.solvers
from pressuresolver.operators import *
from pressuresolver.preconditioners import *
from pressuresolver.smoothers import *
from pressuresolver.mu_tilde import *
from pressuresolver.lumpedmass import *
from pressuresolver.hierarchy import *
from orography import *
from auxilliary.logger import *
from auxilliary.ksp_monitor import *
from auxilliary.gaussian_expression import *
import profile_wrapper
from parameters import Parameters
from mpi4py import MPI
from pyop2 import profiling
from pyop2.profiling import timed_region
from pyop2 import performance_summary
from pyop2.base import ParLoop
from firedrake.petsc import PETSc
parameters["pyop2_options"]["profiling"] = True

def initialise_parameters(filename=None):
    '''Set default parameters and read from file.

    :arg filename: Name of file to read from, do not read if this is None
    '''
    logger = Logger()
    # General parameters
    param_general = Parameters('General',
        # Carry out a warmup solve?
        {'warmup_run':True,
        # CFL number of sound waves
        'nu_cfl':10.0,
        # Sound wave speed
        'speed_c':300.0, # m/s
        # Buoyancy frequency
        'speed_N':0.01, # 1/s
        # PETSc solve?
        'solve_petsc':True,
        # Matrixfree solve?
        'solve_matrixfree':True,
        # Number of Gaussians in initial condition
        'n_gaussian':16})

    # Output parameters
    param_output = Parameters('Output',
        # Directory for output
        {'output_dir':'output',
        # Save fields to disk?
        'savetodisk':False})

    # Grid parameters
    param_grid = Parameters('Grid',
        # Number of refinement levels to construct coarsest multigrid level
        {'ref_count_coarse':3,
        # Number of vertical layers
        'nlayer':4,
        # Radius of earth [m]
        'r_earth':6.371E6,
        # Thickness of spherical shell
        'thickness':1.0E4, # m (=10km)
        # Number of multigrid levels
        'nlevel':4})

    # Orography parameters
    param_orography = Parameters('Orography',
        # Enable orography
        {'enabled':False,
        # Height of mountain in m
         'height':2.E3,
        # Width of mountain in m
         'width':1.E4})

    # Mixed system parameters
    param_mixed = Parameters('Mixed system',
        # KSP type for PETSc solver
        {'ksp_type':'gmres',
        # Use higher order discretisation?
        'higher_order':False,
        # Use diagonal only in Schur complement preconditioner
        'schur_diagonal_only':False,
        # tolerance
        'tolerance':1.0E-5,
        # maximal number of iterations
        'maxiter':20,
        # verbosity level
        'verbose':2})

    # Pressure solve parameters
    param_pressure = Parameters('Pressure solve',
        # KSP type for PETSc solver
        {'ksp_type':'cg',
        # tolerance
        'tolerance':1.E-5,
        # maximal number of iterations
        'maxiter':10,
        # verbosity level
        'verbose':1,
        # use multigrid? 
        'multigrid':True})

    # Multigrid parameters
    param_multigrid = Parameters('Multigrid',
        # multigrid smoother relaxation factor
        {'mu_relax':1.0,
        # presmoothing steps
        'n_presmooth':1,
        # postsmoothing steps
        'n_postsmooth':1,
        # number of coarse grid smoothing steps
        'n_coarsesmooth':1,
        # Solver to be used on coarsest grid
        'coarsesolver':'jacobi'})

    # Single level preconditioner parameters
    param_singlelevel = Parameters('Singlelevel',
        # smoother relaxation factor
        {'mu_relax':1.0,
        # smoothing steps
         'n_smooth':1,
        # preconditioner to use
         'preconditioner':'jacobi'})

    if (filename != None):
        for param in (param_general,
                      param_output,
                      param_grid,
                      param_orography,
                      param_mixed,
                      param_pressure,
                      param_multigrid,
                      param_singlelevel):
            if (logger.rank == 0):
                param.read_from_file(filename)
            param.broadcast(logger.comm)

    all_param = (param_general,
                 param_output,
                 param_grid,
                 param_orography,
                 param_mixed,
                 param_pressure,
                 param_multigrid,
                 param_singlelevel)
    logger.write('*** Parameters ***')
    for param in all_param:
        logger.write(str(param))

    return all_param

def build_mesh_hierarchy(param_grid,param_orography):
    '''Build extruded mesh hierarchy.

    Build mesh hierarchy based on an extruded icosahedral mesh.

    :arg param_grid: Grid parameters
    :arg param_orography: Orography parameters
    '''
    ref_count_coarse = param_grid['ref_count_coarse']
    nlevel = param_grid['nlevel']
    nlayer = param_grid['nlayer']
    thickness = param_grid['thickness']
    r_earth = param_grid['r_earth']
    # Create coarsest mesh
    coarse_host_mesh = IcosahedralSphereMesh(r_earth,
                                             refinement_level=ref_count_coarse)
    host_mesh_hierarchy = MeshHierarchy(coarse_host_mesh,nlevel)
    mesh_hierarchy = ExtrudedMeshHierarchy(host_mesh_hierarchy,
                                           layers=nlayer,
                                           extrusion_type='radial',
                                           layer_height= thickness/nlayer)
    # Distort grid, if required
    if param_orography['enabled']:
        directions = ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))
        for n in directions:            
            mountain = Mountain(n,
                                param_orography['width'],
                                param_orography['height'],
                                r_earth,
                                thickness)
            for mesh in mesh_hierarchy:
                mountain.distort(mesh)
    return mesh_hierarchy

def save_fields(output_dir,prefix,u,p,b):
    '''Save fields to disk in vtu format.

        Save the velocity, pressure and buoyancy fields to disk in vtu format
        The filenames will be `<prefix>_velocity`, `<prefix>_pressure` and
        `<prefix>_buoyancy`.

        :arg output_dir: Name of output directory
        :arg prefix: Prefix of filenames
        :arg u: Velocity field
        :arg p: Pressure field
        :arg b: Buoyancy field
    '''
    DFile_u = File(os.path.join(output_dir,prefix+'_velocity.pvd'))
    DFile_u << u
    DFile_p = File(os.path.join(output_dir,prefix+'_pressure.pvd'))
    DFile_p << p
    DFile_b = File(os.path.join(output_dir,prefix+'_buoyancy.pvd'))
    DFile_b << b

def build_function_spaces(mesh_hierarchy,
                          higher_order=False):
    '''Build function spaces and function space hierarchy for h-multigrid.

    Return the following function spaces:
    
        * W2: HDiv velocity space on finest grid and highest order
        * W3: L2 pressure space on finest grid and highest order
        * Wb: Buoyancy space on finest grid and highest order
        * W2_horiz: Horizontal component of HDiv velocity space on finest grid
            and highest order
        * W2_vert: Vertical component of HDiv velocity space on finest grid and highest order
        
    In addition, return the following lowest order hierarchies, which are required
    for building the h-multigrid preconditioner:

        * W2_horiz_hierarchy: Horizontal component of HDiv velocity space
        * W2_vert_hierarchy: Vertical component of HDiv velocity space
        * W3_hierarchy: L2 pressure space

    Note that if lowest_order=True, some of the function spaces are identical to the
    finest element of the function space hierarchy.

    :arg mesh_hierarchy: Extruded mesh hierarchy
    :arg higher_order: Build higher order elements?
    '''
    mesh = mesh_hierarchy[-1]
    # Lowest order horizontal elements
    U1_lo = FiniteElement('RT',triangle,1)
    U2_lo = FiniteElement('DG',triangle,0)
    # Lowest order vertical elements
    V0_lo = FiniteElement('CG',interval,1)
    V1_lo = FiniteElement('DG',interval,0)
    # Lowest order product elements
    W2_elt_horiz_lo = HDiv(OuterProductElement(U1_lo,V1_lo))
    W2_elt_vert_lo = HDiv(OuterProductElement(U2_lo,V0_lo))
    W3_elt_lo = OuterProductElement(U2_lo,V1_lo)
    # Velocity space hierarchy
    W2_vert_hierarchy = FunctionSpaceHierarchy(mesh_hierarchy,W2_elt_vert_lo)
    W2_horiz_hierarchy = FunctionSpaceHierarchy(mesh_hierarchy,W2_elt_horiz_lo)
    # Pressure space hierarchy
    W3_hierarchy = FunctionSpaceHierarchy(mesh_hierarchy,W3_elt_lo)
    for i in range(len(W3_hierarchy)-1):
        W3_hierarchy.cell_node_map(i)
    if (higher_order):
        # Higher order elements
        U1 = FiniteElement('BDFM',triangle,2)
        U2 = FiniteElement('DG',triangle,1)
        V0 = FiniteElement('CG',interval,2)
        V1 = FiniteElement('DG',interval,1)
        W2_elt_vert = HDiv(OuterProductElement(U2,V0))
        W2_elt_horiz = HDiv(OuterProductElement(U1,V1))
        W2_elt = HDiv(OuterProductElement(U1,V1)) + HDiv(OuterProductElement(U2,V0))
        W3_elt = OuterProductElement(U2,V1)
        Wb_elt = OuterProductElement(U2,V0)
        # Function spaces
        W2 = FunctionSpace(mesh,W2_elt)
        W3 = FunctionSpace(mesh,W3_elt)
        Wb = FunctionSpace(mesh,Wb_elt)
        W2_horiz = FunctionSpace(mesh,W2_elt_horiz)
        W2_vert = FunctionSpace(mesh,W2_elt_vert)
    else:
        W2_elt_lo = HDiv(OuterProductElement(U1_lo,V1_lo)) \
                  + HDiv(OuterProductElement(U2_lo,V0_lo))
        Wb_elt_lo = OuterProductElement(U2_lo,V0_lo)
        W2 = FunctionSpace(mesh,W2_elt_lo)
        W3 = W3_hierarchy[-1]
        Wb = FunctionSpace(mesh,Wb_elt_lo)
        W2_horiz = W2_horiz_hierarchy[-1]
        W2_vert = W2_vert_hierarchy[-1]

    return W2,W3,Wb,W2_horiz,W2_vert,W2_horiz_hierarchy,W2_vert_hierarchy,W3_hierarchy

def matrixfree_solver_setup(functionspaces,dt,all_param):
    '''Set up matrixfree solver.

        :arg functionspaces: The function spaces and function space hierarchies
            as returned by :func:`build_function_spaces`
        :arg dt: Time step size
        :arg all_param: Parameters as returned by :func:`initialise_parameters`
    '''
    W2,W3,Wb,W2_horiz,W2_vert, \
        W2_horiz_hierarchy,W2_vert_hierarchy,W3_hierarchy = functionspaces
    param_general, \
        param_output, \
        param_grid, \
        param_orography, \
        param_mixed, \
        param_pressure, \
        param_multigrid, \
        param_singlelevel = all_param
    c = param_general['speed_c']
    N = param_general['speed_N']
    omega_c = 0.5*c*dt
    omega_N = 0.5*N*dt
    mixed_ksp_monitor = KSPMonitor('mixed',verbose=param_mixed['verbose'])
    pressure_ksp_monitor = KSPMonitor('pressure',verbose=param_pressure['verbose'])
    if (param_pressure['multigrid']):
        with timed_region('matrixfree multigrid setup'):
            with timed_region('matrixfree op_Hhat setup'):
                op_Hhat_hierarchy = HierarchyContainer(Operator_Hhat,
                  zip(W3_hierarchy,
                      W2_horiz_hierarchy,
                      W2_vert_hierarchy),
                  omega_c,
                  omega_N)
            nlevel = len(op_Hhat_hierarchy)
            # Start counting at 1 at higher order since the p-operator is 0
            with timed_region('matrixfree smoother setup'):
                presmoother_hierarchy = HierarchyContainer(Jacobi,
                                            zip(op_Hhat_hierarchy),
                                                mu_relax=param_multigrid['mu_relax'],
                                                n_smooth=param_multigrid['n_presmooth'])

                postsmoother_hierarchy = HierarchyContainer(Jacobi,
                                            zip(op_Hhat_hierarchy),
                                                mu_relax=param_multigrid['mu_relax'],
                                                n_smooth=param_multigrid['n_postsmooth'])

                if (param_multigrid['coarsesolver'] == 'jacobi'):
                    coarsegrid_solver = Jacobi(op_Hhat_hierarchy[0],
                                               mu_relax=param_multigrid['mu_relax'],
                                               n_smooth=param_multigrid['n_coarsesmooth'],
                                               level=0)
                else:
                    # Build W2 function space on coarsest mesh
                    coarse_mesh = W3_hierarchy[0].mesh()
                    W2_coarse = FunctionSpace(coarse_mesh, W2.ufl_element())
                    W3_coarse = FunctionSpace(coarse_mesh, W3.ufl_element())
                    mixed_operator = MixedOperator(W2_coarse,W3_coarse,dt,c,N)
                    mutilde = Mutilde(mixed_operator,lumped=True,label='coarse_mutilde')

                    op_H = Operator_H(W3_coarse,W2_coarse,mutilde,omega_c)

                    coarsegrid_solver = DirectSolver(op_Hhat_hierarchy[0],
                                                     W2_coarse,
                                                     dt, c, N,
                                                     param_multigrid['coarsesolver'],
                                                     op_H=op_H)

            hmultigrid = hMultigrid(W3_hierarchy,
                                    op_Hhat_hierarchy,
                                    presmoother_hierarchy,
                                    postsmoother_hierarchy,
                                    coarsegrid_solver)

            if (param_mixed['higher_order']):
                with timed_region('matrixfree op_Hhat setup'):
                    op_Hhat = Operator_Hhat(W3,W2_horiz,W2_vert,omega_c,omega_N,
                                            level=nlevel)
                with timed_region('matrixfree smoother setup'):
                    presmoother = Jacobi(op_Hhat,
                                         mu_relax=param_multigrid['mu_relax'],
                                         n_smooth=param_multigrid['n_presmooth'],
                                         level=nlevel)
                    postsmoother = Jacobi(op_Hhat,
                                          mu_relax=param_multigrid['mu_relax'],
                                          n_smooth=param_multigrid['n_postsmooth'],
                                          level=nlevel)
                preconditioner = hpMultigrid(hmultigrid,
                                             op_Hhat,
                                             presmoother,
                                             postsmoother)
            else:
                preconditioner = hmultigrid
                op_Hhat = op_Hhat_hierarchy[-1]

    else:
        with timed_region('matrixfree singlelevel setup'):
            with timed_region('matrixfree op_Hhat setup'):
                op_Hhat = Operator_Hhat(W3,W2_horiz,W2_vert,omega_c,omega_N,
                                        level=0)
            with timed_region('matrixfree smoother setup'):
                if (param_singlelevel['preconditioner']=='jacobi'):
                    preconditioner = Jacobi(op_Hhat,
                                            param_singlelevel['mu_relax'],
                                            param_singlelevel['n_smooth'],
                                            level=0)
                else:
                    preconditioner = DirectSolver(op_Hhat,
                                                  W2,
                                                  dt, c, N,
                                                  param_singlelevel['preconditioner'],)

    with timed_region('matrixfree mixed operator setup'):
        mixed_operator = MixedOperator(W2,W3,dt,c,N)

    with timed_region('matrixfree pc_hdiv setup'):
        mutilde = Mutilde(mixed_operator,lumped=True,label='full')

    with timed_region('matrixfree op_H setup'):
        op_H = Operator_H(W3,W2,mutilde,omega_c)

    # Construct pressure solver based on operator and preconditioner 
    # built above
    with timed_region('matrixfree KSP setup'):
        pressure_solver = pressuresolver.solvers.PETScSolver(op_H,
                                              preconditioner,
                                              param_pressure['ksp_type'],
                                              ksp_monitor=pressure_ksp_monitor,
                                              tolerance=param_pressure['tolerance'],
                                              maxiter=param_pressure['maxiter'])

        # Construct mixed gravity wave solver
        if (param_orography['enabled']):
            Solver = gravitywaves.MatrixFreeSolverOrography
        else:
            Solver = gravitywaves.MatrixFreeSolver
        gravitywave_solver_matrixfree = Solver(Wb,mixed_operator,mutilde,
                                               ksp_type=param_mixed['ksp_type'],
                                               schur_diagonal_only = \
                                                 param_mixed['schur_diagonal_only'],
                                               ksp_monitor=mixed_ksp_monitor,
                                               tolerance=param_mixed['tolerance'],
                                               maxiter=param_mixed['maxiter'],
                                               pressure_solver=pressure_solver)
    logger.write('matrix-explicit mixed operator apply, nnz per row')
    nnz_labels = ('uu','up','pu','pp')
    for i,nnz in enumerate(mixed_operator.get_nnz()):
        logger.write('  A_'+nnz_labels[i]+' = '+('%8.4f' % nnz))
    return gravitywave_solver_matrixfree

def solve_matrixfree(functionspaces,dt,all_param,expression):
    '''Solve with matrixfree solver.

        :arg functionspaces: The function spaces and function space hierarchies
            as returned by :func:`build_function_spaces`
        :arg dt: Time step size
        :arg all_param: Parameters as returned by :func:`initialise_parameters`
        :arg expression: expression for RHS of pressure equation
    '''
    W2,W3,Wb,W2_horiz,W2_vert, \
        W2_horiz_hierarchy,W2_vert_hierarchy,W3_hierarchy = functionspaces
    param_general, \
        param_output, \
        param_grid, \
        param_orography, \
        param_mixed, \
        param_pressure, \
        param_multigrid, \
        param_singlelevel = all_param

    r_u = Function(W2)
    r_p = Function(W3)
    r_b = Function(Wb)
    c = param_general['speed_c']
    N = param_general['speed_N']
    mixed_operator_matrixfree = MixedOperator(W2,W3,dt,c,N,
                                              preassemble=False)

    # Warm up run
    if (param_general['warmup_run']):
        logger.write('Warmup...')
        stdout_save = sys.stdout
        with timed_region("warmup"), PETSc.Log().Stage("warmup"):
            with open(os.devnull,'w') as sys.stdout:
                gravitywave_solver_matrixfree = matrixfree_solver_setup(functionspaces,
                                                                        dt,all_param)
                r_u.assign(0.0)
                r_p.project(expression,solver_parameters={'ksp_type':'cg','pc_type':'jacobi'})
                r_b.assign(0.0)
                u,p,b = gravitywave_solver_matrixfree.solve(r_u,r_p,r_b)
                mixed_operator_matrixfree.apply(u,p,r_u,r_p)
        sys.stdout = stdout_save
        # Reset timers
        profiling.reset_timers()
        logger.write('...done')
        logger.write('')

    r_u.assign(0.0)
    r_p.project(expression,solver_parameters={'ksp_type':'cg','pc_type':'jacobi'})
    r_b.assign(0.0)

    # Reset all performance counters
    ParLoop.perfdata = {}
    with timed_region("matrixfree mixed system solve"):
        with timed_region("matrixfree total solver setup"):
            gravitywave_solver_matrixfree = matrixfree_solver_setup(functionspaces,
                                                                    dt,all_param)
        with PETSc.Log().Stage("solve_matrixfree"):
            with PETSc.Log().Event("Full matrixfree solve"):
                u,p,b = gravitywave_solver_matrixfree.solve(r_u,r_p,r_b)
    with timed_region("matrixfree apply matrixfree mixed operator"):
        mixed_operator_matrixfree.apply(u,p,r_u,r_p)
    nflop_per_cell = mixed_operator_matrixfree.apply(u,p,r_u,r_p,
                                                      count_flops=True)
    logger.write('matrix-free mixed operator apply, FLOPs per cell = '+str(nflop_per_cell))

    op_Hhat_v = gravitywave_solver_matrixfree._pressure_solver._preconditioner._operator._Hhat_v
    n_col = op_Hhat_v._n_col
    n_row = op_Hhat_v._n_row
    bandwidth = op_Hhat_v.bandwidth
    logger.write('Helmholtz operator: size = '+str(n_row)+' x '+str(n_row)+' , bandwidth = '+str(bandwidth))


    conv_hist_filename = os.path.join(param_output['output_dir'],'history_matrixfree.dat')
    gravitywave_solver_matrixfree._ksp_monitor.save_convergence_history(conv_hist_filename)
    comm = MPI.COMM_WORLD
    if (comm.Get_rank() == 0):
        xml_root = ET.Element("SolverInformation")
        xml_tree = ET.ElementTree(xml_root)
        gravitywave_solver_matrixfree.add_to_xml(xml_root,"gravitywave_solver")
        xml_tree.write('solver.xml')
    return u,p,b

def solve_petsc(functionspaces,dt,all_param,expression):
    ''' Solve with PETSc solver.

        :arg functionspaces: The function spaces and function space hierarchies
            as returned by :func:`build_function_spaces`
        :arg dt: Time step size
        :arg all_param: Parameters as returned by :func:`initialise_parameters`
        :arg expression: expression for RHS of pressure equation
    '''
    W2,W3,Wb,W2_horiz,W2_vert, \
        W2_horiz_hierarchy,W2_vert_hierarchy,W3_hierarchy = functionspaces
    param_general, \
        param_output, \
        param_grid, \
        param_orography, \
        param_mixed, \
        param_pressure, \
        param_multigrid, \
        param_singlelevel = all_param
    mixed_ksp_monitor = KSPMonitor('mixed',verbose=param_mixed['verbose'])
    c = param_general['speed_c']
    N = param_general['speed_N']
    omega_c = 0.5*c*dt
    omega_N = 0.5*N*dt

    r_u = Function(W2)
    r_p = Function(W3)
    r_b = Function(Wb)
    # Construct mixed gravity wave solver
    gravitywave_solver_petsc = gravitywaves.PETScSolver(W2,W3,Wb,
                                             dt,c,N,
                                             ksp_type=param_mixed['ksp_type'],
                                             ksp_monitor=mixed_ksp_monitor,
                                             tolerance=param_mixed['tolerance'],
                                             maxiter=param_mixed['maxiter'],
                                             multigrid=param_pressure['multigrid'])
    # Warm up run
    if (param_general['warmup_run']):
        logger.write('Warmup...')
        stdout_save = sys.stdout
        with open(os.devnull,'w') as sys.stdout:
            with timed_region("warmup"), PETSc.Log().Stage("warmup"):
                r_u.assign(0.0)
                r_p.project(expression)
                r_b.assign(0.0)
                u,p,b = gravitywave_solver_petsc.solve(r_u,r_p,r_b)
        sys.stdout = stdout_save
        # Reset timers
        profiling.reset_timers()
        logger.write('...done')
        logger.write('')

    r_u.assign(0.0)
    r_p.project(expression)
    r_b.assign(0.0)

    with timed_region("petsc mixed system solve"):
        with PETSc.Log().Stage("solve_petsc"):
            with PETSc.Log().Event("Full PETSc solve"):
                u,p,b = gravitywave_solver_petsc.solve(r_u,r_p,r_b)
    conv_hist_filename = os.path.join(param_output['output_dir'],'history_petsc.dat')
    mixed_ksp_monitor.save_convergence_history(conv_hist_filename)

    vmixed = Function(W2 * W3)
    up_solver = gravitywave_solver_petsc.up_solver
    ksp = up_solver.snes.getKSP()
    # Print solver to disk
    viewer = PETSc.Viewer()
    file_viewer = viewer.createASCII('petsc_ksp.log')
    ksp.view(file_viewer)
    ksp_hdiv, ksp_schur = ksp.getPC().getFieldSplitSubKSP()

    # HDiv space
    op_hdiv, op_pc_hdiv = ksp_hdiv.getOperators()
    pc_hdiv = ksp_hdiv.getPC()
    x, y = op_pc_hdiv.getVecs()
    x.setArray(np.random.rand(x.getLocalSize()))
    pc_hdiv.apply(x,y)
    with timed_region('petsc pc_hdiv'):
        pc_hdiv.apply(x,y)

    # Pressure space
    op_schur, op_pc_schur = ksp_schur.getOperators()
    pc_schur = ksp_schur.getPC()
    x, y = op_pc_schur.getVecs()
    x.setArray(np.random.rand(x.getLocalSize()))
    y = x.duplicate()
    op_pc_schur.mult(x,y)
    with timed_region('petsc op_schur'):
        op_pc_schur.mult(x,y)
    pc_schur.apply(x,y)
    with timed_region('petsc pc_schur'):
        pc_schur.apply(x,y)
    return u,p,b

##########################################################
# M A I N
##########################################################
def main(parameter_filename=None):
    # Create parallel logger
    logger = Logger()

    logger.write('+---------------------------+')
    logger.write('! Mixed Gravity wave solver !')
    logger.write('+---------------------------+')
    logger.write('')
    logger.write('Running on '+('%8d' % logger.size)+' MPI processes')
    logger.write('')
 
    all_param = initialise_parameters(parameter_filename)
    param_general, \
      param_output, \
      param_grid, \
      param_orography, \
      param_mixed, \
      param_pressure, \
      param_multigrid, \
      param_singlelevel = all_param
    # ------------------------------------------------------

    # Create output directory if it does not already exist
    if (logger.rank == 0):
        if (not os.path.exists(param_output['output_dir'])):
            os.mkdir(param_output['output_dir'])
    mesh_hierarchy = build_mesh_hierarchy(param_grid,param_orography)

    # Extract mesh on finest level
    mesh = mesh_hierarchy[-1]
    ncells = MPI.COMM_WORLD.allreduce(mesh.cell_set.size)
    logger.write('Number of cells on finest grid = '+str(ncells))

    # Set time step size dt such that c*dt/dx = nu_cfl
    dx = 2.*param_grid['r_earth']/math.sqrt(3.)*math.sqrt(4.*math.pi/(ncells))
    dt = param_general['nu_cfl']/param_general['speed_c']*dx
    logger.write('dx = '+('%12.3f' % (1.E-3*dx))+' km,  dt = '+('%12.3f' % dt)+' s')
    omega_c = 0.5*param_general['speed_c']*dt
    omega_N = 0.5*param_general['speed_N']*dt

    # Construct function spaces and hierarchies
    functionspaces = build_function_spaces(mesh_hierarchy,param_mixed['higher_order'])
    W2,W3,Wb,W2_horiz,W2_vert, \
      W2_horiz_hierarchy, \
      W2_vert_hierarchy, \
      W3_hierarchy = functionspaces
    
    mixed_ksp_monitor = KSPMonitor('mixed',verbose=param_mixed['verbose'])
    pressure_ksp_monitor = KSPMonitor('pressure',verbose=param_pressure['verbose'])

    # Right hand side function
    g = MultipleGaussianExpression(param_general['n_gaussian'],
                                   param_grid['r_earth'],
                                   param_grid['thickness'])
    expression = Expression(str(g))

    if (param_general['solve_matrixfree']):
        logger.write('*** Matrix free solve ***')
        u_matrixfree,p_matrixfree,b_matrixfree = solve_matrixfree(functionspaces,dt,all_param,expression)
        logger.write('')
    
        if (logger.rank == 0):
            profiling.summary()
        performance_summary()
        
        # If requested, write fields to disk
        if (param_output['savetodisk']):
            save_fields(param_output['output_dir'],'matrixfree',u,p,b)

    if (param_general['solve_petsc']):
        logger.write('*** PETSc solve ***')
        u_petsc,p_petsc,b_petsc = solve_petsc(functionspaces,dt,all_param,expression)

        if (logger.rank == 0):
            profiling.summary()
        performance_summary()
    
        # If requested, write fields to disk
        if (param_output['savetodisk']):
            save_fields(param_output['output_dir'],'petsc',u,p,b)

    # Compare solutions
    if (param_general['solve_petsc'] and param_general['solve_matrixfree']):
        norm_u = norm(u_matrixfree)
        norm_p = norm(p_matrixfree)
        norm_b = norm(b_matrixfree)
        norm_du = norm(assemble(u_petsc-u_matrixfree))
        norm_dp = norm(assemble(p_petsc-p_matrixfree))
        norm_db = norm(assemble(b_petsc-b_matrixfree))
        norm_dtotal = math.sqrt(norm_du**2+norm_dp**2+norm_db**2)
        norm_total = math.sqrt(norm_u**2+norm_p**2+norm_b**2)
        logger.write('')
        logger.write('  ||u||      = ' + ('%8.4e' % norm_u) + \
            '  ||u_{petsc} - u_{matrixfree}|| = ' + ('%8.4e' % norm_du))
        logger.write('  ||p||      = ' + ('%8.4e' % norm_p) + \
            '  ||p_{petsc} - p_{matrixfree}|| = ' + ('%8.4e' % norm_dp))
        logger.write('  ||b||      = ' + ('%8.4e' % norm_b) + \
            '  ||b_{petsc} - b_{matrixfree}|| = ' + ('%8.4e' % norm_db))
        logger.write('  total norm = ' + ('%8.4e' % norm_total) + \
            '   total difference              = ' + ('%8.4e' % norm_dtotal))
        logger.write('')

##########################################################
# Call main program
##########################################################
if (__name__ == '__main__'):
    # Create parallel logger
    logger = Logger()
    parameter_filename = None
    if (len(sys.argv) >= 2):
        parameter_filename = sys.argv[1]
    main(parameter_filename)


