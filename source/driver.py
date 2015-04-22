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
from firedrake.petsc import PETSc

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
        'verbose':1})
    
    # Multigrid parameters
    param_multigrid = Parameters('Multigrid',
        # multigrid smoother relaxation factor
        {'mu_relax':1.0,
        # presmoothing steps
        'n_presmooth':1,
        # postsmoothing steps
        'n_postsmooth':1,
        # number of coarse grid smoothing steps
        'n_coarsesmooth':1})

    if (filename != None):
        for param in (param_general,
                      param_output,
                      param_grid,
                      param_orography,
                      param_mixed,
                      param_pressure,
                      param_multigrid):
            if (logger.rank == 0):
                param.read_from_file(filename)
            param.broadcast(logger.comm)

    all_param = (param_general,
                 param_output,
                 param_grid,
                 param_orography,
                 param_mixed,
                 param_pressure,
                 param_multigrid)
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
        param_multigrid = all_param
    c = param_general['speed_c']
    N = param_general['speed_N']
    omega_c = 0.5*c*dt
    omega_N = 0.5*N*dt
    op_Hhat_hierarchy = HierarchyContainer(Operator_Hhat,
      zip(W3_hierarchy,
          W2_horiz_hierarchy,
          W2_vert_hierarchy),
      omega_c,
      omega_N)
    mixed_ksp_monitor = KSPMonitor('mixed',verbose=param_mixed['verbose'])
    pressure_ksp_monitor = KSPMonitor('pressure',verbose=param_pressure['verbose'])
    presmoother_hierarchy = HierarchyContainer(Jacobi,
                                zip(op_Hhat_hierarchy),
                                    mu_relax=param_multigrid['mu_relax'],
                                    n_smooth=param_multigrid['n_presmooth'])

    postsmoother_hierarchy = HierarchyContainer(Jacobi,
                                zip(op_Hhat_hierarchy),
                                    mu_relax=param_multigrid['mu_relax'],
                                    n_smooth=param_multigrid['n_postsmooth'])

    coarsegrid_solver = Jacobi(op_Hhat_hierarchy[0],
                               mu_relax=param_multigrid['mu_relax'],
                               n_smooth=param_multigrid['n_coarsesmooth'])

    hmultigrid = hMultigrid(W3_hierarchy,
                            op_Hhat_hierarchy,
                            presmoother_hierarchy,
                            postsmoother_hierarchy,
                            coarsegrid_solver)

    if (param_mixed['higher_order']):
        op_Hhat = Operator_Hhat(W3,W2_horiz,W2_vert,omega_c,omega_N)
        presmoother = Jacobi(op_Hhat,
                             mu_relax=param_multigrid['mu_relax'],
                             n_smooth=param_multigrid['n_presmooth'])
        postsmoother = Jacobi(op_Hhat,
                              mu_relax=param_multigrid['mu_relax'],
                              n_smooth=param_multigrid['n_postsmooth'])
        preconditioner = hpMultigrid(hmultigrid,
                                     op_Hhat,
                                     presmoother,
                                     postsmoother)
    else:
        preconditioner = hmultigrid
        op_Hhat = op_Hhat_hierarchy[-1]

    mixed_operator = MixedOperator(W2,W3,dt,c,N)

    mutilde = Mutilde(mixed_operator,
                      lumped=(not param_mixed['higher_order']),
                      tolerance_u=1.E-1,maxiter_u=100)

    op_H = Operator_H(W3,W2,mutilde,omega_c)

    # Construct pressure solver based on operator and preconditioner 
    # built above
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
    gravitywave_solver_matrixfree = Solver(mixed_operator,
                                           W2,W3,Wb,
                                           dt,
                                           param_general['speed_c'],
                                           param_general['speed_N'],
                                           ksp_type=param_mixed['ksp_type'],
                                           schur_diagonal_only = \
                                             param_mixed['schur_diagonal_only'],
                                           ksp_monitor=mixed_ksp_monitor,
                                           tolerance=param_mixed['tolerance'],
                                           maxiter=param_mixed['maxiter'],
                                           pressure_solver=pressure_solver)
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
        param_multigrid = all_param

    r_u = Function(W2)
    r_p = Function(W3)
    r_b = Function(Wb)
    # Warm up run
    if (param_general['warmup_run']):
        logger.write('Warmup...')
        stdout_save = sys.stdout
        with timed_region("warmup"), PETSc.Log().Stage("warmup"):
            with open(os.devnull,'w') as sys.stdout:
                gravitywave_solver_matrixfree = matrixfree_solver_setup(functionspaces,
                                                                        dt,all_param)
                r_u.assign(0.0)
                r_p.project(expression)
                r_b.assign(0.0)
                u,p,b = gravitywave_solver_matrixfree.solve(r_u,r_p,r_b)
        sys.stdout = stdout_save
        # Reset timers
        profiling.reset_timers()
        logger.write('...done')
        logger.write('')

    r_u.assign(0.0)
    r_p.project(expression)
    r_b.assign(0.0)

    with timed_region("matrixfree mixed system solve"):
        with timed_region("matrixfree solver setup"):
            gravitywave_solver_matrixfree = matrixfree_solver_setup(functionspaces,
                                                                    dt,all_param)
        with PETSc.Log().Stage("solve_matrixfree"):
            with PETSc.Log().Event("Full matrixfree solve"):
                u,p,b = gravitywave_solver_matrixfree.solve(r_u,r_p,r_b)
    
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
        param_multigrid = all_param
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
                                             maxiter=param_mixed['maxiter'])
    # Warm up run
    if (param_general['warmup_run']):
        logger.write('Warmup...')
        stdout_save = sys.stdout
        with timed_region("warmup"), PETSc.Log().Stage("warmup"):
            with open(os.devnull,'w') as sys.stdout:
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
    ksp_hdiv, ksp_schur = ksp.getPC().getFieldSplitSubKSP()

    # HDiv space
    op_hdiv, op_pc_hdiv = ksp_hdiv.getOperators()
    pc_hdiv = ksp_hdiv.getPC()
    x, y = op_pc_hdiv.getVecs()
    x.setArray(np.random.rand(x.getLocalSize()))
    pc_hdiv.apply(x,y)
    with timed_region('pc_hdiv'):
        pc_hdiv.apply(x,y)

    # Pressure space
    op_schur, op_pc_schur = ksp_schur.getOperators()
    pc_schur = ksp_schur.getPC()
    x, y = op_pc_schur.getVecs()
    x.setArray(np.random.rand(x.getLocalSize()))
    y = x.duplicate()
    op_pc_schur.mult(x,y)
    with timed_region('op_schur'):
        op_pc_schur.mult(x,y)
    pc_schur.apply(x,y)
    with timed_region('pc_schur'):
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
      param_multigrid = all_param
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
        u,p,b = solve_matrixfree(functionspaces,dt,all_param,expression)
        logger.write('')
    
        if (logger.rank == 0):
            profiling.summary()
        
        # If requested, write fields to disk
        if (param_output['savetodisk']):
            save_fields(param_output['output_dir'],'matrixfree',u,p,b)

    if (param_general['solve_petsc']):
        logger.write('*** PETSc solve ***')
        u,p,b = solve_petsc(functionspaces,dt,all_param,expression)

        if (logger.rank == 0):
            profiling.summary()
    
        # If requested, write fields to disk
        if (param_output['savetodisk']):
            save_fields(param_output['output_dir'],'petsc',u,p,b)

##########################################################
# Call main program
##########################################################
if (__name__ == '__main__'):
    # Create parallel logger
    logger = Logger()
    if (len(sys.argv) > 2):
        logger.write('Usage: python '+sys.argv[0]+' [<parameterfile>]')
        sys.exit(1)
    parameter_filename = None
    if (len(sys.argv) == 2):
        parameter_filename = sys.argv[1]
    main(parameter_filename)


