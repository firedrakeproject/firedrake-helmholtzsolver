import sys
import os
import math
import xml.etree.cElementTree as ET
from firedrake import * 
op2.init(log_level="WARNING")
from ffc import log
log.set_level(log.ERROR)
import gravitywaves
import pressuresolver.solvers
from pressuresolver.operators import *
from pressuresolver.preconditioners import *
from pressuresolver.smoothers import *
from pressuresolver.mu_tilde import *
from pressuresolver.lumpedmass import *
from pressuresolver.hierarchy import *
from pressuresolver.mpi_utils import *
from pressuresolver.ksp_monitor import *
import profile_wrapper
from parameters import Parameters
from mpi4py import MPI
from pyop2 import profiling
from pyop2.profiling import timed_region
# Set correct COFFEE debug level
parameters["coffee"]["O2"] = False

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

    # ------------------------------------------------------
    # --- User defined Parameters --------------------------
    # ------------------------------------------------------

    # General parameters
    param_general = Parameters('General',
        # Carry out a warmup solve?
        {'warmup_run':True,
        # CFL number of sound waves
        'nu_cfl':10.0,
        # Sound wave speed
        'speed_c':1.0,
        # Gravity wave speed
        'speed_N':1.0,
        # Solve using the PETSc split solver?
        'use_petscsplitsolver':False,     
        # Use the matrix-free solver
        'use_matrixfreesolver':True})

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
        # Thickness of spherical shell
        'thickness':0.1,
        # Number of multigrid levels
        'nlevel':4})

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

    if parameter_filename:
        for param in (param_general,
                      param_output,
                      param_grid,
                      param_mixed,
                      param_pressure,
                      param_multigrid):
            if (logger.rank == 0):
                param.read_from_file(parameter_filename)
            param.broadcast(logger.comm)

    logger.write('*** Parameters ***')
    for param in (param_general,
                  param_output,
                  param_grid,
                  param_mixed,
                  param_pressure,
                  param_multigrid):
        logger.write(str(param))
        
    # ------------------------------------------------------

    # Create output directory if it does not already exist
    if (logger.rank == 0):
        if (not os.path.exists(param_output['output_dir'])):
            os.mkdir(param_output['output_dir'])

    # Create coarsest mesh
    coarse_host_mesh = UnitIcosahedralSphereMesh(refinement_level= \
                                                 param_grid['ref_count_coarse'])
    host_mesh_hierarchy = MeshHierarchy(coarse_host_mesh,param_grid['nlevel'])
    mesh_hierarchy = ExtrudedMeshHierarchy(host_mesh_hierarchy,
                                           layers=param_grid['nlayer'],
                                           extrusion_type='radial',
                                           layer_height=param_grid['thickness'] \
                                            / param_grid['nlayer'])

    # Extract mesh on finest level
    mesh = mesh_hierarchy[-1]
    ncells = MPI.COMM_WORLD.allreduce(mesh.cell_set.size)

    logger.write('Number of cells on finest grid = '+str(ncells))

    # Set time step size dt such that c*dt/dx = nu_cfl
    dx = 2./math.sqrt(3.)*math.sqrt(4.*math.pi/(ncells))   
    dt = param_general['nu_cfl']/param_general['speed_c']*dx
    omega_c = 0.5*param_general['speed_c']*dt
    omega_N = 0.5*param_general['speed_N']*dt

    # Build function spaces
    if (param_mixed['higher_order']):
        # NOT IMPLEMENTED
        print 'HIGHER ORDER SPACES NOT IMPLEMENTED YET'
        sys.exit(-1)
    # Horizontal elements
    U1 = FiniteElement('RT',triangle,1)
    U2 = FiniteElement('DG',triangle,0)
    # Vertical elements
    V0 = FiniteElement('CG',interval,1)
    V1 = FiniteElement('DG',interval,0)
    # Velocity space
    W2_elt_horiz = HDiv(OuterProductElement(U1,V1))
    W2_elt_vert = HDiv(OuterProductElement(U2,V0))
    W2_vert_hierarchy = FunctionSpaceHierarchy(mesh_hierarchy,W2_elt_vert)
    W2_horiz_hierarchy = FunctionSpaceHierarchy(mesh_hierarchy,W2_elt_horiz)
    W2_horiz = FunctionSpace(mesh,W2_elt_horiz)
    W2_elt = W2_elt_horiz + W2_elt_vert 
    W2_hierarchy = FunctionSpaceHierarchy(mesh_hierarchy,W2_elt)
    W2 = W2_hierarchy[-1]
    # Pressure space
    W3_elt = OuterProductElement(U2,V1)
    W3_hierarchy = FunctionSpaceHierarchy(mesh_hierarchy,W3_elt)
    W3 = W3_hierarchy[-1] 
    # Buoyancy space
    Wb_elt = OuterProductElement(U2,V0)
    Wb_hierarchy = FunctionSpaceHierarchy(mesh_hierarchy,Wb_elt)
    Wb = Wb_hierarchy[-1]

    mutilde = Mutilde(W2,Wb,omega_N)

    op_H = Operator_H(W3,W2,mutilde,omega_c)

    op_Hhat_hierarchy = HierarchyContainer(Operator_Hhat,
      zip(W3_hierarchy,
          W2_horiz_hierarchy,
          W2_vert_hierarchy),
      omega_c,
      omega_N)

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

    preconditioner = hMultigrid(W3_hierarchy,
      op_Hhat_hierarchy,
      presmoother_hierarchy,
      postsmoother_hierarchy,
      coarsegrid_solver)

    mixed_ksp_monitor = KSPMonitor('mixed',verbose=param_mixed['verbose'])
    pressure_ksp_monitor = KSPMonitor('pressure',verbose=param_pressure['verbose'])

    # Construct pressure solver based on operator and preconditioner 
    # built above
    pressure_solver = pressuresolver.solvers.PETScSolver(op_H,
                                          preconditioner,
                                          param_pressure['ksp_type'],
                                          ksp_monitor=pressure_ksp_monitor,
                                          tolerance=param_pressure['tolerance'],
                                          maxiter=param_pressure['maxiter'])

    gravitywave_solver = gravitywaves.PETScSolver(W2,W3,Wb,
                                                  pressure_solver,
                                                  dt,
                                                  param_general['speed_c'],
                                                  param_general['speed_N'],
                                                  ksp_type=param_mixed['ksp_type'],
                                                  schur_diagonal_only = \
                                                    param_mixed['schur_diagonal_only'],
                                                  ksp_monitor=mixed_ksp_monitor,
                                                  tolerance=param_mixed['tolerance'],
                                                  maxiter=param_mixed['maxiter'])

    comm = MPI.COMM_WORLD
    if (comm.Get_rank() == 0):
        xml_root = ET.Element("SolverInformation")
        xml_tree = ET.ElementTree(xml_root)
        gravitywave_solver.add_to_xml(xml_root,"gravitywave_solver")
        xml_tree.write('solver.xml')

    # Right hand side function
    r_u = Function(W2)
    r_p = Function(W3)
    p = Function(W3)
    r_b = Function(Wb)
    expression = Expression('exp(-0.5*(x[0]*x[0]+x[1]*x[1])/(0.25*0.25))')

    # Warm up run
    if (param_general['warmup_run']):
        logger.write('Warmup...')
        stdout_save = sys.stdout
        with timed_region("warmup"):
            with open(os.devnull,'w') as sys.stdout:
                r_u.assign(0.0)
                r_p.project(expression)
                r_b.assign(0.0)
                u,p,b = gravitywave_solver.solve(r_u,r_p,r_b)
        sys.stdout = stdout_save
        # Reset timers
        profiling.reset_timers()
        logger.write('...done')
        logger.write('')

    r_u.assign(0.0)
    r_p.project(expression)
    r_b.assign(0.0)

    with timed_region("matrix-free solve"):
        u,p,b = gravitywave_solver.solve(r_u,r_p,r_b)
    conv_hist_filename = os.path.join(param_output['output_dir'],'history.dat')
    mixed_ksp_monitor.save_convergence_history(conv_hist_filename)

    # If requested, write fields to disk
    if (param_output['savetodisk']):
        # Write output to disk
        DFile_u = File(os.path.join(param_output['output_dir'],
                                    'velocity.pvd'))
        DFile_u << u
        DFile_p = File(os.path.join(param_output['output_dir'],
                                    'pressure.pvd'))
        DFile_p << p
        DFile_b = File(os.path.join(param_output['output_dir'],
                                    'buoyancy.pvd'))
        DFile_b << b
    if (logger.rank == 0):
        profiling.summary()

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


