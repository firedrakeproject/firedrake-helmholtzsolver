import sys
sys.path.insert(0,'/work/n02/n02/eike/PyOP2')
import os
import math
import xml.etree.cElementTree as ET
from firedrake import * 
op2.init(log_level="WARNING")
from ffc import log
log.set_level(log.ERROR)
import helmholtz
import pressuresolver
from pressuresolver import operators, smoothers, solvers, preconditioners, lumpedmass, hierarchy, mpi_utils
import profile_wrapper
from parameters import Parameters
from pressuresolver import ksp_monitor
from mpi4py import MPI
import sys

##########################################################
# M A I N
##########################################################
if (__name__ == '__main__'):
   
    # Create parallel logger
    logger = mpi_utils.Logger()

    logger.write('+------------------------+')
    logger.write('! Mixed Helmholtz solver !')
    logger.write('+------------------------+')
    logger.write('')
    logger.write('Running on '+('%8d' % logger.size)+' MPI processes')
    logger.write('')

    if (len(sys.argv) > 2):
        logger.write('Usage: python '+sys.argv[0]+' [<parameterfile>]')
        sys.exit(1)
    if (len(sys.argv) == 2):
        parameter_filename = sys.argv[1]
    else:
        parameter_filename = None

    # ------------------------------------------------------
    # --- User defined Parameters --------------------------
    # ------------------------------------------------------

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
        # Number of multigrid levels
        'nlevel':4})

    # Mixed system parameters
    param_mixed = Parameters('Mixed system',
        # KSP type for PETSc solver
        {'ksp_type':'gmres',
        # Use higher order discretisation?
        'higher_order':False,
        # Lump mass matrix in Schur complement substitution
        'lump_mass':True,
        # Use diagonal only in Schur complement preconditioner
        'schur_diagonal_only':False,
        # Preconditioner to use: Multigrid or Jacobi (1-level method)
        'preconditioner':'Multigrid',
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
        # Lump mass in Helmholtz operator in pressure space
        'lump_mass':True,
        # tolerance
        'tolerance':1.E-5,
        # maximal number of iterations
        'maxiter':10,
        # verbosity level
        'verbose':1})
    
    # Multigrid parameters
    param_multigrid = Parameters('Multigrid',
        # Lump mass in multigrid
        {'lump_mass':True,
        # multigrid smoother relaxation factor
        'mu_relax':1.0,
        # presmoothing steps
        'n_presmooth':1,
        # postsmoothing steps
        'n_postsmooth':1,
        # number of coarse grid smoothing steps
        'n_coarsesmooth':1})

    if parameter_filename:
        for param in (param_output,
                      param_grid,
                      param_mixed,
                      param_pressure,
                      param_multigrid):
            if (logger.rank == 0):
                param.read_from_file(parameter_filename)
            param.broadcast(logger.comm)

    logger.write('*** Parameters ***')
    for param in (param_output,
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
    coarse_mesh = UnitIcosahedralSphereMesh(refinement_level= \
                                   param_grid['ref_count_coarse'])
    global_normal = Expression(("x[0]","x[1]","x[2]"))

    # Create mesh hierarchy
    mesh_hierarchy = MeshHierarchy(coarse_mesh,param_grid['nlevel'])
    for level_mesh in mesh_hierarchy:
        global_normal = Expression(("x[0]","x[1]","x[2]"))
        level_mesh.init_cell_orientations(global_normal)

    # Extract mesh on finest level
    fine_level = len(mesh_hierarchy)-1
    mesh = mesh_hierarchy[fine_level]
    ncells = MPI.COMM_WORLD.allreduce(mesh.cell_set.size)

    logger.write('Number of cells on finest grid = '+str(ncells))
    dx = 2./math.sqrt(3.)*math.sqrt(4.*math.pi/(ncells))
   
    # Calculate parameter omega for 2nd order term 
    omega = 8.*0.5*dx

    # Build function spaces and velocity mass matrix on finest level
    if (param_mixed['higher_order']):
        V_pressure = FunctionSpace(mesh,'DG',1)
        V_velocity = FunctionSpace(mesh,'BDFM',2)
        lumped_mass_fine = lumpedmass.LumpedMassBDFM1(V_velocity)
    else:
        V_pressure = FunctionSpace(mesh,'DG',0)
        V_velocity = FunctionSpace(mesh,'RT',1)
        lumped_mass_fine = lumpedmass.LumpedMassRT0(V_velocity)
    full_mass_fine = lumpedmass.FullMass(V_velocity)

    lumped_mass_fine.test_kinetic_energy()
    
    # Construct preconditioner
    if (param_mixed['preconditioner'] == 'Jacobi'):
        # Case 1: Jacobi
        if (param_pressure['lump_mass']):
            velocity_mass_matrix_operator = lumped_mass_fine
        else:
            velocity_mass_matrix_operator = full_mass_fine
        operator = operators.Operator(V_pressure,
                                      V_velocity,
                                      velocity_mass_matrix_operator,
                                      omega)
        if (param_mixed['higher_order']):
            preconditioner = smoothers.Jacobi_HigherOrder(operator,
                velocity_mass_matrix_operator)
        else:
            preconditioner = smoothers.Jacobi_LowestOrder(operator,
                velocity_mass_matrix_operator)
        # Case 2: Multigrid
    elif (param_mixed['preconditioner'] == 'Multigrid'):
        # Build hierarchies for h-multigrid 
        # (i) Function spaces
        V_pressure_hierarchy = FunctionSpaceHierarchy(mesh_hierarchy,'DG',0)
        V_velocity_hierarchy = FunctionSpaceHierarchy(mesh_hierarchy,'RT',1)
        # (ii) Lumped mass matrices
        if (param_multigrid['lump_mass']):
            mass_hierarchy = \
                hierarchy.HierarchyContainer(lumpedmass.LumpedMassRT0,
                                             zip(V_velocity_hierarchy))
            lumped_mass_hierarchy = mass_hierarchy
        else:                
            mass_hierarchy = \
                hierarchy.HierarchyContainer(lumpedmass.FullMass,
                                             zip(V_velocity_hierarchy))
            lumped_mass_hierarchy = \
                hierarchy.HierarchyContainer(lumpedmass.LumpedMassRT0,
                                             zip(V_velocity_hierarchy))
        # (iii) operators
        operator_hierarchy = hierarchy.HierarchyContainer(
            operators.Operator,
            zip(V_pressure_hierarchy,
                V_velocity_hierarchy,
                mass_hierarchy),
            omega)
        # (iv) pre- and post-smoothers
        presmoother_hierarchy = \
            hierarchy.HierarchyContainer(smoothers.Jacobi_LowestOrder,
               zip(operator_hierarchy,
                   lumped_mass_hierarchy),
                n_smooth=param_multigrid['n_presmooth'],
                mu_relax=param_multigrid['mu_relax'])
        postsmoother_hierarchy = \
            hierarchy.HierarchyContainer(smoothers.Jacobi_LowestOrder,
                zip(operator_hierarchy,
                    lumped_mass_hierarchy),
                n_smooth=param_multigrid['n_postsmooth'],
                mu_relax=param_multigrid['mu_relax'])
        # Construct coarse grid solver and set number of smoothing steps
        coarsegrid_solver = smoothers.Jacobi_LowestOrder(operator_hierarchy[0],
                                                         lumped_mass_hierarchy[0])
        coarsegrid_solver.n_smooth = param_multigrid['n_coarsesmooth']
        # Construct h-multigrid instance
        hmultigrid = preconditioners.hMultigrid(V_pressure_hierarchy,
                                                operator_hierarchy,
                                                presmoother_hierarchy,
                                                postsmoother_hierarchy,
                                                coarsegrid_solver)
        # For the higher-order case, also build an hp-multigrid instance
        if (param_mixed['higher_order']):
            if (param_multigrid['lump_mass']):
                velocity_mass_matrix_mg = lumped_mass_fine
            else:                
                velocity_mass_matrix_mg = full_mass_fine
            # Constuct operator and smoothers
            operator_mg = operators.Operator(V_pressure,
                                             V_velocity,
                                             velocity_mass_matrix_mg,
                                             omega)
            higherorder_presmoother = smoothers.Jacobi_HigherOrder(operator_mg,
                lumped_mass_fine,
                mu_relax=param_multigrid['mu_relax'],
                n_smooth=param_multigrid['n_presmooth'])
            higherorder_postsmoother = higherorder_presmoother
            # Construct hp-multigrid instance
            hpmultigrid = preconditioners.hpMultigrid(hmultigrid,
                                                      operator_mg,
                                                      higherorder_presmoother,
                                                      higherorder_postsmoother)
            preconditioner = hpmultigrid
        else:
            preconditioner = hmultigrid
    else:
        print 'Unknown preconditioner: \''+prec_name+'\'.'
        sys.exit(-1)

    pressure_ksp_monitor = ksp_monitor.KSPMonitor('pressure',
                                                  verbose=param_pressure['verbose'])

    if (param_pressure['lump_mass']):
        velocity_mass_matrix_op = lumped_mass_fine
    else:                
        velocity_mass_matrix_op = full_mass_fine

    operator = operators.Operator(V_pressure,
                                  V_velocity,
                                  velocity_mass_matrix_op,
                                  omega)

    # Construct pressure solver based on operator and preconditioner 
    # built above
    pressure_solver = solvers.PETScSolver(operator,
                                          preconditioner,
                                          param_pressure['ksp_type'],
                                          ksp_monitor=pressure_ksp_monitor,
                                          tolerance=param_pressure['tolerance'],
                                          maxiter=param_pressure['maxiter'])

    # Specify the lumped mass matrix to use in the Schur-complement
    # substitution
    if (param_mixed['lump_mass']):
        velocity_mass_matrix_schursub = lumped_mass_fine
    else:
        velocity_mass_matrix_schursub = full_mass_fine

    mixed_ksp_monitor = ksp_monitor.KSPMonitor('mixed',
                                               verbose=param_mixed['verbose'])
 
    # Construct mixed Helmholtz solver
    helmholtz_solver = helmholtz.PETScSolver(V_pressure,
                                             V_velocity,
                                             pressure_solver,
                                             param_mixed['ksp_type'],
                                             omega,
                                             velocity_mass_matrix = \
                                                velocity_mass_matrix_schursub,
                                             schur_diagonal_only = \
                                                param_mixed['schur_diagonal_only'],
                                             ksp_monitor=mixed_ksp_monitor,
                                             tolerance=param_mixed['tolerance'],
                                             maxiter=param_mixed['maxiter'])

    comm = MPI.COMM_WORLD
    if (comm.Get_rank() == 0):
        xml_root = ET.Element("SolverInformation")
        xml_tree = ET.ElementTree(xml_root)
        helmholtz_solver.add_to_xml(xml_root,"helmholtz_solver")
        xml_tree.write('solver.xml')


    # Right hand side function
    r_phi = Function(V_pressure).project(Expression('exp(-0.5*(x[0]*x[0]+x[1]*x[1])/(0.25*0.25))'))
    r_u = Function(V_velocity)
    r_u.assign(0.0)

    # Solve and return both pressure and velocity field
    phi, w = helmholtz_solver.solve(r_phi,r_u)
    conv_hist_filename = os.path.join(param_output['output_dir'],'history.dat')
    mixed_ksp_monitor.save_convergence_history(conv_hist_filename)

    # If requested, write fields to disk
    if (param_output['savetodisk']):
        # Write output to disk
        DFile_w = File(os.path.join(param_output['output_dir'],
                                    'velocity.pvd'))
        DFile_w << w
        DFile_phi = File(os.path.join(param_output['output_dir'],
                                      'pressure.pvd'))
        DFile_phi << phi

