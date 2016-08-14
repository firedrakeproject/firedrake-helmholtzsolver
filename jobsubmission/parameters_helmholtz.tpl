############################################################################### 
# General parameters
###############################################################################
General:
    warmup_run = True               # Carry out a warmup run first?
    nu_cfl = %(nu_cfl)f             # CFL number
    speed_c = 300.0                 # Sound wave speed [m/s]
    speed_N = 0.01                  # Buoyancy frequency [1/s]
    solve_matrixfree = True         # Use matrix-free solver?
    solve_petsc = True              # Use PETSc solver
    n_gaussian = 16                 # Number of Gaussians for initialisation

############################################################################### 
# Output parameters
############################################################################### 
Output:
    output_dir = output             # Directory for output
    savetodisk = False              # Save fields to disk?

################################################################################ 
# Grid parameters
################################################################################ 
Grid:
    ref_count_coarse = %(ref_count_coarse)d           
                                    # Number of refinement levels to construct 
                                    # coarsest multigrid level
    r_earth = 6.371E6               # Radius of earth
    nlayer = 64                     # Number of vertical layers
    thickness = 1.E4                # Thickness of spherical shell [m]
    nlevel = %(n_level)d            # Number of multigrid levels

################################################################################ 
# Orography parameters
################################################################################ 
Orography:
    enabled = False                 # Enable orography?
    height = 2.E3                   # Height of mountain in [m]
    width = 2.E3                   # Width of mountain in [m]

############################################################################### 
# Mixed system parameters
############################################################################### 
Mixed system:
    ksp_type = gmres                # KSP type for PETSc solver
    higher_order = %(higher_order)s # Use higher order discretisation?
    schur_diagonal_only = False     # Use diagonal only in Schur complement?
    tolerance = 1.E-5               # tolerance
    maxiter = 500                    # maximal number of iterations
    verbose = 2                     # verbosity level

############################################################################### 
# Pressure solve parameters
############################################################################### 
Pressure solve:
    ksp_type = preonly              # KSP type for PETSc solver
    tolerance = 1.E-14              # tolerance
    maxiter = 1                     # maximal number of iterations
    verbose = 0                     # verbosity level
    multigrid = %(multigrid)s       # Use multigrid?

############################################################################### 
# Multigrid parameters
############################################################################### 
Multigrid:
    mu_relax = 0.8                  # multigrid smoother relaxation factor
    n_presmooth = 1                 # presmoothing steps
    n_postsmooth = 1                # postsmoothing steps
    n_coarsesmooth = %(ncoarsesmooth)d  # number of coarse grid smoothing steps
    direct_coarse_solver = %(direct_coarse_solver)s # Use direct solver on coarsest grid?

################################################################################ 
# Single level preconditioner parameters
################################################################################ 
Singlelevel:
    mu_relax = 0.8                  # single level smoother relaxation factor
    n_smooth = 2		    # Number of smoother iterations