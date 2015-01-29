############################################################################### 
# General parameters
###############################################################################
General:
    warmup_run = True               # Carry out a warmup run first?
    nu_cfl = 10.0                   # CFL number
    speed_c = 300.0                 # Sound wave speed [m/s]
    speed_N = 0.01                  # Buoyancy frequency [1/s]
    matrixfree = True               # Use the matrix-free preconditioner
    orography = True                # Assume orography in outer loop

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
    nlayer = 32                     # Number of vertical layers
    thickness = 1.E4                # Thickness of spherical shell [m]
    nlevel = 4                      # Number of multigrid levels

############################################################################### 
# Mixed system parameters
############################################################################### 
Mixed system:
    ksp_type = gmres                # KSP type for PETSc solver
    higher_order = False            # Use higher order discretisation?
    schur_diagonal_only = False     # Use diagonal only in Schur complement?
    tolerance = 1.E-5               # tolerance
    maxiter = 20                    # maximal number of iterations
    verbose = 2                     # verbosity level

############################################################################### 
# Pressure solve parameters
############################################################################### 
Pressure solve:
    ksp_type = cg                   # KSP type for PETSc solver
    tolerance = 1.E-14              # tolerance
    maxiter = 3                   # maximal number of iterations
    verbose = 1                     # verbosity level

############################################################################### 
# Multigrid parameters
############################################################################### 
Multigrid:
    mu_relax = 1.0                  # multigrid smoother relaxation factor
    n_presmooth = 1                 # presmoothing steps
    n_postsmooth = 1                # postsmoothing steps
    n_coarsesmooth = 1              # number of coarse grid smoothing steps

