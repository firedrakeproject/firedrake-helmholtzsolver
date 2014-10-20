################################################################################ 
# General parameters
################################################################################ 
General:
    use_petscsplitsolver = False    # Solve using the PETSc split solver?
    use_matrixfreesolver = True     # Use the matrix-free solver

################################################################################ 
# Output parameters
################################################################################ 
Output:
    output_dir = output             # Directory for output
    savetodisk = False              # Save fields to disk?

################################################################################ 
# Grid parameters
################################################################################ 
Grid:
    ref_count_coarse = 3            # Number of refinement levels to construct 
                                    # coarsest multigrid level
    nlevel = 4                      # Number of multigrid levels

################################################################################ 
# Mixed system parameters
################################################################################ 
Mixed system:
    use_petscsplitsolver = False    # Solve using the PETSc split solver?
    use_matrixfreesolver = True     # Use the matrix-free solver
    ksp_type = gmres                # KSP type for PETSc solver
    higher_order = %(higher_order)s # Use higher order discretisation?
    lump_mass = %(lump_mass)s       # Lump mass matrix in Schur complement substitution
    schur_diagonal_only = False     # Use diagonal only in Schur complement preconditioner
    preconditioner = Multigrid      # Preconditioner to use: Multigrid or Jacobi (1-level method)
    tolerance = 1.E-5               # tolerance
    maxiter = 40                     # maximal number of iterations
    verbose = 2                     # verbosity level

################################################################################ 
# Pressure solve parameters
################################################################################ 
Pressure solve:
    ksp_type = %(inner_solver)s           # KSP type for PETSc solver
    lump_mass  = %(lump_mass)s      # Lump mass in Helmholtz operator in pressure space
    tolerance = 1.E-14              # tolerance
    maxiter = %(maxiter_inner)d     # maximal number of iterations
    verbose = 2                    # verbosity level

################################################################################ 
# Multigrid parameters
################################################################################ 
Multigrid:
    lump_mass = True                # Lump mass in multigrid
    mu_relax = 1.00                  # multigrid smoother relaxation factor
    n_presmooth = 1                 # presmoothing steps
    n_postsmooth = 1                # postsmoothing steps
    n_coarsesmooth = 1              # number of coarse grid smoothing steps

