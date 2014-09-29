import sys

from parameters import Parameters

class HelmholtzParameters(object):
    def __init__(self,filename=None):
        '''Class for Helmholtz parameters.
    
            :arg filename: Optional file to read from
        '''
        self.filename = filename
    
        self.param_output = Parameters('Output',
            # Directory for output
            {'output_dir':'output',
            # Save fields to disk?
            'savetodisk':False})

        # Grid parameters
        self.param_grid = Parameters('Grid',
            # Number of refinement levels to construct coarsest multigrid level
            {'ref_count_coarse':3,
            # Number of multigrid levels
            'nlevel':4})

        # Mixed system parameters
        self.param_mixed = Parameters('Mixed system',
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
        self.param_pressure = Parameters('Pressure solve',
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
        self.param_multigrid = Parameters('Multigrid',
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

        if self.filename:
            for param in (self.param_output,
                          self.param_grid,
                          self.param_mixed,
                          self.param_pressure,
                          self.param_multigrid):
                param.read_from_file(filename)

    def __str__(self):
        '''Convert to string representation.
        '''
        s = ''
        for param in (self.param_output,
                      self.param_grid,
                      self.param_mixed,
                      self.param_pressure,
                      self.param_multigrid):
            s += str(param)+'\n'
        return s
