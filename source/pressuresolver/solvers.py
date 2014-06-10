from operators import *

class IterativeSolver(object):
    '''Abstract iterative solver base class.

    The solver converges if the relative residual has been reduced by at least a
    factor tolerance.

    :arg operator: Instance :math:`H` of linear Schur complement :class:`.Operator` in 
        pressure space
    :arg preconditioner: Instance :math:`P` of :class:`.Preconditioner`
    :arg maxiter: Maximal number of iterations
    :arg tolerance: Relative tolerance for solve
    :arg verbose: Verbosity level (0=no output, 1=minimal output, 2=show convergence rates)
    '''
    def __init__(self,operator,
                 preconditioner,
                 maxiter=100,
                 tolerance=1.E-6,
                 verbose=2):
        self.operator = operator
        self.V_pressure = self.operator.V_pressure
        self.preconditioner = preconditioner
        self.maxiter = maxiter
        self.tolerance = tolerance
        self.dx = self.V_pressure.mesh()._dx
        self.verbose = verbose
        self.fmt = '  {iter: >10}  {res: ^10}  {rel_res: ^16}  {rho: ^20}'

    def solve(self,b,phi):
        '''Solve linear system :math:`H\phi = b`.

        :arg b: Right hand side :math:`b` in pressure space
        :arg phi: State vector :math:`\phi` in pressure space
        '''
        pass

class LoopSolver(IterativeSolver):
    '''Loop solver (preconditioned Richardson iteration) 
    
    :arg operator: Instance :math:`H` of linear Schur complement :class:`.Operator` in 
        pressure space
    :arg preconditioner: Instance :math:`P` of :class:`.Preconditioner`
    :arg maxiter: Maximal number of iterations
    :arg tolerance: Relative tolerance for solve
    :arg verbose: Verbosity level (0=no output, 1=minimal output, 2=show convergence rates)
    '''
    def __init__(self,operator,
                 preconditioner,
                 maxiter=100,
                 tolerance=1.E-6,
                 verbose=2):
        super(LoopSolver,self).__init__(operator,preconditioner,maxiter,tolerance,verbose)

    def solve(self,b,phi):
        '''Solve linear system :math:`H\phi = b`.

        Solve iteratively using the preconditioned Richardson iteration

        .. math::
            
            \phi \mapsto \phi + P^{-1} (b-H\phi)
        
        :arg b: Right hand side :math:`b` in pressure space
        :arg phi: State vector :math:`\phi` in pressure space
        '''    
        if (self.verbose > 0):
            print '    -- Loop solver --'
        residual = Function(self.V_pressure)
        error = Function(self.V_pressure)
        error.assign(0.0)
        residual.assign(self.operator.residual(b,phi))
        res_norm_0 = sqrt(assemble(residual*residual*self.dx))
        res_norm_old = res_norm_0
        if (self.verbose > 1):
            print '      Initial residual = ' + ('%8.4e' % res_norm_0)
            print self.fmt.format(iter='iter i',
                                  res='||r_i||',
                                  rel_res='||r_i||/||r_0||',
                                  rho='||r_i||/||r_{i-1}||')
        for i in range(1,self.maxiter+1):
            self.preconditioner.solve(residual,error)
            phi.assign(phi+error)
            residual.assign(self.operator.residual(b,phi))
            res_norm = sqrt(assemble(residual*residual*self.dx))
            if (self.verbose > 1):
                print self.fmt.format(iter='{0:4d}'.format(i),
                                      res='{0:8.4e}'.format(res_norm),
                                      rel_res='{0:8.4e}'.format(res_norm/res_norm_0),
                                      rho='{0:6.3f}'.format(res_norm/res_norm_old))
            if (res_norm/res_norm_0 < self.tolerance):
                break
            res_norm_old = res_norm
        if (self.verbose > 0):
            if (res_norm/res_norm_0 < self.tolerance):
                print '  Multigrid converged after '+str(i)+' iterations.'
            else:
                print '  Multigrid failed to converge after '+str(self.maxiter)+' iterations.'


class CGSolver(IterativeSolver):
    '''Preconditioned Conjugate gradient solver.
    
    :arg operator: Instance :math:`H` of linear Schur complement :class:`.Operator` in 
        pressure space
    :arg preconditioner: Instance :math:`P` of :class:`.Preconditioner`
    :arg maxiter: Maximal number of iterations
    :arg tolerance: Relative tolerance for solve
    :arg verbose: Verbosity level (0=no output, 1=minimal output, 2=show convergence rates)
    '''
    def __init__(self,operator,
                 preconditioner,
                 maxiter=100,
                 tolerance=1.E-6,
                 verbose=2):
        super(CGSolver,self).__init__(operator,preconditioner,maxiter,tolerance,verbose)

    def solve(self,b,phi):
        '''Solve linear system :math:`H\phi = b`.

        Solve iteratively using the preconditioned CG iteration
        
        :arg b: Right hand side :math:`b` in pressure space
        :arg phi: State vector :math:`\phi` in pressure space
        '''    
        if (self.verbose > 0):
            print '    -- CG solve --'
        r = self.operator.residual(b,phi)
        z = Function(self.V_pressure)
        self.preconditioner.solve(r,z)
        p = Function(self.V_pressure)
        p.assign(z)
        res_norm_0 = sqrt(assemble(r*r*dx))
        res_norm_old = res_norm_0
        rz = assemble(r*z*dx)
        if (self.verbose > 1):
            print '      Initial residual = ' + ('%8.4e' % res_norm_0)
            print self.fmt.format(iter='iter i',
                                  res='||r_i||',
                                  rel_res='||r_i||/||r_0||',
                                  rho='||r_i||/||r_{i-1}||')
        alpha = Constant(0)
        beta = Constant(0)
        for i in range(self.maxiter):
            Ap = self.operator.apply(p)
            pAp = assemble(p*Ap*dx)
            alpha.assign(rz/pAp)
            phi += alpha*p
            r -= alpha*Ap
            res_norm = sqrt(assemble(r*r*dx))
            if (self.verbose > 1):
                print self.fmt.format(iter='{0:4d}'.format(i),
                                      res='{0:8.4e}'.format(res_norm),
                                      rel_res='{0:8.4e}'.format(res_norm/res_norm_0),
                                      rho='{0:6.3f}'.format(res_norm/res_norm_old))
            if (res_norm/res_norm_0 < self.tolerance):
                break
            self.preconditioner.solve(r,z)
            rz_old = rz
            rz = assemble(r*z*dx)
            beta.assign(rz/rz_old)
            p.assign(z + beta*p)
            res_norm_old = res_norm
        if (self.verbose > 0):
            if (res_norm/res_norm_0 < self.tolerance):
                print '  CG converged after '+str(i)+' iterations.'
            else:
                print '  CG failed to converge after '+str(self.maxiter)+' iterations.'

