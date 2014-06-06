from operators import *

##########################################################
# Richardson iteration solver
##########################################################

class LoopSolver(object):

##########################################################
# Constructor
##########################################################
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

##########################################################
# Solve
##########################################################
    def solve(self,b,phi):
        if (self.verbose > 0):
            print '    -- Loop solver --'
        residual = Function(self.V_pressure)
        error = Function(self.V_pressure)
        error.assign(0.0)
        residual.assign(self.operator.residual(b,phi))
        res_norm_0 = sqrt(assemble(residual*residual*self.dx))
        if (self.verbose > 1):
            print '      Initial residual = ' + ('%8.4e' % res_norm_0)
        for i in range(1,self.maxiter+1):
            self.preconditioner.solveApprox(residual,error)
            phi.assign(phi+error)
            residual.assign(self.operator.residual(b,phi))
            res_norm = sqrt(assemble(residual*residual*self.dx))
            if (self.verbose > 1):
                print '     i = '+('%4d' % i) +  \
                      ' : '+('%8.4e' % res_norm) + \
                      ' [ '+('%8.4e' % (res_norm/res_norm_0))+' ] '
            if (res_norm/res_norm_0 < self.tolerance):
                break
        if (self.verbose > 0):
            if (res_norm/res_norm_0 < self.tolerance):
                print '  Multigrid converged after '+str(i)+' iterations.'
            else:
                print '  Multigrid failed to converge after '+str(self.maxiter)+' iterations.'


##########################################################
# CG solver
##########################################################

class ConjugateGradient(object):
    
##########################################################
# Constructor
##########################################################
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
        self.verbose = verbose

##########################################################
# Solve
##########################################################
    def solve(self,b,phi):
        if (self.verbose > 0):
            print '    -- CG solve --'
        r = self.operator.residual(b,phi)
        z = Function(self.V_pressure)
        self.preconditioner.solveApprox(r,z)
        p = Function(self.V_pressure)
        p.assign(z)
        res_norm_0 = sqrt(assemble(r*r*dx))
        rz = assemble(r*z*dx)
        if (self.verbose > 1):
            print '      Initial residual = ' + ('%8.4e' % res_norm_0)
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
                print '     i = '+('%4d' % i) +  \
                      ' : '+('%8.4e' % res_norm) + \
                      ' [ '+('%8.4e' % (res_norm/res_norm_0))+' ] '
            if (res_norm/res_norm_0 < self.tolerance):
                break
            self.preconditioner.solveApprox(r,z)
            rz_old = rz
            rz = assemble(r*z*dx)
            beta.assign(rz/rz_old)
            p.assign(z + beta*p)
        if (self.verbose > 0):
            if (res_norm/res_norm_0 < self.tolerance):
                print '  CG converged after '+str(i)+' iterations.'
            else:
                print '  CG failed to converge after '+str(self.maxiter)+' iterations.'

