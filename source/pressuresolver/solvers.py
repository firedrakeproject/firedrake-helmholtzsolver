from operators import *

##########################################################
# CG solver
##########################################################

class ConjugateGradient(InverseOperator):
    
##########################################################
# Constructor
##########################################################
    def __init__(self,operator,preconditioner,
                 maxiter=100,
                 tolerance=1.E-6,
                 mu_relax=2./3.,
                 verbose=2):
        super(ConjugateGradient,self).__init__(operator)
        self.preconditioner = preconditioner
        self.maxiter = maxiter
        self.tolerance = tolerance
        self.mu_relax = mu_relax
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
        z_val = z.vector().array()
        p = Function(self.V_pressure,val=z_val)
        res_norm_0 = sqrt(assemble(r*r*dx))
        rz = assemble(r*z*dx)
        if (self.verbose > 0):
            print '      Initial residual = ' + ('%8.4e' % res_norm_0)
        for i in range(self.maxiter):
            Ap = self.operator.apply(p)
            pAp = assemble(p*Ap*dx)
            alpha = rz/pAp
            phi += alpha*p
            r -= alpha*Ap
            res_norm = sqrt(assemble(r*r*dx))
            if (self.verbose > 1):
                print '     i = '+('%4d' % i) +  \
                      ' : '+('%8.4e' % res_norm) + \
                      ' [ '+('%8.4e' % (res_norm/res_norm_0))+' ] '
            if (res_norm/res_norm_0 < self.tolerance):
                break
            self.preconditioner.solve(r,z)
            rz_old = rz
            rz = assemble(r*z*dx)
            beta = rz/rz_old
            p = z + beta*p
        if (self.verbose > 0):
            if (res_norm/res_norm_0 < self.tolerance):
                print '  CG converged after '+str(i)+' iterations.'
            else:
                print '  CG failed to converge after '+str(maxiter)+' iterations.'

