from locallyassembledmatrix import *

class HybridSolver(object):
    '''Solver for hybridised system.

        The hybridised system has the form

        .. math::
        
            \\begin{pmatrix}
              M_u        & -\omega D & Q \\\\
              \omega D^T & M_p       & 0 \\\\
              Q^T        & 0         & 0
            \end{pmatrix}
            \\begin{pmatrix}
             \\vec{U} \\\\ \\vec{P} \\\\ \\vec{\Lambda}
            \end{pmatrix}
            =
            \\begin{pmatrix}
             \\vec{R}_u \\\\ \\vec{R}_p \\\\ 0
            \end{pmatrix}

        where :math:`M_x` are mass matrices, :math:`D_{ij}=\int p_i \div(\\vec{u}_i)\;dx`
        is the weak derivative matrix and
        :math:`Q_{ij}=\int \lambda_i [[\\vec{u}\cdot\\vec{n}]]\; dx`.

        The function spaces are assumed to be discontinuous

        :arg W2: discontinuous HDiv space for velocity
        :arg W3: L2 space for pressure
        :arg Wtrace: Trace space for Lagrangian multipliers
        :arg omega: parameter :math:`\omega`
    '''
    def __init__(self,W2,W3,Wtrace,omega=1.0):
        self._W2 = W2
        self._W3 = W3
        self._omega = omega
        self._mesh = self._W3.mesh()
        self._dx = self._mesh._dx
        self._build_Ainv()

    def _build_Ainv(self):
        '''Build inverse of system matrix :math:`A`.

            The system matrix :math:`A`is defined as

            .. math::

                A = 
                \\begin{pmatrix}
                  M_u        & -\omega D \\\\
                  \omega D^T & M_p
                \end{pmatrix}
             
            It's inverse is given as

            .. math::
                A^{-1} = 
                \\begin{pmatrix}
                A^{(inv)}_{uu} &
                A^{(inv)}_{up} \\\\
                A^{(inv)}_{pu} &
                A^{(inv)}_{pp} &
                \end{pmatrix}
                = 
                \\begin{pmatrix}
                  M_u^{-1} - \omega^2 M_u^{-1} DH^{-1}D^T M_u^{-1} & 
                  \omega M_u^{-1} DH^{-1} \\\\
                  -\omega H^{-1} D^T M_u^{-1} & 
                  H^{-1}
                \end{pmatrix}

            with the Helmholz operator

            .. math::
                
                H = M_p + \omega^2 D^T M_u^{-1} D
        '''
        u_test = TestFunction(self._W2)
        u_trial = TestFunction(self._W2)
        p_test = TestFunction(self._W3)
        p_trial = TestFunction(self._W3)

        mat_Mu = LocallyAssembledMatrix(self._W2,self._W2,dot(u_test,u_trial)*self._dx)
        mat_Mp = LocallyAssembledMatrix(self._W3,self._W3,p_test*p_trial*self._dx)
        mat_D  = LocallyAssembledMatrix(self._W3,self._W2,p_test*div(u_trial)*self._dx)
        mat_DT = mat_D.transpose()
        mat_Mu_inv = mat_Mu.inverse() 
        # Helmholtz operator H = M_p + omega^2 D^T.M_u^{-1}.D
        mat_H  = mat_Mp.matadd(mat_DT.matmul(mat_Muinv.matmul(mat_D)),omega=self._omega**2)
        mat_H_inv = mat_H.inverse()
        # A_{up} = omega*M_u^{-1}.D.H^{-1}
        self.mat_Ainv_up = mat_Mu_inv.matmul(mat_D.matmul(mat_H_inv))
        self.mat_Ainv_up.scale(self._omega)
        # A_{pu} = -A_{up}^T
        self.mat_Ainv_pu = mat_up.transpose()
        self.mat_Ainv_pu.scale(-1.0)
        # A_{pp} = H^{-1}
        self.mat_Ainv_pp = mat_H_inv
        # A_{uu} = M_u^{-1} - omega^2 M_u^{-1}.D.H^{-1}.D^T.M_u^{-1}
        # = M_u^{-1} + A_{up}.A_{pp}.A_{pu}
        self.mat_Ainv_uu = Mu_inv.matadd(selt.mat_Ainv_up.matmul(self.mat_Ainv_pu))

    def solve(self):
        pass
