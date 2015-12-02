import math
import numpy as np

def init_random(seed,verbose=False):
    '''Initialise random number generator with fixed seed.

        This is required to obtain reproducible results.

        :arg seed: Seed for numpy random number generator
        :arg verbose: Verbosity level
    '''
    np.random.seed(seed)
    if (verbose):
        r, = np.random.rand(1)
        print 'random number = ',r


class GaussianExpression(object):
    '''Expression for Gaussian.

    Defines a Gaussian which is centred in the direction given by the vector :math:`\\vec{n}_0`
    and the radius :math:`r_0`. The width (in angular distance from the direction 
    :math:`\\vec{n}_0`) is given as :math:`\sigma_{\\theta}` and in the height it is given as
    :math:`\sigma_r`. The amplitude is :math:`A`.

        .. maths:
            A\cdot exp(-((\\theta-\\theta_0)^2/\sigma_{\\theta}^2+(r-r_0)^2/\sigma_r^2)/2)

    where we have for the angular distance
    :math:`\cos(\\theta-\\theta_0) = \\vec{x}\cdot\\vec{n}_0/(|\\vec{x}|\cdot|\\vec{n}_0|)`

    :arg n0: Direction :math:`\\vec{n}_0` of centre
    :arg r0: Radial distance :math:`r_0` on centre
    :arg sigma_theta: Width :math:`\sigma_{\\theta}` in angular direction
    :arg sigma_r: Width :math: `\sigma_r` in radial direction
    :arg amplitude: Amplitude :math:`A`
    '''
    def __init__(self,n0,r0,sigma_theta,sigma_r,amplitude=1):
        self._n0 = n0
        self._r0 = r0
        self._sigma_theta = sigma_theta
        self._sigma_r = sigma_r
        self._amplitude = amplitude

    def _r_str(self):
        '''Expression for :math:`|\\vec{x}|=\sqrt{x_0^2+x_1^2+x_2^2}`'''
        return 'sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])'

    def _theta_str(self):
        '''Expression for angular distance from mean

            Returns :math:`theta=\\acos(\\vec{x}\cdot\\vec{n}_0/(|\\vec{x}|\cdot|\\vec{n}_0|))`
        '''
        n_cross_x = 'sqrt(0.0'
        n_dot_x = '0.0'
        for i in range(3):
            tmp = 'x['+str(i)+']*(%(N'+str((i+1)%3)+')f)-x['+str((i+1)%3)+']*(%(N'+str(i)+')f)'
            n_cross_x += '+('+tmp+')*('+tmp+')'
            n_dot_x += '+x['+str(i)+']*(%(N'+str(i)+')f)'
        n_cross_x += ')'
        s = 'atan2('+n_cross_x+','+n_dot_x+')'
        d = {'N0':self._n0[0],
             'N1':self._n0[1],
             'N2':self._n0[2]}
        return s % d

    def _dr_str(self):
        '''Expression for radial distance from mean :math:`r-r_0`'''
        s = '(%(RSTR)s-%(R0)f)'
        d = {'RSTR':self._r_str(),
             'R0':self._r0}
        return s % d

    def __str__(self):
        '''c-expression for Gaussian'''
        s = '%(AMPLITUDE)f*exp(-0.5*(pow(%(THETASTR)s/%(SIGMATHETA)f,2.)+pow(%(DRSTR)s/%(SIGMAR)f,2.)))'
        d = {'AMPLITUDE':self._amplitude,
             'THETASTR':self._theta_str(),
             'DRSTR':self._dr_str(),
             'SIGMATHETA':self._sigma_theta,
             'SIGMAR':self._sigma_r}
        return s % d

class MultipleGaussianExpression(object):
    '''Multiple Gaussian expressions.

    This is the sum of expressions of the form :class:`GaussianExpression`
    with variable directions, widths and heights. The directions are drawing randomly
    and are uniformly distributed over the surface of a sphere. Same for mean heights, which
    are uniformly distributed through the entire thickness of the atmosphere.
    The widths both in the angle and in height decrease uniformly, with the maximal angular 
    width being 1.0 and the maximal width in the radial direction being the thickness
    of the atmospher.

    :arg n_gaussian: Number of Gaussians
    :arg r_earth: Radius of the earth
    :arg thickness: Thickness of the atmosphere
    '''
    def __init__(self,n_gaussian,r_earth,thickness):
        self._n_gaussian = n_gaussian
        self._r_earth = r_earth
        self._thickness = thickness

    def __str__(self):
        '''c-Expression for multiple Gaussians'''
        s = ''
        for i in range(self._n_gaussian):
            nrm = 0.0
            while ( nrm < 0.5 ) or (nrm > 1.0):
                n = 2.*np.random.rand(3)-1.
                nrm = np.linalg.norm(n)
            r0 = self._r_earth + self._thickness*np.random.rand()
            sigma_theta = 1.0-0.9*(i/float(self._n_gaussian))
            sigma_r = (1.0-0.9*(i/float(self._n_gaussian)))*self._thickness
            amplitude = 1.0
            g = GaussianExpression(n,r0,sigma_theta,sigma_r,amplitude)
            s += '+'+str(g)
        return s
        
