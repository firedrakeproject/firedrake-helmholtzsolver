from firedrake import *
import numpy as np
import math

class Mountain(object):
    '''Isolated mountain with specified width :math:`w`, height :math:`h` and location.

        The position of the mountain is given by a unit vector :math:`\\vec{n}`, 
        which identifies a point on the sphere.
        For a given position :math:`\\vec{r}` in the mesh, define 
        :math:`r = |\\vec{r}|`, :math:`\cos(\\theta) = \\vec{n}\cdot\\vec{r}/r`,
        :math:`z=r-R_{earth}`, :math:`x=R_{earth}\\theta`. Then for 
        :math:`x<w` replace
        :math:`z\mapsto z_{new} = z+(1-z/H_{atmos})h(1+\cos(\pi x/w))/2`
        where :math:`R_{earth}` is the radius of the Earth and :math:`H_{atmos}` the 
        height of the atmosphere.

        :arg n: Position vector :math:`\\vec{n}`
        :arg width: Width :math:`w` of mountain
        :arg height: Height :math:`h` of mountain
        :arg r_earth: Radius of Earth
        :arg h_atmos: Height of atmosphere
    '''
    def __init__(self,n,width,height,r_earth,h_atmos):
        self.n = n
        self.width = width
        self.height = height
        self.h_atmos = h_atmos
        self.r_earth = r_earth
        self.dim = len(n)

    @property
    def steepness(self):
        '''Steepness defined as height/width'''
        return self.height/self.width

    def distort(self,mesh):
        '''Distort a mesh.

        Distort mesh according to formula given above

        :arg mesh: Mesh point to distort
        '''
        kernel = '''
            double r = sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
            double rn = (x[0]*%(N0)f+x[1]*%(N1)f+x[2]*%(N2)f);
            double y = r*acos(rn/r);
            if (y < %(WIDTH)f) {
                double rho = 1. + 0.5*(1.-(r-%(REARTH)f)/%(HATMOS)f)
                          * %(HEIGHT)f*(1.+cos(%(PI)f*y/%(WIDTH)f))/r;
                x[0] *= rho;
                x[1] *= rho;
                x[2] *= rho;
            }
        '''
        d = {'N0':self.n[0],'N1':self.n[1],'N2':self.n[2],
             'WIDTH':self.width,
             'HEIGHT':self.height,
             'REARTH':self.r_earth,
             'HATMOS':self.h_atmos,
             'PI':math.pi}
        par_loop(kernel % d,direct,{'x':(mesh.coordinates,RW)})
