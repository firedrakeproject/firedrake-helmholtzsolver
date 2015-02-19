from firedrake import *
from firedrake.petsc import PETSc

class MixedArray(object):
    '''Class for combining the mixed dof vectors
        :math:`(\\vec{x}^{(1)},...,\\vec{x}^{(n)})` into one large vector.

        PETSc solvers operate on one large vector, which is a concatenation of
        the dof-vectors in the velocity, pressure and buoyancy spaces. This class
        can be used to work out the corresponding bounds and copy in and out of these
        mixed vector. 

        We store the combined mixed dof-vector in an array as 
        :math:`(x^{(1)}_0,...,x^{(1)}_{ndof_{x^{(1)}}-1},...,x^{(n)},...,x^{(n)}_{ndof_{x^{(n)}}-1})`

        :arg *args: List of function spaces 
    '''
    def __init__(self,*args):
        self._ndof = []
        self._min_idx = []
        self._max_idx = []
        self._ndof_total = 0
        self._iset = []
        for fs in args:
            ndof = fs.dof_dset.size
            self._ndof.append(ndof)
            min_idx = self._ndof_total
            self._min_idx.append(min_idx)
            self._ndof_total+=ndof
            max_idx = self._ndof_total
            self._max_idx.append(max_idx)
        # Build IndexSets
        v = PETSc.Vec().create()
        v.setSizes((self._ndof_total,None))
        v.setUp()
        for fs, min_idx, max_idx in zip(args,self._min_idx,self._max_idx):
            global_min_idx=v.owner_range[0] + min_idx
            iset = PETSc.IS().createStride(max_idx-min_idx,
                                           first=global_min_idx,
                                           step=1,
                                           comm=v.comm)
            self._iset.append(iset)

    def range(self,i):
        '''Range (min,max) for :math:`x^{(i)}`-dof-vector.

        :arg i: Function space index
        '''
        return (self._min_idx[i], self._max_idx[i])

    @property
    def ndof(self):
        '''Total length of combined dof-vector.

        This is the sum of the lengths of the :math:`x^{(1)}`,..., :math:`x^{(n)}` dof
        vectors.
        '''
        return self._ndof_total

    def combine(self,v,*args):
        '''Combine field given as separate components
            :math:`(\\vec{x}^{(1),...,\\vec{x}^{(n)})` into combined dof vector v
        
            :arg v: Resulting combined field :math:`v = (\\vec{x}^{(1)},...,\\vec{x}^{(n)})`
            :arg *args: Individual fields
        '''
        for i,x in enumerate(args):
            min_idx,max_idx = self.range(i)
            v.array[min_idx:max_idx] = x.array[:]
            

    def split(self,v,*args):
        '''Split field given as combined vector v into the components
            :math:`(\\vec{x}^{(1),...,\\vec{x}^{(n)})` 

            :arg v: Combined field :math:`v = (\\vec{x}^{(1)},...,\\vec{x}^{(n)})`
            :arg *args: Resulting individual fields
        '''
        for i,x in enumerate(args):
            iset = self._iset[i]
            tmp = v.getSubVector(iset)
            tmp.copy(x)
            v.restoreSubVector(iset, tmp)
