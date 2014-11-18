from ufl import HDiv
from firedrake.fiat_utils import *


class CellIndirection(object):

    _count = 0

    def __init__(self, fs):
        """Build a representation of intra-cell indirections

        :arg fs: the function space to inspect for the element dof
        ordering.

        This provides maps from (k, i) indices where k is the vertical
        offset inside the cell and i is the horizontal offset to the
        dof on the cell.
        """
        ufl_ele = fs.ufl_element()
        # Unwrap non HDiv'd element if necessary
        if isinstance(ufl_ele, HDiv):
            ele = ufl_ele._element
        else:
            ele = ufl_ele

        element = fiat_from_ufl_element(ele)
        self._element = element
        self._n_h = element.A.space_dimension()
        self._n_v = element.B.space_dimension()
        self._ndof = self._n_h * self._n_v
        self._name = "permutation_%d" % CellIndirection._count
        CellIndirection._count += 1

        # Where are the nodes?
        pe = [p.get_point_dict().keys()[0] for p in element.dual_basis()]
        # Permutation from reference-element ordering into "layers in cell" ordering
        # i.e. all the dofs with z=0, then z=1, z=2, ...
        self._permutation = zip(*sorted(zip(range(self._n_h*self._n_v), pe),
                                        key=lambda x: x[1][::-1]))[0]

    def maptable(self):
        """Return a C string declaring the permutation."""
        return "const int %s[%d] = {%s}" % \
            (self._name, len(self._permutation),
             ', '.join("%d" % p for p in self._permutation))

    def ki_to_index(self, k, i):
        """Return a C string mapping k and i to a dof index"""
        return "((%(k)s / %(nv)d) * %(nd)d + %(name)s[(%(k)s %% %(nv)d) + %(i)s])" % \
            {'k': k,
             'i': i,
             'name': self._name,
             'nv': self._n_v,
             'nd': self._ndof}
