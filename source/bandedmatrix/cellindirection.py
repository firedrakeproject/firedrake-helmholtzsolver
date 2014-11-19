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

        tdim = ele.cell().topological_dimension()
        element = fiat_from_ufl_element(ele)
        self._element = element
        self._ndof_bottom_facet = len(element.entity_dofs()[(tdim-1, 0)][0])
        self._ndof_cell = len(element.entity_dofs()[(tdim-1, 1)][0])
        self._ndof_h = element.A.space_dimension()
        self._ndof_v = element.B.space_dimension()
        self._ndof = element.space_dimension()
        self._name = "permutation_%d" % CellIndirection._count
        CellIndirection._count += 1

        # Where are the nodes?
        pe = [p.get_point_dict().keys()[0] for p in element.dual_basis()]
        # Permutation from reference-element ordering into "layers in cell" ordering
        # i.e. all the dofs with z=0, then z=1, z=2, ...
        self._permutation = zip(*sorted(zip(range(self._ndof), pe),
                                        key=lambda x: x[1][::-1]))[0]

    @property
    def name(self):
        """The name of the permutation map"""
        return self._name

    @property
    def permutation(self):
        """The permutation map itself"""
        return self._permutation

    @property
    def ndof(self):
        """The total number of dofs per cell"""
        return self._ndof

    @property
    def ndof_cell(self):
        """The number of dofs associated with the (tdim-1, 1) entity"""
        return self._ndof_cell

    @property
    def ndof_bottom_facet(self):
        """The number of dofs associated with the (tdim-1, 0) bottom facet"""
        return self._ndof_bottom_facet

    @property
    def horiz_extent(self):
        """The number of dofs in the horizontal direction"""
        return self._ndof_h

    @property
    def vert_extent(self):
        """The number of dofs in the vertical direction"""
        return self._ndof_v

    def maptable(self):
        """Return a C string declaring the permutation."""
        return "const int %s[%d] = {%s}" % \
            (self.name, self.ndof,
             ', '.join("%d" % p for p in self.permutation))

    def ki_to_local_index(self, k, i):
        """Return a C string mapping k and i to a cell-local dof index"""
        return "(%(name)s[(%(k)s %% %(nv)d)*%(nd)d + %(i)s])" % {'k': k,
                                                          'i': i,
                                                          'name': self.name,
                                                          'nv': self.vert_extent,
                                                          'nd': self.horiz_extent}
    def ki_to_index(self, k, i):
        """Return a C string mapping k and i to a dof index"""
        return "(((%(k)s / %(nv)d) * %(nd)d) + %(local_idx)s)" % \
            {'k': k,
             'i': i,
             'local_idx': self.ki_to_local_index(k, i),
             'nv': self.vert_extent,
             'nd': self.ndof}
