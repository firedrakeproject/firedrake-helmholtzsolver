.. Helmholtzsolver documentation master file, created by
   sphinx-quickstart on Sat Jun  7 13:21:01 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Helmholtzsolver's documentation!
===========================================

Contents:

.. toctree::
   :maxdepth: 2

   modules

===============
Overview
===============
Set of classes for solving the mixed formulation of the Helmholtz
equation in two dimensions based on the `firedrake <http://www.firedrakeproject.org/>`_ package. 
Different matrix-free preconditioner 
for the Schur complement pressure correction system are provided,
including a geometric multigrid solver. The equation which is solved
in the domain :math:`\Omega` is

.. math::
  
  \phi + \omega (\nabla\cdot\phi^*\vec{u}) = r_\phi

  \vec{u} - \omega \nabla{\phi} = \vec{r}_u

where the pressure field :math:`\phi` is defined in a :math:`DG` space and the velocity 
:math:`\vec{u}` is a :math:`H(div)` function space. :math:`omega` is a real and positive
parameter. Currently the field :math:`\phi^*` is set to a constant value 1 and only the lowest order DG space (:math:`P0`) and lowest order Raviart Thomas elements (:math:`RT1`) are supported.

For more details see `Notes in LaTeX <./FEMmultigrid.pdf>`_.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

