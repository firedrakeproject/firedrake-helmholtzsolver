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
Set of classes for solving the mixed formulation of the 3d linear gravity wave system
equation in a 2+1 dimensionsal extruded mesh based on the 
`firedrake <http://www.firedrakeproject.org/>`_ package. 
Both a lowest- and higher- order matrix-free multigrid preconditioner
for the Schur complement pressure correction system are provided.
The PDE system which is solved
in the domain :math:`\Omega` is

.. math::
  \frac{\partial \vec{u}}{\partial t} = \nabla p + b\hat{\vec{z}}

  \frac{\partial p}{\partial t} = -c^2 \nabla\cdot\vec{u}

  \frac{\partial b}{\partial t}=-N^2\vec{u}\cdot\hat{\vec{z}}

where the pressure field :math:`p` is defined in a :math:`L_2` space and the velocity 
:math:`\vec{u}` is a :math:`H(div)` function space with vertical boundary condition
:math:`\vec{u}\cdot\vec{n}=0`. The buoyancy field is denoted by :math:`b`, and lives in
a function space which is equivalent to the vertical component of the velocity space.
The real and positive parameters :math:`c` and :math:`N` are the speed of sound and the 
buoyancy frequency.

For more details see `Notes in LaTeX <./GravityWaves.pdf>`_.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

