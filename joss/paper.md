---
title: "dolfin-adjoint 2018.1: automated adjoints for FEniCS and Firedrake" 
tags:
- finite element method
- optimization
- adjoint
- gradients
- partial differential equations
authors:
- name: Sebastian K. Mitusch
  orcid: 0000-0002-8888-6568
  affiliation: 1
- name: Simon W. Funke
  orcid: 0000-0003-4709-8415
  affiliation: 1
- name: Jørgen S. Dokken
  orcid: 0000-0001-6489-8858
  affiliation: 1
affiliations:
 - name: Simula Research Laboratory
   index: 1
date: 31 December 2018
bibliography: references.bib
---

# Summary

Adjoint models play an important role in scientific computing.
They enable for instance sensitivity and stability analysis, goal-oriented mesh adaptivity and optimisation.
However, the derivation and implementation of adjoint models is challenging, especially for models governed by  non-linear or time-dependent partial differential equations (PDEs).
In [@FarrellEtAl2013], the authors proposed to automatically derive adjoint models through high-level algorithmic differentiation, where the forward model is considered as  a sequence of variational problems.
The implementation, named dolfin-adjoint, automatically and robustly derives adjoint models for models written in the finite element software FEniCS [@LoggEtAl2012].
However, the assumption that the model consists of a sequence of variational problems can be limiting.
For instance when considering Dirichlet boundary conditions that are not explicitly stated in the variational formulation, when considering complex functionals that cannot be represented as an integral, or when coupling FEniCS to other non-PDE models.

We present a new implementation of dolfin-adjoint that overcomes these limitations.
The core of our implementation is a generic, operator-overloading based, algorithmic differentiation tool for Python called pyadjoint.
To apply pyadjoint to a Python module, one implements a *pyadjoint.Block* subclass for each module function which can recompute the function with new inputs and compute the function’s derivatives.
During runtime, pyadjoint builds a graph of Block instances, and applies the chain rule to automatically compute gradients and Hessian actions.
Further, pyadjoint includes gradient verification tools and an optimisation framework that interfaces external packages such as scipy, ipopt, moola and ROL.

To support automated adjoints for FEniCS and Firedrake [@RathgeberEtAl2017] models, we overloaded their user-interface functions.
In FEniCS and Firedrake, variational problems are represented in the domain-specific language UFL [@AlnaesEtAl2014].
UFL allows the definition and manipulation of discrete variational formulations, which we leverage to automatically obtain the desired equations in a format that can be solved by FEniCS/Firedrake.
This allows us to efficiently derive the adjoint and tangent-linear equations and solve them using the existing solver methods in FEniCS/Firedrake, as described in [@FarrellEtAl2013].
In addition, we have implemented support for computing the adjoint solution at the boundary, which enables the automatic differentiation of PDE solutions with respect to strongly imposed Dirichlet boundary conditions.

The dolfin-adjoint repository contains a wide range of tests and demos.
The demos are documented and available at www.dolfin-adjoint.org.


# Acknowledgements

We would like to thank Imperial College London and the Firedrake team for their contributions to pyadjoint and dolfin-adjoint.
A special thanks to Lawrence Mitchell for his work on the Firedrake specific implementations, and David Ham for his
input on strong Dirichlet boundary condition controls. 
Finally, thanks to everyone who has contributed to the pyadjoint repository.


# References
