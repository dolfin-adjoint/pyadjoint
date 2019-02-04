from itertools import chain
from setuptools import setup

extras = {
    'moola': ['moola>=0.1.6'],
    'test': ['pytest'],
    'doc': ['sphinx', 'sphinxcontrib-bibtex'],
    'visualisation': ['tensorflow'],
}
# 'all' includes all of the above
extras['all'] = list(chain(*extras.values()))

setup(name='dolfin_adjoint',
      version='2018.1.0',
      description='High-level automatic differentiation library for FEniCS.',
      author='Sebastian Kenji Mitusch',
      author_email='sebastkm@math.uio.no',
      packages=['fenics_adjoint',
                'fenics_adjoint.types',
                'dolfin_adjoint',
                'firedrake_adjoint',
                'firedrake_adjoint.types',
                'pyadjoint',
                'pyadjoint.optimization'],
      package_dir={'fenics_adjoint': 'fenics_adjoint', 'pyadjoint': 'pyadjoint',
                   'firedrake_adjoint': 'firedrake_adjoint', 'dolfin_adjoint': 'dolfin_adjoint'},
      install_requires=['scipy'],
      extras_require=extras,
      )
