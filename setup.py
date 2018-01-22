from distutils.core import setup

setup(name='dolfin_adjoint',
      version='2017.2.0',
      description='High-level automatic differentiation library for FEniCS.',
      author='Sebastian Kenji Mitusch',
      author_email='sebastkm@math.uio.no',
      packages=['fenics_adjoint', 'fenics_adjoint.types', 'dolfin_adjoint', 'firedrake_adjoint', 'pyadjoint', 'pyadjoint.optimization'],
      package_dir = {'fenics_adjoint': 'fenics_adjoint', 'pyadjoint': 'pyadjoint',
                     'firedrake_adjoint': 'firedrake_adjoint', 'dolfin_adjoint': 'dolfin_adjoint'},
      install_requires=['networkx', 'scipy', 'pytest', 'sphinx', 'sphinxcontrib-bibtex', 'moola'],
      dependency_links=['git+https://github.com/funsim/moola.git@master ']
     )
