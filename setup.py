from distutils.core import setup

setup(name='dolfin_adjoint2',
      version='0.0.1',
      description='High-level automatic differentiation library for FEniCS.',
      author='Sebastian Kenji Mitusch',
      author_email='sebastkm@math.uio.no',
      packages=['fenics_adjoint', 'pyadjoint', 'firedrake_adjoint'],
      package_dir = {'fenics_adjoint': 'fenics_adjoint', 'pyadjoint': 'pyadjoint',
                     'firedrake_adjoint': 'firedrake_adjoint'},
     )
