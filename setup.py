from setuptools import setup

setup(name='dolfin_adjoint',
      version='2018.1.0.r1',
      description='High-level automatic differentiation library for FEniCS and Firedrake.',
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
      install_requires=['scipy', 'pytest', 'sphinx', 'sphinxcontrib-bibtex', 'moola>=0.1.6', 'tensorflow'],
      )
