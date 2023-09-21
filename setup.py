from itertools import chain
from setuptools import setup

extras = {
    'moola': ['moola>=0.1.6'],
    'test': ['pytest>=3.10', 'flake8', 'coverage'],
    'visualisation': ['tensorflow', 'protobuf',
                      'networkx', 'pygraphviz'
                     ],
    'meshing': ['pygmsh', 'meshio'],
    'doc': ['sphinx', 'sphinx-autobuild', 'sphinxcontrib-bibtex']
}
# 'all' includes all of the above
extras['all'] = list(chain(*extras.values()))

setup(name='pyadjoint',
      version='2023.0.0',
      description='High-level automatic differentiation library for FEniCS.',
      author='Jørgen Dokken',
      author_email='dokken@simula.no',
      packages=['firedrake_adjoint',
                'numpy_adjoint',
                'pyadjoint',
                'pyadjoint.optimization'],
      package_dir={'pyadjoint': 'pyadjoint',
                   'firedrake_adjoint': 'firedrake_adjoint',
                   'numpy_adjoint': 'numpy_adjoint'},
      install_requires=['scipy>=1.0', 'checkpoint-schedules'],
      extras_require=extras
      )
