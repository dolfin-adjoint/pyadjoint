# Builds a Docker image with dolfin-adjoint dev version built from
# git sources.
#
# Authors:
# Simon W. Funke <simon@simula.no>
# Jack S. Hale <jack.hale@uni.lu>
# Jørgen S. Dokken <dokken@simula.no>

FROM ghcr.io/scientificcomputing/fenics-gmsh:2023-04-21

RUN apt-get -qq update && \
    apt-get -y install libjsoncpp-dev curl python-is-python3 && \
    apt-get -y install python3-dev graphviz libgraphviz-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN /bin/bash -l -c "python3 -m pip install --no-cache --ignore-installed scipy"

WORKDIR /src/

ARG IPOPT_VERSION=3.12.9
ARG MOOLA_BRANCH="master"
ARG DOLFIN_ADJOINT_BRANCH="2023.0.0"

# Pull IPOPT and install Metis
RUN curl -O https://www.coin-or.org/download/source/Ipopt/Ipopt-${IPOPT_VERSION}.tgz && \
    tar -xvf Ipopt-${IPOPT_VERSION}.tgz && \
    ./Ipopt-${IPOPT_VERSION}/ThirdParty/Metis/get.Metis

# Build Ipopt (and fix compilation for parallel MPI versions)
WORKDIR /src/Ipopt-${IPOPT_VERSION}
RUN sed -i "s/#define MPI_COMM_WORLD IPOPT_MPI_COMM_WORLD//g" Ipopt/src/Algorithm/LinearSolvers/IpMumpsSolverInterface.cpp && \
    sed -i "s/MPI_COMM_WORLD/MPI_COMM_SELF/g" Ipopt/src/Algorithm/LinearSolvers/IpMumpsSolverInterface.cpp && \
    ./configure --with-blas="-lblas -llapack" --with-lapack="-llapack" --prefix="/usr/local"  --enable-debug --enable-shared --with-mumps-incdir="${PETSC_DIR}/${PETSC_ARCH}/include  -I/usr/include/x86_64-linux-gnu/mpich/" --with-mumps-lib="${PETSC_DIR}/${PETSC_ARCH}/lib" && \
    make install

# Build pyipopt
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}
WORKDIR /src
RUN python3 -m pip install --no-cache cython && \
    python3 -m pip install --no-cache ipopt

# Install moola
RUN /bin/bash -l -c "python3 -m pip install --no-cache git+https://github.com/funsim/moola@${MOOLA_BRANCH}"

# Install dolfin-adjoint
WORKDIR /src/
ARG DOLFIN_ADJOINT_REPO="https://github.com/dolfin-adjoint/pyadjoint"
RUN git clone --single-branch --branch=${DOLFIN_ADJOINT_BRANCH} ${DOLFIN_ADJOINT_REPO}
RUN /bin/bash -l -c "python3 -m pip install --no-cache -e ./pyadjoint/[test]"


# Run tests
WORKDIR /src/pyadjoint
RUN cd tests/pyadjoint && python3 -m pytest -xvs .
RUN cd tests/fenics_adjoint && python3 -m pytest -xvs .
RUN cd tests/migration && python3 -m pytest -xvs .

WORKDIR /src