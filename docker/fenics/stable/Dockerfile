# Builds a Docker image with dolfin-adjoint stable version built from
# git sources. The image is at:
#
#    https://quay.io/repository/dolfinadjoint/dolfin-adjoint
#
# Authors:
# Simon W. Funke <simon@simula.no>
# Jack S. Hale <jack.hale@uni.lu>

FROM quay.io/fenicsproject/stable:latest
MAINTAINER Simon W. Funke <simon@simula.no>

USER root
RUN apt-get -qq update && \
    apt-get -y install libjsoncpp-dev && \
    apt-get -y install python-dev graphviz libgraphviz-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN /bin/bash -l -c "pip3 install --no-cache --ignore-installed scipy"

COPY --chown=fenics dolfin-adjoint.conf $FENICS_HOME/dolfin-adjoint.conf

ARG IPOPT_VER=3.12.9
RUN /bin/bash -l -c "source $FENICS_HOME/dolfin-adjoint.conf && \
                     update_ipopt && \
                     update_pyipopt"

ARG MOOLA_BRANCH="master"
RUN /bin/bash -l -c "pip3 install --no-cache git+git://github.com/funsim/moola.git@${MOOLA_BRANCH}"

ARG DOLFIN_ADJOINT_BRANCH="2019.1.0"
RUN /bin/bash -l -c "pip3 install --no-cache git+https://github.com/dolfin-adjoint/pyadjoint.git@${DOLFIN_ADJOINT_BRANCH}"

USER fenics
COPY --chown=fenics WELCOME $FENICS_HOME/WELCOME
RUN echo "source $FENICS_HOME/dolfin-adjoint.conf" >> $FENICS_HOME/.bash_profile

RUN /bin/bash -l -c "python3 -c \"import fenics_adjoint\""
RUN /bin/bash -l -c "python3 -c \"import dolfin; import pyadjoint.ipopt\""

USER root
