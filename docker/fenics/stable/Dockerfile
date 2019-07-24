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
ARG DOLFIN_ADJOINT_BRANCH="2019.1.0"
ARG MOOLA_BRANCH="master"

USER fenics
ENV IPOPT_VER=3.12.9

COPY dolfin-adjoint.conf $FENICS_HOME/dolfin-adjoint.conf

USER root
RUN apt-get -qq update && \
    apt-get -y install libjsoncpp-dev && \
    apt-get -y install python-dev graphviz libgraphviz-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN /bin/bash -l -c "pip3 install git+git://github.com/funsim/moola.git@${MOOLA_BRANCH}"
RUN /bin/bash -l -c "pip3 install git+https://bitbucket.org/dolfin-adjoint/pyadjoint.git@${DOLFIN_ADJOINT_BRANCH}"
RUN /bin/bash -l -c "pip3 install --ignore-installed scipy"
RUN /bin/bash -l -c "source $FENICS_HOME/dolfin-adjoint.conf && \
                     update_ipopt && \
                     update_pyipopt"

USER fenics
COPY WELCOME $FENICS_HOME/WELCOME
RUN echo "source $FENICS_HOME/dolfin-adjoint.conf" >> $FENICS_HOME/.bash_profile

RUN /bin/bash -l -c "python3 -c \"import fenics_adjoint\""
RUN /bin/bash -l -c "python3 -c \"import dolfin; import pyipopt\""

USER root
