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
ARG DOLFIN_ADJOINT_BRANCH="release"
ARG MOOLA_BRANCH="master"

USER fenics
ENV IPOPT_VER=3.12.9

COPY dolfin-adjoint.conf $FENICS_HOME/dolfin-adjoint.conf

USER root
RUN apt-get -qq update && \
    apt-get -y install libjsoncpp-dev patchelf && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN /bin/bash -l -c "source $FENICS_HOME/dolfin-adjoint.conf && \
                     install_pyadjoint && \
                     update_ipopt && \
                     update_pyipopt"

USER fenics
COPY WELCOME $FENICS_HOME/WELCOME
RUN echo "source $FENICS_HOME/dolfin-adjoint.conf" >> $FENICS_HOME/.bash_profile

RUN /bin/bash -l -c "python3 -c \"import fenics_adjoint\""
RUN /bin/bash -l -c "python3 -c \"import dolfin; import pyipopt\""

USER root
