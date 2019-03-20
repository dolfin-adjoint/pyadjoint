# Builds a Docker image with dolfin-adjoint stable version built from
# git sources. The image is at:
#
# Authors:
# Simon W. Funke <simon@simula.no>
# Jørgen Dokken <dokken@simula.no>

FROM firedrakeproject/firedrake:latest
MAINTAINER Simon W. Funke <simon@simula.no>
ARG DOLFIN_ADJOINT_BRANCH="master"
ARG MOOLA_BRANCH="master"

USER firedrake


USER root
RUN apt-get -qq update && \
    apt-get -y install libjsoncpp-dev && \
    apt-get -y install python-dev graphviz libgraphviz-dev wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

USER firedrake

RUN /bin/bash -l -c "source /home/firedrake/firedrake/bin/activate && pip3 install git+git://github.com/funsim/moola.git@${MOOLA_BRANCH}"
RUN /bin/bash -l -c "source /home/firedrake/firedrake/bin/activate && pip3 install git+https://bitbucket.org/dolfin-adjoint/pyadjoint.git@${DOLFIN_ADJOINT_BRANCH}"

RUN mkdir shared

RUN /bin/bash -l -c "source /home/firedrake/firedrake/bin/activate && python3 -c \"import firedrake_adjoint\""

RUN echo "source /home/firedrake/firedrake/bin/activate" >> .bashrc
