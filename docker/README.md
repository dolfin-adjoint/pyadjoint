# Docker for dolfin-adjoint

This repository contains the scripts for building various Docker
images for dolfin-adjoint (<http://dolfin-adjoint.org>).

The dolfin-adjoint containers build off of the official Docker containers
maintained by the FEniCS Project (<https://bitbucket.org/fenics-project/docker>).

## Introduction

To install Docker for your platform (Windows, Mac OS X, Linux, cloud platforms,
etc.), follow the instructions at
<https://docs.docker.com/engine/installation/>.

Once you have Docker installed, you can run any of the images below using the
following command:

    docker run -ti quay.io/dolfinadjoint/dolfin-adjoint:latest

To start with you probably want to try the `dolfin-adjoint` image which
includes a full stable version of FEniCS and dolfin-adjoint with PETSc, SLEPc,
petsc4py and slepc4py already compiled. The optimisation routines from scipy,
moola and TAO are included. For licensing reasons, we cannot include
IPOPT which depends on the non-Open Source HSL routines.

If you want to share your current working directory into the container use
the following command:

    docker run -ti -v $(pwd):/home/fenics/shared quay.io/dolfinadjoint/<image-name>:latest


## Documentation

More extensive documentation, including suggested workflows is currently under
construction at <https://fenics-containers.readthedocs.org/>.

## Images

We currently offer following end-user images.

| Image name       | Build status                                                                                                                                                                            | Description                                   |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------|
| dolfin-adjoint   | [![Docker Repository on Quay](https://quay.io/repository/dolfinadjoint/dolfin-adjoint/status "Docker Repository on Quay")](https://quay.io/repository/dolfinadjoint/dolfin-adjoint)      | As `quay.io/fenicsproject/stable:current`, but with dolfin-adjoint.         |
| dev-dolfin-adjoint   | [![Docker Repository on Quay](https://quay.io/repository/dolfinadjoint/dev-dolfin-adjoint/status "Docker Repository on Quay")](https://quay.io/repository/dolfinadjoint/dev-dolfin-adjoint)      | As `quay.io/fenicsproject/dev`, but with master dolfin-adjoint.         |

> Note: The *Build status* column refers to the latest *attempted* build. Even
> if a build is marked as failed, there will still be a working image available
> on the `latest` tag that you can use.

## Building images

Images are hosted on quay.io, and are automatically built in the cloud on from
the Dockerfiles in this repository. The dolfin-adjoint quay.io page is at
<https://quay.io/organization/dolfinadjoint/>.

## Authors

* Jack S. Hale (<jack.hale@uni.lu>)
* Simon Funke (<simon@simula.no>)
