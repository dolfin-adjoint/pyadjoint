# Installation with docker


1. Install docker: https://docs.docker.com/engine/installation/
2. Build docker container with
       `docker build -t dev-dolfin-adjoint .`
3. Start docker container with
       `docker run -it -v $(pwd):/root/shared dev-dolfin-adjoint`
