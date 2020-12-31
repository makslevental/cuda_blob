OpenMPI needs to compiled from source with `./configure --with-cuda`

"For Linux 64, Open MPI is built with CUDA awareness but this support is disabled by default.
To enable it, please set the environmental variable OMPI_MCA_opal_cuda_support=true before
launching your MPI processes. Equivalently, you can set the MCA parameter in the command line:
mpiexec --mca opal_cuda_support 1 ..."

We use [`mpi4py==3.1.0a0`](https://docs.cupy.dev/en/stable/reference/interoperability.html#mpi4py) features so you need to `pip install https://github.com/mpi4py/mpi4py/archive/master.zip` if 3.1.0 isn't released yet.
