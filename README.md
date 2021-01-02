OpenMPI needs to compiled from source with `./configure --with-cuda`

"For Linux 64, Open MPI is built with CUDA awareness but this support is disabled by default.
To enable it, please set the environmental variable OMPI_MCA_opal_cuda_support=true before
launching your MPI processes. Equivalently, you can set the MCA parameter in the command line:
mpiexec --mca opal_cuda_support 1 ..."

We use [`mpi4py==3.1.0a0`](https://docs.cupy.dev/en/stable/reference/interoperability.html#mpi4py) features so you need to `pip install https://github.com/mpi4py/mpi4py/archive/master.zip` if 3.1.0 isn't released yet.

# How to debug OpenMPI remotely (i.e. if you're remote developing to begin with)

[https://stackoverflow.com/a/57938838](https://stackoverflow.com/a/57938838)

The short of it is 

1. create a `Python Debug Server` run configuration
2. install the `pydevd_pycharm` pip package (matching your PyCharm version)
3. check allow parallel runs
4. run as many debug runs as mpi processes on the remote machine
5. reverse ssh tunnel the ports that the debug server on your local machine starts on e.g. `ssh max@localhost -p2222 -R 65300:localhost:65300 -R 65303:localhost:65303`
6. put the `pydevd_pycharm.settrace("localhost", port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)` where you want the breakpoint
7. run the script using `mpiexec`
8. map the source in the run configuration (need to map the specific files)
9. you can only set one port in the run configuration and that prevents multiple parallel runs. better alternative (so you don't have to keep changing the ports in `set_trace` is to change the reverse tunnel instead)

if you get `mpi4py.MPI.Exception: MPI_ERR_TRUNCATE: message truncated` you forgot to add `dtype` to cupy.
