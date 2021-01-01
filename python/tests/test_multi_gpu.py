import unittest

import cupy as cp
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


# import pydevd_pycharm
# port_mapping = [65300, 65303]
# pydevd_pycharm.settrace(
#     "localhost", port=port_mapping[rank], stdoutToServer=True, stderrToServer=True
# )


class MyTestCase(unittest.TestCase):
    def test_basic_mpi(self):
        # Allreduce
        sendbuf = cp.arange(10, dtype="i")
        recvbuf = cp.empty_like(sendbuf)
        comm.Allreduce(sendbuf, recvbuf)
        self.assertTrue(cp.allclose(recvbuf, sendbuf * size))

        # Bcast
        if rank == 0:
            buf = cp.arange(100, dtype=cp.complex64)
        else:
            buf = cp.empty(100, dtype=cp.complex64)
        comm.Bcast(buf)
        self.assertTrue(cp.allclose(buf, cp.arange(100, dtype=cp.complex64)))

        # Send-Recv
        if rank == 0:
            buf = cp.arange(20, dtype=cp.float64)
            comm.Send(buf, dest=1, tag=88)
        else:
            buf = cp.empty(20, dtype=cp.float64)
            comm.Recv(buf, source=0, tag=88)
            self.assertTrue(cp.allclose(buf, cp.arange(20, dtype=cp.float64)))

        # Scatter
        sendbuf = None
        if rank == 0:
            sendbuf = cp.empty((size, 100), dtype="i")
            sendbuf.T[:, :] = cp.arange(size)
        recvbuf = cp.empty(100, dtype="i")
        # root: Rank of sending process (integer).
        comm.Scatter(sendbuf, recvbuf, root=0)
        # element wise comparison ("all close to each other")
        self.assertTrue(cp.allclose(recvbuf, rank))


if __name__ == "__main__":
    unittest.main()
