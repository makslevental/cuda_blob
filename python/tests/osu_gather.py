# http://mvapich.cse.ohio-state.edu/benchmarks/

import cupy as cp
from mpi4py import MPI
from numpy import zeros


def numpy_allocate(n, dtype="f"):
    return zeros(n, dtype=dtype)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()
print(f"rank {rank}")

cp.cuda.Device(rank).use()


def osu_bcast(
    BENCHMARH="MPI Gather Latency Test",
    skip=1000,
    loop=10000,
    skip_large=10,
    loop_large=100,
    large_message_size=8192,
    MAX_MSG_SIZE=1 << 28,
    allocator=numpy_allocate,
):
    if world_size < 2:
        if rank == 0:
            errmsg = "This test requires at least two processes"
        else:
            errmsg = None
        raise SystemExit(errmsg)

    if rank == 0:
        r_buf = allocator(MAX_MSG_SIZE * world_size)
    else:
        s_buf = allocator(MAX_MSG_SIZE)

    if rank == 0:
        print("# %s" % (BENCHMARH,))
    if rank == 0:
        print("# %-8s%20s" % (f"Size [{r_buf.dtype}]", "Latency [us]"))

    for size in message_sizes(MAX_MSG_SIZE):
        if size > large_message_size:
            skip = skip_large
            loop = loop_large
        iterations = list(range(loop + skip))
        if rank == 0:
            s_msg = MPI.IN_PLACE
            r_msg = [r_buf, size, MPI.FLOAT]
        else:
            s_msg = [s_buf, size, MPI.FLOAT]
            r_msg = None
        #
        comm.Barrier()
        for i in iterations:
            if i == skip:
                t_start = MPI.Wtime()

            comm.Gather(s_msg, r_msg, 0)
        t_end = MPI.Wtime()
        comm.Barrier()
        #
        if rank == 0:
            latency = (t_end - t_start) * 1e6 / loop
            print("%-10d%20.2f" % (size, latency))


def message_sizes(max_size):
    return [0] + [(1 << i) for i in range(30) if (1 << i) <= max_size]


def cupy_allocate(n, dtype="f"):
    img_h, img_w = 1024, 1024
    n_sigma_bins = n // (img_h * img_w)
    arr = cp.empty((n_sigma_bins, img_h, img_w), dtype=dtype)
    print(f"max len {arr.size}")
    return arr




if __name__ == "__main__":
    # osu_bcast()
    osu_bcast(allocator=cupy_allocate)

