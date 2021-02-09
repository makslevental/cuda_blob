import ctypes

import py3nvml.py3nvml as nvml
from cupy.cuda.device import Device
from cupy.cuda.runtime import (
    eventCreate,
    eventRecord,
    eventSynchronize,
    eventElapsedTime,
    memGetInfo,
)

_cudart = ctypes.CDLL("libcudart.so")


def cuda_profiler_start():
    # As shown at http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PROFILER.html,
    # the return value will unconditionally be 0. This check is just in case it changes in
    # the future.
    ret = _cudart.cudaProfilerStart()
    if ret != 0:
        raise Exception("cudaProfilerStart() returned %d" % ret)


def cuda_profiler_stop():
    ret = _cudart.cudaProfilerStop()
    if ret != 0:
        raise Exception("cudaProfilerStop() returned %d" % ret)


class GPUTimer:
    def __init__(self, msg: str):
        self._msg = msg
        self._start = eventCreate()
        self._stop = eventCreate()

    def start(self):
        eventRecord(self._start, 0)

    def stop(self):
        eventRecord(self._stop, 0)

    def elapsed_time(self):
        eventSynchronize(self._stop)
        e = eventElapsedTime(self._start, self._stop)
        return e

    def __enter__(self):
        self.start()

    def __exit__(self, type, value, traceback):
        self.stop()
        print(f"{self._msg} time {self.elapsed_time():.3f}ms")


def get_used_cuda_mem(device):
    with Device(device):
        free, total = memGetInfo()
    return (total - free) / 1024.0 / 1024.0


def try_get_info(f, h, default="N/A"):
    try:
        v = f(h)
    except:
        v = default
    return v


def get_gpu_utilization(h):
    util = try_get_info(nvml.nvmlDeviceGetUtilizationRates, h)
    gpu_util = util.gpu
    return gpu_util


if __name__ == "__main__":
    # g = GPUTimer()
    # g.start()
    # time.sleep(5)
    # g.stop()
    # print(g.elapsed_time())
    nvml.nvmlInit()
    h = nvml.nvmlDeviceGetHandleByIndex(0)
    get_gpu_utilization(h)
