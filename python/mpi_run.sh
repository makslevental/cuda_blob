PYTHONPATH=../ mpiexec --mca opal_cuda_support 1 --mca btl_smcuda_use_cuda_ipc 0 -n 2 python main.py
