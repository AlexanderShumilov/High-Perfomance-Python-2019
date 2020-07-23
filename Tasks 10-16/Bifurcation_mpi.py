import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import random
import math

import warnings
warnings.filterwarnings("ignore")

def bif_map(r, x):
    return r * x * (1 - x)

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    r_num = int(sys.argv[1])
    n_iter = 400
    n_store_iter = 1

    x = np.random.random()
    buf_size = math.ceil(r_num / size)
    r_num = buf_size * size

    if rank == 0:
        r_space = np.linspace(0, 4.5, r_num)
        r_space_buf = r_space[:buf_size]

        for i in range(1, size):
            comm.Send(r_space[i * buf_size : (i + 1) * buf_size], dest=i)

    else:
        r_space_buf = np.empty(buf_size, dtype='d')
        comm.Recv(r_space_buf, source=0)

    for i in range(n_iter):
        x = bif_map(r_space_buf, x)

    x_result = np.empty((buf_size, n_store_iter), dtype='d')
    for i in range(n_store_iter):
        x = bif_map(r_space_buf, x)
        x_result[:, i] = x

    x_res_buf = None
    if rank == 0:
        x_res_buf = np.empty((r_num, n_store_iter), dtype='d')

    comm.Gather(x_result, x_res_buf, root=0)

