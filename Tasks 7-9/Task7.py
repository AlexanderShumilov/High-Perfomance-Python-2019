#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from math import pi
from mpi4py import MPI

def f(x):
    return 1/(1 + x**2)

def trapz(f, a, b, n_points = 1):
    
    dx = (b - a)/n_points
    I = 1/2 * (f(a) + f(b))
    
    for i in range(1, int(n_points)):
        I = I + f(a + i*dx)     
    return I*dx

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

a = -pi/4
b = pi/4
steps = 1e+5
my_int = np.zeros(1)
integral_sum = np.zeros(1)

if rank == 0:
    start_time = MPI.Wtime()

dx = (b - a) / steps  

if steps >= size:
    if rank != size - 1:
        tosend = int(np.floor(steps/size))
    else:
        tosend = int(steps - np.floor(steps*(size - 1)/size))
else:
    if rank < steps:
        tosend = 1
    else:
        tosend = 0

dx = (b - a)/steps
a_i = a + rank * dx * tosend
b_i = a + (rank + 1) * dx * tosend

my_int[0] = trapz(f, a_i, b_i, tosend)

comm.Barrier()

if rank == 0:
    end_time = MPI.Wtime()

comm.Reduce(my_int, integral_sum, MPI.SUM, root = 0)

if rank == 0:
    print(end_time - start_time)






