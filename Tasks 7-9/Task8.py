#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import imageio

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

np.random.seed(12)
n_iter = 1000
N = 50
m = 200

def Gosper(pos):
    j = 1
    pos[6+j, 2+j] = pos[6+j, 3+j] = pos[7+j, 2+j] = pos[7+j, 3+j]  = 1
    pos[4+j, 36+j] = pos[4+j, 37+j] = pos[5+j, 36+j] = pos[5+j, 37+j]  = 1
    
    pos[6+j, 12+j] = pos[7+j, 12+j] = pos[8+j, 12+j] = 1
    pos[5+j, 13+j] = pos[4+j, 14+j] = pos[9+j, 13+j] = pos[10+j, 14+j] = 1
    pos[4+j, 15+j] = pos[10+j, 15+j] = 1
    pos[7+j, 16+j] = pos[5+j, 17+j] = pos[9+j, 17+j] = 1
    pos[7+j, 18+j] = pos[6+j, 18+j] = pos[8+j, 18+j] = pos[7+j, 19+j] = 1
    
    pos[6+j, 22+j] = pos[5+j, 22+j] = pos[4+j, 22+j] = 1
    pos[6+j, 23+j] = pos[5+j, 23+j] = pos[4+j, 23+j] = 1
    pos[7+j, 24+j] = pos[3+j, 24+j] = 1
    pos[7+j, 26+j] = pos[3+j, 26+j] = pos[8+j, 26+j] = pos[2+j, 26+j] = 1
    return pos

@profile
def fun(n_iter, A, N, N_start, N_end):
	for j in range(n_iter):

	    A_shifted = shift(A[:, [N_start - 1 % N] + list(range(N_start, N_end + 1))])

	    comm.Barrier()

	    A_shifted = comm.gather(A_shifted, 0)
	    return A_shifted
	


A = np.zeros((N, N))
A = Gosper(A)

def shift(B):
    B = np.roll(np.array(B), 1, axis = 1)
    return B[:, 1:]

if rank == 0:
    frames_path = "./pics_for_shift/{i}.jpg"
    #im = plt.imshow(A, cmap='binary')
    #plt.savefig(frames_path.format(i=0))
    #plt.close()
    gif_path = "./shift.gif"

N_ = int(np.floor(N/size))
if N >= size:
    if rank != size - 1:
        N_start = N_*rank
        N_end = N_*(rank + 1) - 1
    else:
        N_start = N_*rank
        N_end = N_start + int(N - N_*(size - 1)) - 1
else:
    if rank < disc:
        N_start = rank
        N_end = rank
    else:
        N_start = 0
        N_end = 0

A_shifted = fun(n_iter, A, N, N_start, N_end)

if rank == 0:
    # print("Matrix A is:\n", A)
    B = np.hstack([A_shifted[i] for i in range(len(A_shifted))])
        # print("Shifted matrix B is:\n", B)
        #im = plt.imshow(B, cmap='binary')
        #plt.savefig(frames_path.format(i=j + 1))
        #plt.close()
    A = B.copy()
comm.Barrier()
comm.Bcast(A, root=0)		   


# In[ ]:




