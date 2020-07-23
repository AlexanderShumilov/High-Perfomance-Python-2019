#!/bin/bash

for i in {1..16}  
do
mpirun -n $i python -m memory_profiler Task8.py >> mem_8_.txt
done
