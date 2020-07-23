#!/bin/bash

for i in {1..16}  
do
mpirun -n $i python Task8.py >> time_8.txt
done
