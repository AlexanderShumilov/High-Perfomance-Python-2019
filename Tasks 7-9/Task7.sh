#!/bin/bash

for i in {1..16}  
do
mpirun -n $i python topopt.py >> time.txt
done
