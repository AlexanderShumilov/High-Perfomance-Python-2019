#!/bin/bash

for i in {1..16}  
do
mpirun -n $i python Task9.py >> time_9.txt
done
