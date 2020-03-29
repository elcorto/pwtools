#!/bin/bash

for d in md-{nvt,npt} vc-relax; do
    cd $d
    lammps < lmp.in
    cd ..
done
