#!/bin/bash

for d in md-{nvt,npt} vc-relax; do
    cd $d
    lmp < lmp.in
    cd ..
done
