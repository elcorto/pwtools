#!/bin/bash

rm -fv *.tgz
for d in md-{nvt,npt} vc-relax; do
    tar -vczf $d.tgz $d/
done
