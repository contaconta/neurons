#!/bin/bash


for i in `seq 1 9 144`; do
    let end=$i+9
    matlab  -nojvm -r "postProcessBaselMatResults($i,$end)" &
done