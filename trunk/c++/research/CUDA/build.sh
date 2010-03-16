#!/bin/bash

dir=`pwd`
cd common
make
cd $dir
for i in `ls src`; do
cd src/$i
make
cd $dir
done