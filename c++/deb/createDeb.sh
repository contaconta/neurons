#!/bin/bash

## Removes any previous package
rm -rf package/*

## Creates a fresh compilation of the library and the viewer
path=`pwd`
cd ../cmake/
# cmake .
# make -j 3 vivaView
cd $path

## Copies the library and the viewer to the right place
mkdir -p package/usr/lib
mkdir -p package/usr/bin
mkdir -p package/DEBIAN

cp ../lib/libviva.so package/usr/lib/
cp ../bin/vivaView package/usr/bin/

## creates the md5
cd package
for i in `find usr -type f`; do
    md5sum $i >> DEBIAN/md5sums
done
cd ..

cp control package/DEBIAN
cp debian-binary package

dpkg -b package/ viva_0.01_i386.deb