#!/bin/bash

if [ $# -gt 0 ]
then
    if [ $1 == "clean" ]
    then
        echo 'Cleaning project'
        make clean
        rm *.o
	rm ../bin/*.mexglx
    fi
fi

# Set the include path for mex
if [ -d /usr/local/extern/include/ ]
then
    MEXPATH='/usr/local/extern/include/'
elif [ -d /usr/local/matlab/extern/include/ ]
then
    MEXPATH='/usr/local/matlab/extern/include/'
else
    echo 'Error : please set the MEXPATH variable manually'
    #exit -1
fi
MEXPATH='/home/alboot/usr/bin/extern/include/'

make
#GCC=/usr/bin/c++
GCC=g++
#MEX_ARG="-cxx `pkg-config --libs opencv`"
MEX_ARG="-cxx -O"
MEX_EXE=`which mex`
if [ a$MEX_EXE = 'a' ]
then
	# if we could not find the MEX path, we add it manually
	MEX_EXE=/usr/local/matlab/bin/mex
fi
#CFLAGS="-w -c -O3 `pkg-config --cflags opencv`" #$(OPENMP)
CFLAGS="-w -c -O3" #$(OPENMP)

$GCC -fPIC $CFLAGS -I$MEXPATH mexBoxIntegral.c
$GCC -fPIC $CFLAGS -I$MEXPATH mexIntegralImage.c
$GCC -fPIC $CFLAGS -I$MEXPATH mexRectangleFeature.c
$GCC -fPIC $CFLAGS -I$MEXPATH mexEnumerateLearners.c
$GCC -fPIC $CFLAGS -I$MEXPATH mexStoreResponse.c
$GCC -fPIC $CFLAGS -I$MEXPATH mexLoadResponse.c
$GCC -fPIC $CFLAGS -I$MEXPATH mexIntensityFeature.c
$GCC -fPIC $CFLAGS -I$MEXPATH mexNormCorr.c
$GCC -fPIC $CFLAGS -I$MEXPATH mexDotProduct.c
$MEX_EXE mexBoxIntegral.o CMakeFiles/libProjectX.dir/integral.o CMakeFiles/libProjectX.dir/loadImage.o -lgcc -outdir ../bin/ $MEX_ARG
$MEX_EXE mexIntegralImage.o CMakeFiles/libProjectX.dir/integral.o CMakeFiles/libProjectX.dir/loadImage.o -lgcc -outdir ../bin/  $MEX_ARG
$MEX_EXE mexRectangleFeature.o CMakeFiles/libProjectX.dir/integral.o CMakeFiles/libProjectX.dir/loadImage.o -lgcc -outdir ../bin/  $MEX_ARG
$MEX_EXE mexEnumerateLearners.o CMakeFiles/libProjectX.dir/enumerateLearners.o CMakeFiles/libProjectX.dir/integral.o CMakeFiles/libProjectX.dir/loadImage.o -lgcc -outdir ../bin/  $MEX_ARG
$MEX_EXE mexStoreResponse.o CMakeFiles/libProjectX.dir/common.o CMakeFiles/libProjectX.dir/memClient.o -lgcc -outdir ../bin $MEX_ARG
$MEX_EXE mexLoadResponse.o CMakeFiles/libProjectX.dir/common.o CMakeFiles/libProjectX.dir/memClient.o -lgcc -outdir ../bin $MEX_ARG
$MEX_EXE mexIntensityFeature.o CMakeFiles/libProjectX.dir/intensityFeature.o CMakeFiles/libProjectX.dir/utils.o CMakeFiles/libProjectX.dir/Cloud.o -lgcc -outdir ../bin $MEX_ARG
$MEX_EXE mexNormCorr.o -lgcc -outdir ../bin $MEX_ARG
$MEX_EXE mexDotProduct.o -lgcc -outdir ../bin $MEX_ARG
