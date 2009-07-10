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
    exit -1
fi

make
#GCC=/usr/bin/c++
GCC=g++
MEX_ARG=-cxx
#MEX_EXE=/usr/local/matlab/bin/mex
#MEX_EXE=/usr/local/bin/mex
MEX_EXE=`which mex`
CFLAGS="-w -c -O3" #$(OPENMP)

$GCC -fPIC $CFLAGS -I$MEXPATH mexBoxIntegral.c
$GCC -fPIC $CFLAGS -I$MEXPATH mexIntegralImage.c
$GCC -fPIC $CFLAGS -I$MEXPATH mexRectangleFeature.c
$GCC -fPIC $CFLAGS -I$MEXPATH mexEnumerateLearners.c
$GCC -fPIC $CFLAGS -I$MEXPATH mexStoreResponse.c
$GCC -fPIC $CFLAGS -I$MEXPATH mexLoadResponse.c
$MEX_EXE mexBoxIntegral.o CMakeFiles/IntegralImages.dir/integral.o CMakeFiles/IntegralImages.dir/loadImage.o -lgcc -outdir ../bin/ $MEX_ARG
$MEX_EXE mexIntegralImage.o CMakeFiles/IntegralImages.dir/integral.o CMakeFiles/IntegralImages.dir/loadImage.o -lgcc -outdir ../bin/  $MEX_ARG
$MEX_EXE mexRectangleFeature.o CMakeFiles/IntegralImages.dir/integral.o CMakeFiles/IntegralImages.dir/loadImage.o -lgcc -outdir ../bin/  $MEX_ARG
$MEX_EXE mexEnumerateLearners.o CMakeFiles/IntegralImages.dir/enumerateLearners.o CMakeFiles/IntegralImages.dir/integral.o CMakeFiles/IntegralImages.dir/loadImage.o -lgcc -outdir ../bin/  $MEX_ARG
$MEX_EXE mexStoreResponse.o CMakeFiles/IntegralImages.dir/common.o CMakeFiles/IntegralImages.dir/memClient.o -lgcc -outdir ../bin $MEX_ARG
$MEX_EXE mexLoadResponse.o CMakeFiles/IntegralImages.dir/common.o CMakeFiles/IntegralImages.dir/memClient.o -lgcc -outdir ../bin $MEX_ARG
