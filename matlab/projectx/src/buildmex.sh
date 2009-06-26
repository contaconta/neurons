#!/bin/sh

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
GCC=/usr/bin/c++
MEX_ARG=-cxx

$GCC -fPIC -c -I$MEXPATH mexBoxIntegral.c
$GCC -fPIC -c -I$MEXPATH mexIntegralImage.c
$GCC -fPIC -c -I$MEXPATH mexRectangleFeature.c
$GCC -fPIC -c -I$MEXPATH mexEnumerateLearners.c
mex mexBoxIntegral.o CMakeFiles/IntegralImages.dir/integral.o CMakeFiles/IntegralImages.dir/loadImage.o CMakeFiles/IntegralImages.dir/utils.o -lgcc `pkg-config --libs opencv` -outdir ../bin/ $MEX_ARG
mex mexIntegralImage.o CMakeFiles/IntegralImages.dir/integral.o CMakeFiles/IntegralImages.dir/loadImage.o CMakeFiles/IntegralImages.dir/utils.o -lgcc `pkg-config --libs opencv` -outdir ../bin/  $MEX_ARG
mex mexRectangleFeature.o CMakeFiles/IntegralImages.dir/integral.o CMakeFiles/IntegralImages.dir/loadImage.o CMakeFiles/IntegralImages.dir/utils.o -lgcc `pkg-config --libs opencv` -outdir ../bin/  $MEX_ARG
mex mexEnumerateLearners.o CMakeFiles/IntegralImages.dir/enumerateLearners.o CMakeFiles/IntegralImages.dir/integral.o CMakeFiles/IntegralImages.dir/loadImage.o CMakeFiles/IntegralImages.dir/utils.o -lgcc `pkg-config --libs opencv` -outdir ../bin/  $MEX_ARG
