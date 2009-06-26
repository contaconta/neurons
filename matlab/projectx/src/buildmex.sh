make
GCC=/usr/bin/c++
MEXPATH=/usr/local/matlab/extern/include/
MEX_ARG=-cxx

$GCC -fPIC -c -I$MEXPATH mexBoxIntegral.c
$GCC -fPIC -c -I$MEXPATH mexIntegralImage.c
$GCC -fPIC -c -I$MEXPATH mexRectangleFeature.c
$GCC -fPIC -c -I$MEXPATH mexEnumerateLearners.c
mex mexBoxIntegral.o CMakeFiles/IntegralImages.dir/integral.o CMakeFiles/IntegralImages.dir/loadImage.o CMakeFiles/IntegralImages.dir/utils.o -lgcc `pkg-config --libs opencv` $MEX_ARG -outdir ../bin/
mex mexIntegralImage.o CMakeFiles/IntegralImages.dir/integral.o CMakeFiles/IntegralImages.dir/loadImage.o CMakeFiles/IntegralImages.dir/utils.o -lgcc `pkg-config --libs opencv` $MEX_ARG -outdir ../bin/
mex mexRectangleFeature.o CMakeFiles/IntegralImages.dir/integral.o CMakeFiles/IntegralImages.dir/loadImage.o CMakeFiles/IntegralImages.dir/utils.o -lgcc `pkg-config --libs opencv` $MEX_ARG -outdir ../bin/
mex mexEnumerateLearners.o CMakeFiles/IntegralImages.dir/enumerateLearners.o CMakeFiles/IntegralImages.dir/integral.o CMakeFiles/IntegralImages.dir/loadImage.o CMakeFiles/IntegralImages.dir/utils.o -lgcc `pkg-config --libs opencv` $MEX_ARG -outdir ../bin/

