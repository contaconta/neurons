Creating C++ MEX-files
make
g++ -fPIC -c -I/usr/local/extern/include/ mexInterface.c
mex mexInterface.o CMakeFiles/IntegralImages.dir/integral.o CMakeFiles/IntegralImages.dir/loadImage.o CMakeFiles/IntegralImages.dir/utils.o -lgcc `pkg-config --libs opencv`

