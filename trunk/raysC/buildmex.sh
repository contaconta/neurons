#!/bin/bash

if [ $# -gt 0 ]
then
    if [ $1 == "clean" ]
    then
        echo 'Cleaning project'
        #make clean
        rm CMakeFiles/librays.dir/*.o
	rm ../bin/*.mexglx
    fi
fi

# Set the include path for mex
if [ -d /usr/bin/MATLAB79/extern/include/ ]
then
    MEXPATH='/usr/bin/MATLAB79/extern/include/'
elif [ -d /usr/local/MATLAB79/extern/include/ ]
then
    MEXPATH='/usr/local/MATLAB79/extern/include/'
else
    echo 'Error : please set the MEXPATH variable manually'
    #exit -1
fi
#MEXPATH='/usr/bin/MATLAB79/extern/include/'
#MEXPATH='/usr/local/MATLAB79/extern/include/'
export MEXPATH

make
#GCC=/usr/bin/c++
GCC=g++-4.2
MEX_ARG="-cxx `pkg-config --libs opencv` -O"
#MEX_ARG="-cxx -O"
MEX_EXE=`which mex`
if [ a$MEX_EXE = 'a' ]
then
	# if we could not find the MEX path, we add it manually
	MEX_EXE=/usr/local/matlab/bin/mex
fi
CFLAGS="-w -c -O3 `pkg-config --cflags opencv`" #$(OPENMP)
#CFLAGS="-w -c -O3" #$(OPENMP)

#$GCC -fPIC $CFLAGS -I$MEXPATH testMex.c
#$MEX_EXE testMex.o -lgcc -outdir ../bin $MEX_ARG
$GCC -fPIC $CFLAGS -I$MEXPATH mexRays.c
$MEX_EXE CMakeFiles/librays.dir/rays.o mexRays.o -lgcc -outdir ./bin $MEX_ARG
