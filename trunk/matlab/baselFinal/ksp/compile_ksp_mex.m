system('make');

mex -v  -cxx -I/usr/local/include/  ksp_graph.o ksp_computer.o ksp_matlab.o -o ksp_matlab 