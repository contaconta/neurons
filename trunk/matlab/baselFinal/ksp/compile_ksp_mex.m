system('make');

mex -v  -cxx -I/usr/local/include/  ksp_graph.o ksp_computer.o ksp_EMD_weights.o -o ksp_EMD_weights

mex -v  -cxx -I/usr/local/include/  ksp_graph.o ksp_computer.o ksp_euclidean_dist_weights.o -o ksp_euclidean_dist_weights