
/////////////////////////////////////////////////////////////////////////
// This program is free software; you can redistribute it and/or       //
// modify it under the terms of the GNU General Public License         //
// version 2 as published by the Free Software Foundation.             //
//                                                                     //
// This program is distributed in the hope that it will be useful, but //
// WITHOUT ANY WARRANTY; without even the implied warranty of          //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU   //
// General Public License for more details.                            //
//                                                                     //
// Written and (C) by German Gonzalez                                  //
// Contact < german.gonzalez@epfl.ch > for comments & bug reports      //
/////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "Timer.h"

using namespace std;

int arrsize = 5000;

int main(int argc, char **argv) {

  double arr[arrsize];

  Timer timer;

#ifdef _OPENMP
  printf("The maximum number of threads is %i\n",  omp_get_max_threads());
#endif

  unsigned long timeb = timer.getMicroseconds();
#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
  for(int i = 0; i < arrsize; i++)
    for(int x = 0; x < arrsize; x++)
      arr[i] = sqrt((double)i)*sqrt((double)x);
  unsigned long timee = timer.getMicroseconds();
  printf("The time in computing %ix%i square roots with %i threads is %i\n",
         arrsize, arrsize,  omp_get_max_threads(), int(timee-timeb));

  timeb = timer.getMicroseconds();
  for(int i = 0; i < arrsize; i++)
    for(int x = 0; x < arrsize; x++)
      arr[i] = sqrt((double)i)*sqrt((double)x);
  timee = timer.getMicroseconds();
  printf("The time in computing %ix%i square roots with %i threads is %i\n",
         arrsize, arrsize,  1, int(timee-timeb));


/*
  int nthreads, tid, procs, maxt, inpar, dynamic, nested;

#pragma omp parallel private(nthreads, tid)
  {

  tid = omp_get_thread_num();

  if (tid == 0) 
    {
    printf("Thread %d getting environment info...\n", tid);

    procs = omp_get_num_procs();
    nthreads = omp_get_num_threads();
    maxt = omp_get_max_threads();
    inpar = omp_in_parallel();
    dynamic = omp_get_dynamic();
    nested = omp_get_nested();

    printf("Number of processors = %d\n", procs);
    printf("Number of threads = %d\n", nthreads);
    printf("Max threads = %d\n", maxt);
    printf("In parallel? = %d\n", inpar);
    printf("Dynamic threads enabled? = %d\n", dynamic);
    printf("Nested parallelism supported? = %d\n", nested);

    }

  }
*/

}
