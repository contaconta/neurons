
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
// Contact <ggonzale@atenea> for comments & bug reports                //
/////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "cv.h"
#include "highgui.h"
#include "Image.h"
#include "polynomial.h"
#include "SteerableFilter2D.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_min.h>
#include <gsl/gsl_vector.h>
#include "Mask.h"


using namespace std;

int main(int argc, char **argv) {

  if(argc!=4){
    printf("Usage: normalizeAlphaCoords input_alphas output_alphas sigma\n");
    exit(0);
  }
  
  double sigma = atof(argv[3]);

  FILE* f = fopen(argv[1], "r");
  //This is a hack to automatically know the dimension of the vector
  char number[1024];
  int nNumbers = 0;
  while(fgets(number, 1024, f) != NULL){
    nNumbers++;
  }
  fclose(f);

  int M; // order of the filter
  for(M = 0; M < 10; M++){
    if(nNumbers == (M+3)*M/2)
      break;
  }

  f = fopen(argv[1], "r");
  gsl_vector* alpha = gsl_vector_alloc(nNumbers);
  int err = gsl_vector_fscanf(f, alpha);
  if(err == GSL_EFAILED){
    printf("Error reading the vectorx in %s\n", argv[1]);
    exit(0);
  }

  gsl_vector* out =  gsl_vector_alloc(nNumbers);
  int n_idx = 0;
  for(int k = 1; k <= M; k++){
    for(int i = 0; i <= k; i++){
      gsl_vector_set(out, n_idx + i,
                     gsl_vector_get(alpha,n_idx+i)/
                     Mask::energy2DGaussianMask(k-i,i,sigma)
                     );
    }
    n_idx = n_idx + k + 1;
  }

  fclose(f);

  FILE* of = fopen(argv[2],"w");
  err = gsl_vector_fprintf(of, out, "%f");
  if(err == GSL_EFAILED){
    printf("Error writing the vectorx in %s\n", argv[2]);
    exit(0);
  }

  fclose(of);


}
