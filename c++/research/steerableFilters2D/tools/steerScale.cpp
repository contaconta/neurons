
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
#include "SteerableFilter2DMultiScale.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_min.h>



using namespace std;
SteerableFilter2DMultiScale* stf;
double old_theta = 0;

int main(int argc, char **argv) {

  if(argc!=7){
    printf("Usage: steerScale image coefficients sigma_start sigma_end sigma_step  angle\n");
    exit(0);
  }

  stf = new SteerableFilter2DMultiScale(argv[1], argv[2], atof(argv[3]),
                                        atof(argv[4]),atof(argv[5]));
//   stf->theta = 0;
//   gsl_vector_memcpy(stf->b_theta, stf->alpha);
//   gsl_vector_fprintf(stdout, stf->b_theta, "%03f");
//   filter_max();
  stf->filter(atof(argv[6])*3.14159/180);
//   stf->filter(0);
}
