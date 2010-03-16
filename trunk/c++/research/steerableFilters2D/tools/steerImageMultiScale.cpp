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
#include "SteerableFilter2DMultiScale.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#ifdef _OPENMP
#include <omp.h>
#endif


using namespace std;


int main(int argc, char **argv) {


  // if( (argc != 6) || (argc != 8) ){
    // printf("Usage: steerImageMultiScale image coefficients sigma_start sigma_end sigma_step <output_file = result.jpg> <output_orientation = resultTheta.jpg> \n");
    // exit(0);
  // }

  SteerableFilter2DMultiScale* stf;


  if(argc == 6)
    stf =  new SteerableFilter2DMultiScale(argv[1], argv[2], atof(argv[3]),
                                       atof(argv[4]),atof(argv[5]));
  else
    stf =  new SteerableFilter2DMultiScale(argv[1], argv[2], atof(argv[3]),
                                       atof(argv[4]),atof(argv[5]),
                                       argv[6], argv[7]);


  vector< Image<float>* > responses;
  string filename = argv[1];

  string directory = filename.substr(0,filename.find_last_of("/\\")+1);
  stf->result->put_all(0.0);
  stf->orientation->put_all(0.0);


  //Invert the values of the image to see if that helps
  for(int nS = 0; nS < stf->stf.size(); nS++)
    for(int i = 0; i < stf->stf[nS]->alpha->size; i++)
      gsl_vector_set(stf->stf[nS]->alpha, i, -gsl_vector_get(stf->stf[nS]->alpha,i));

  char buff[512];

  for(int angle = 0; angle < 180; angle+=10){
    printf("Angle = %i\n", angle);
    int idx= angle/10;
    sprintf(buff, "%s/resp_%i.jpg", directory.c_str(), angle);
    Image<float>* new_img = stf->result->create_blank_image_float(buff);
    responses.push_back(new_img);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int x = 0; x < stf->result->width; x++){
      for(int y  = 0; y < stf->result->height; y++){

        responses[idx]->put(x,y, stf->response(float(angle*M_PI)/180, x, y));
        if(angle == 0){
          stf->result->put(x,y,responses[idx]->at(x,y));
        } else if( responses[idx]->at(x,y) > stf->result->at(x,y) ){
          stf->result->put(x,y,responses[idx]->at(x,y));
          stf->orientation->put(x,y,angle);
        }
      }
    }
    responses[idx]->save();
  }
  stf->result->save();
  stf->orientation->save();
  for(int i = 0; i < responses.size(); i++)
    responses[i]->save();
}
