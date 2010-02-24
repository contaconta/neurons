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
#include <gsl/gsl_rng.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

using namespace std;

typedef struct
{
  int x;
  int y;
  float scale;
  float theta;
  int type;
} Point;


SteerableFilter2D* stf;
double old_theta = 0;
float sigma = 0;

vector< Point > points;

void load_points(string points_name){
  std::ifstream points_in(points_name.c_str());
  char line[2048];
  while(!points_in.getline(line, 2048).fail() )
    {
      Point p;
      sscanf(line, "%i %i %f %f %i", &p.x, &p.y, &p.scale,
             &p.theta, &p.type);
      p.theta = p.theta*3.1416/180;
      points.push_back(p);
    }
  points_in.close();
}

void output_response_all(){
  double r;
//   for(int i = 0; i < 360; i+= 5){
//     printf("%i ", i);
//     for(int j = 0; j < points.size(); j++){
//       printf("%f ",stf->response(i*3.1416/180, points[j].x, points[j].y));
//     }
//     printf("\n");
//   }

  for(int j = 0; j < points.size(); j++)
    printf("%f %i \n", stf->response(points[j].theta, points[j].x, points[j].y),points[j].type);
}

int main(int argc, char **argv) {


  if(argc!=5){
    printf("Usage: test <image> <coefficients> <sigma> <points>\n");
    exit(0);
  }

  sigma = atof(argv[3]);
  stf = new SteerableFilter2D(argv[1], argv[2], sigma);

  string points_f = argv[4];
  load_points(argv[4]);

  output_response_all();

//   //Calculates the vector of the coefficients, the same as training
//   gsl_vector* w_coefficients = gsl_vector_alloc((stf->M+1)*(stf->M+2)/2);
//   gsl_vector_set_all(w_coefficients, 0);
//   int k_idx = 0;
//   for(int k = 0; k <= stf->M; k++){
//     for(int i = 0; i <= k; i++){
//       double w_points = 0;
//       for(int n = 0; n < points.size(); n++){
//         double p_contribution = 0;
//         for(int j = 0; j <= k; j++){
//           double b = 0;
//           for(int l = 0; l <= k-i; l++){
//             for(int m = 0; m <= i; m++){
//               if(!(k-(l+m)==j)) continue;
//               b += stf->combinatorial(k-i,l)*
//                 stf->combinatorial(i,m)*pow(-1.0,m)*
//                 pow(cos(points[n].theta),i+l-m)*
//                 pow(sin(points[n].theta), k+m-i-l);
//             }//m
//           }//l
//           p_contribution += b*
//             stf->derivatives[k_idx+j]->at(points[n].x,points[n].y);
//         }//j
//         p_contribution *= points[n].type;
//         w_points += p_contribution;
//       }//n
//       gsl_vector_set(w_coefficients, k_idx+i, w_points);
//     }//i
//     k_idx += (k+1);
//   }//k

//   double value = 0;
//   gsl_blas_ddot(w_coefficients, stf->alpha, &value);

//   printf("%f\n", value);

}

