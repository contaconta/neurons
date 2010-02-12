
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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "Cube.h"

using namespace std;


/** We will follow the convenction of the SteerableFilters3D.
    See SteerableFilter3D.cpp
    o2ToIdx[2][0][0] = 0;
    o2ToIdx[1][1][0] = 1;
    o2ToIdx[1][0][1] = 2;
    o2ToIdx[0][2][0] = 3;
    o2ToIdx[0][1][1] = 4;
    o2ToIdx[0][0][2] = 5;
 */
void compute_aguet
(vector< Cube< float, double>*> derivatives,
 Cube< float, double >* res,
 Cube< float, double>* aguet_theta = NULL,
 Cube< float, double>* aguet_phi   = NULL, int inverted = 1)
{
  int margin = 0;
  int nthreads = 1;
#ifdef WITH_OPENMP
  nthreads = omp_get_max_threads();
  printf("cubeCalculateAguet: using %i threads\n", nthreads);
#endif

  printf("calculate_aguet [");
  fflush(stdout);

  //Initialization of the places where each thread will work
  vector< gsl_vector* > eign(nthreads);
  vector< gsl_matrix* > evec(nthreads);
  vector< gsl_eigen_symmv_workspace* > w2(nthreads);
  for(int i = 0; i < nthreads; i++){
    eign[i] = gsl_vector_alloc (3);
    evec[i] = gsl_matrix_alloc (3, 3);
    w2[i]   = gsl_eigen_symmv_alloc (3);
  }

#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
  for(int z = margin; z < derivatives[0]->cubeDepth-margin; z++){
    int tn = 1;
#ifdef WITH_OPENMP
    tn = omp_get_thread_num();
#endif
    //Variables defined in the loop for easy parallel processing
    float l1,l2,l3, theta, phi, r;
    double data[9];
    int higher_eival = 0;

    for(int y = margin; y < derivatives[0]->cubeHeight-margin; y++){
      for(int x = margin; x < derivatives[0]->cubeWidth-margin; x++){

        if(inverted > 0){
          data[0] = -2.0*derivatives[0]->at(x,y,z)/3.0
            + derivatives[3]->at(x,y,z)
            + derivatives[5]->at(x,y,z);
          data[1] = 5.0*derivatives[1]->at(x,y,z)/3.0;
          data[2] = -5.0*derivatives[2]->at(x,y,z)/3.0;
          data[3] = data[1];
          data[4] = derivatives[0]->at(x,y,z)
            - 2.0*derivatives[3]->at(x,y,z)/3.0
            + derivatives[5]->at(x,y,z);
          data[5] = 5.0*derivatives[4]->at(x,y,z)/3.0;
          data[6] = data[2];
          data[7] = data[5];
          data[8] = -2.0*derivatives[5]->at(x,y,z)/3.0
            + derivatives[3]->at(x,y,z)
            + derivatives[0]->at(x,y,z);
        }else {
          data[0] = 2.0*derivatives[0]->at(x,y,z)/3.0
            - derivatives[3]->at(x,y,z)
            - derivatives[5]->at(x,y,z);
          data[1] = -5.0*derivatives[1]->at(x,y,z)/3.0;
          data[2] = 5.0*derivatives[2]->at(x,y,z)/3.0;
          data[3] = data[1];
          data[4] = -derivatives[0]->at(x,y,z)
            + 2.0*derivatives[3]->at(x,y,z)/3.0
            - derivatives[5]->at(x,y,z);
          data[5] = -5.0*derivatives[4]->at(x,y,z)/3.0;
          data[6] = data[2];
          data[7] = data[5];
          data[8] = +2.0*derivatives[5]->at(x,y,z)/3.0
            - derivatives[3]->at(x,y,z)
            - derivatives[0]->at(x,y,z);
        }


        gsl_matrix_view M
          = gsl_matrix_view_array (data, 3, 3);

        gsl_eigen_symmv (&M.matrix, eign[tn], evec[tn], w2[tn]);

        l1 = gsl_vector_get (eign[tn], 0);
        l2 = gsl_vector_get (eign[tn], 1);
        l3 = gsl_vector_get (eign[tn], 2);

        if( (l1>=l2)&&(l1>=l3)){
          higher_eival = 0;
          res->put(x,y,z,l1);
        }
        if( (l2>=l1)&&(l2>=l3)){
          higher_eival = 1;
          res->put(x,y,z,l2);
        }
        if( (l3>=l2)&&(l3>=l1)){
          higher_eival = 2;
          res->put(x,y,z,l3);
        }
        bool forceAngle = false;
        if(fabs(data[0]) + fabs(data[1]) + fabs(data[2]) +
           fabs(data[3]) + fabs(data[4]) + fabs(data[5]) +
           fabs(data[6]) + fabs(data[7]) + fabs(data[8]) <
           1e-4)
          forceAngle = true;
        if(aguet_theta!= NULL){
          theta = atan(gsl_matrix_get(evec[tn],1,higher_eival)/
                       gsl_matrix_get(evec[tn],0,higher_eival));
          if(!forceAngle)
            aguet_theta->put(x,y,z,theta);
        }
        if(aguet_phi!=NULL){
          r = sqrt(gsl_matrix_get(evec[tn],0,higher_eival) *
                   gsl_matrix_get(evec[tn],0,higher_eival)+
                   gsl_matrix_get(evec[tn],1,higher_eival)*
                   gsl_matrix_get(evec[tn],1,higher_eival)+
                   gsl_matrix_get(evec[tn],2,higher_eival)*
                   gsl_matrix_get(evec[tn],2,higher_eival)
                   );
          phi   = acos(gsl_matrix_get(evec[tn],2,higher_eival)/r);
          if(!forceAngle)
            aguet_phi->put(x,y,z,phi);
        }
      }
    }
    printf("#");fflush(stdout);
  }
  printf("]\n");
}

vector< Cube< float, double>* >
compute_second_derivatives
(Cube< uchar, ulong >* cube,
 float sigma_x, float sigma_y, float sigma_z,
 Cube< float, double >* tmp = NULL)
{
  bool delete_tmp = false;
  vector< Cube< float, double >* > derivatives(6);
  for(int i = 0; i < 6; i++)
    derivatives[i] =
      new Cube<float, double>(cube->cubeWidth, cube->cubeHeight, cube->cubeDepth);

  if(tmp == NULL){
    tmp = new Cube<float, double>(cube->cubeWidth, cube->cubeHeight, cube->cubeDepth);
    delete_tmp = true;
  }

  cube->calculate_derivative(2,0,0,sigma_x, sigma_y, sigma_z, derivatives[0], tmp);
  cube->calculate_derivative(1,1,0,sigma_x, sigma_y, sigma_z, derivatives[1], tmp);
  cube->calculate_derivative(1,0,1,sigma_x, sigma_y, sigma_z, derivatives[2], tmp);
  cube->calculate_derivative(0,2,0,sigma_x, sigma_y, sigma_z, derivatives[3], tmp);
  cube->calculate_derivative(0,1,1,sigma_x, sigma_y, sigma_z, derivatives[4], tmp);
  cube->calculate_derivative(0,0,2,sigma_x, sigma_y, sigma_z, derivatives[5], tmp);

  if(delete_tmp) delete tmp;
  return derivatives;
}

template <class T, class U>
Cube<T,U>* get_padded_tile
(Cube<T, U>* orig,
 int x0, int y0, int z0,
 int x1, int y1, int z1,
 int pad_x, int pad_y, int pad_z)
{
  Cube<T,U>* toRet =
    new Cube<T,U>(x1-x0 + 1 + 2*pad_x,
                  y1-y0 + 1 + 2*pad_y,
                  z1-z0 + 1 + 2*pad_z);

  // First simple things, add the innerside of the padding
#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
  for(int z = z0-pad_z; z <= z1+pad_z; z++)
    for(int y = y0-pad_y; y <= y1+pad_y; y++)
      for(int x = x0-pad_x; x <= x1+pad_x; x++)
        toRet->put(x - x0 + pad_x,
                   y - y0 + pad_y,
                   z - z0 + pad_z,
                   orig->at
                   (min(abs(x), 2*orig->cubeWidth  -x -2 ) ,
                    min(abs(y), 2*orig->cubeHeight -y -2 ) ,
                    min(abs(z), 2*orig->cubeDepth  -z -2 ) )
                   );
  return toRet;
}

/** Puts the padded tile to the original cube */
template <class T, class U>
void put_padded_tile
(Cube<T, U>* dest,
 Cube<T, U>* tile,
 int x0, int y0, int z0,
 int pad_x, int pad_y, int pad_z)
{
  for(int z = pad_z; z < tile->cubeDepth - pad_z; z++)
    for(int y = pad_y; y < tile->cubeHeight - pad_y; y++)
      for(int x = pad_x; x < tile->cubeWidth - pad_x; x++)
        dest->put(x0+x-pad_x,
                  y0+y-pad_y,
                  z0+z-pad_z,
                  tile->at(x,y,z));
}

int main(int argc, char **argv) {
  int inverted;

  if(!(argc==4 || argc==5)){
    printf("Usage: cubeCalculateAguet cube sigma_xy sigma_z inverted=1\n");
    exit(0);
  }
  if(argc==4)
    inverted = 1;
  else
    inverted = atoi(argv[4]);

  printf("The last argument should be < 0 if the image is white over gray\n");

  /** Old naive way **/
  // Cube<uchar, ulong>* cube = new Cube<uchar, ulong>(argv[1]);
  // cube->calculate_second_derivates(atof(argv[2]), atof(argv[3]));
  // cube->calculate_aguet(atof(argv[2]), atof(argv[3]), inverted);

  string origName(argv[1]);
  float sigma_xy = atof(argv[2]);
  float sigma_z  = atof(argv[3]);
  Cube<uchar, ulong>* cube = new Cube<uchar, ulong>(origName);
  string directory = getDirectoryFromPath(origName);
  char aguetName[1024];
  sprintf(aguetName, "aguet_%02.2f_%02.2f", sigma_xy, sigma_z);
  char aguetNameTheta[1024];
  sprintf(aguetNameTheta, "aguet_%02.2f_%02.2f_theta", sigma_xy, sigma_z);
  char aguetNamePhi[1024];
  sprintf(aguetNamePhi, "aguet_%02.2f_%02.2f_phi", sigma_xy, sigma_z);

  Cube<float, double>* res   = cube->create_blank_cube(aguetName);
  Cube<float, double>* theta = cube->create_blank_cube(aguetNameTheta);
  Cube<float, double>* phi   = cube->create_blank_cube(aguetNamePhi);

  // Padded execution of the command
  int pad_xy = ceil(Mask::gaussian_mask(2, sigma_xy,true).size()/2);
  int pad_z  = ceil(Mask::gaussian_mask(2, sigma_z, true).size()/2);
  int maxTileSize = 512;
  int tile_size_xy = maxTileSize - 2*pad_xy;
  int tile_size_z  = maxTileSize - 2*pad_z;
  int n_tiles_horiz = ceil(float(cube->cubeWidth) /tile_size_xy);
  int n_tiles_vert  = ceil(float(cube->cubeHeight)/tile_size_xy);
  int n_tiles_depth = ceil(float(cube->cubeDepth) /tile_size_z);

  printf(" limits [%i,%i,%i]\n",
         n_tiles_vert, n_tiles_horiz, n_tiles_depth);


  for(int tz = 0; tz < n_tiles_depth; tz++){
    for(int ty = 0; ty < n_tiles_vert; ty++){
      for(int tx = 0; tx < n_tiles_horiz; tx++){
        int x0, y0, z0, x1, y1, z1;
        x0 = tx*tile_size_xy;
        y0 = ty*tile_size_xy;
        z0 = tz*tile_size_z;
        x1 = min((tx+1)*tile_size_xy -1, cube->cubeWidth -1);
        y1 = min((ty+1)*tile_size_xy -1, cube->cubeHeight-1);
        z1 = min((tz+1)*tile_size_z  -1, cube->cubeDepth -1);
        printf(" iteration [%i,%i,%i], pad= [%i,%i,%i]->[%i,%i,%i]\n",
               tx, ty, tz, x0, y0, z0, x1, y1, z1);

        Cube<uchar, ulong>* padded =
          get_padded_tile(cube, x0, y0, z0, x1, y1, z1, pad_xy, pad_xy, pad_z);

        vector< Cube<float, double>*> derivatives =
          compute_second_derivatives(padded, sigma_xy, sigma_xy, sigma_z);

        Cube<float, double>* res_pad =
          new Cube<float, double>(derivatives[0]->cubeWidth,
                                  derivatives[0]->cubeHeight,
                                  derivatives[0]->cubeDepth);

        Cube<float, double>* theta_pad =
          new Cube<float, double>(derivatives[0]->cubeWidth,
                                  derivatives[0]->cubeHeight,
                                  derivatives[0]->cubeDepth);
        Cube<float, double>* phi_pad =
          new Cube<float, double>(derivatives[0]->cubeWidth,
                                  derivatives[0]->cubeHeight,
                                  derivatives[0]->cubeDepth);

        compute_aguet(derivatives, res_pad, theta_pad, phi_pad, inverted);
        // compute_aguet(derivatives, res_pad, NULL, NULL, inverted);
        put_padded_tile(res, res_pad, x0,y0,z0, pad_xy, pad_xy, pad_z);
        put_padded_tile(theta, theta_pad, x0,y0,z0, pad_xy, pad_xy, pad_z);
        put_padded_tile(phi, phi_pad, x0,y0,z0, pad_xy, pad_xy, pad_z);
        for(int i = 0; i < derivatives.size(); i++)
          delete derivatives[i];
        delete res_pad;
        delete padded;
      }
    }
  }


  // Cube<uchar, ulong>* padded =
    // get_padded_tile(cube, 10, 10, 10,
                    // 50, 100, 20,
                    // 20, 20, 20);

  // Cube<uchar, ulong>* result =
    // cube->create_blank_cube_uchar("replica");
  // put_padded_tile(result, padded, 10, 10, 10,
                  // 20, 20, 20);

  // Cube<float, double>* res   = cube->create_blank_cube(aguetName);
  // Cube<float, double>* theta = cube->create_blank_cube(aguetNameTheta);
  // Cube<float, double>* phi   = cube->create_blank_cube(aguetNamePhi);

  // vector< Cube<float, double>*> derivatives =
    // compute_second_derivatives(cube, sigma_xy, sigma_xy, sigma_z);

  // compute_aguet(derivatives, res, theta, phi, inverted);

}
