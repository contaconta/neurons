
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
#include "Cube.h"
#include "Mask.h"
#include "Timer.h"

class CubeInMemmoryFloat
{
 public:
  int width;
  int height;
  int depth;

  float *  voxels_origin;
  float*** voxels;

  CubeInMemmoryFloat(int _width, int _height, int _depth){
    width  = _width;
    height = _height;
    depth  = _depth;
    voxels_origin = (float*)calloc(width*height*depth,sizeof(float));

    voxels = (float***)malloc(depth*sizeof(float**));
    for(int z = 0; z < depth; z++){
      voxels[z] = (float**)malloc(height*sizeof(float*));
      for(int j = 0; j < height; j++){
        voxels[z][j]=(float*)&voxels_origin[z*width*height+j*width];
      }
    }
  }

  float at(int x, int y, int z){
    return voxels[z][y][x];
  }

  float put(int x, int y, int z, float value){
    voxels[z][y][x] = value;
  }

  void saveToFile(string filename){
    FILE* fp = fopen(filename.c_str(), "w");
    for(int z = 0; z < depth; z++)
      for(int y = 0; y < height; y++)
        fwrite(voxels[z][y], sizeof(float), width, fp);
    fclose(fp);
  }


};

class CubeInMemmoryUchar
{
 public:
  int width;
  int height;
  int depth;

  uchar *  voxels_origin;
  uchar*** voxels;

  CubeInMemmoryUchar(int _width, int _height, int _depth){
    width  = _width;
    height = _height;
    depth  = _depth;
    voxels_origin = (uchar*)calloc(width*height*depth,sizeof(uchar));

    voxels = (uchar***)malloc(depth*sizeof(uchar**));
    for(int z = 0; z < depth; z++){
      voxels[z] = (uchar**)malloc(height*sizeof(uchar*));
      for(int j = 0; j < height; j++){
        voxels[z][j]=(uchar*)&voxels_origin[z*width*height+j*width];
      }
    }
  }

  uchar at(int x, int y, int z){
    return voxels[z][y][x];
  }

  uchar put(int x, int y, int z, uchar value){
    voxels[z][y][x] = value;
  }

  void loadFromFile(string filename){
    FILE* fp = fopen(filename.c_str(), "r");
    for(int z = 0; z < depth; z++)
      for(int y = 0; y < height; y++)
        fread(voxels[z][y], sizeof(uchar), width, fp);
    fclose(fp);
  }


  void convolve_horizontally
  (vector< float >& mask,
   CubeInMemmoryFloat* output)
  {
    int mask_side = mask.size()/2;
    int mask_size = mask.size();
    printf("CubeInMemmoryUchar::convolve_horizontally [");

    int printLimit = max(1,(int)(depth/20));
    #ifdef WITH_OPENMP
    #pragma omp parallel for
    #endif
    for(int z = 0; z < depth; z++){
      int x,q;
      float result;
      for(int y = 0; y < height; y++)
        {
          // Beginning of the line
          for(x = 0; x < mask_size; x++){
            result = 0;
            for(q = -mask_side; q <=mask_side; q++){
              if(x+q<0)
                result+=this->at(0,y,z)*mask[mask_side + q];
              else
                result += this->at(x+q,y,z)*mask[mask_side + q];
            }
            output->put(x,y,z,result);
          }

          //Middle of the line
          for(x = mask_size; x <= width-mask_size-1; x++)
            {
              result = 0;
              for(q = -mask_side; q <=mask_side; q++)
                result += this->at(x+q,y,z)*mask[mask_side + q];
              output->put(x,y,z,result);
              // printf("%i %i %i\n", x, y, z);
            }
          //End of the line
          for(x = width-mask_size; x < width; x++){
            result = 0;
            for(q = -mask_side; q <=mask_side; q++){
              if(x+q >= width)
                result+=this->at(width-1,y,z)*mask[mask_side + q];
              else
                result += this->at(x+q,y,z)*mask[mask_side + q];
            }
            output->put(x,y,z,result);
          }
        }
      if(z%printLimit==0)
        printf("#");fflush(stdout);
    }
    printf("]\n");
  }


};



using namespace std;

int main(int argc, char **argv) {


  Timer timer;
  unsigned long timeS, timeE;

  timeS = timer.getMilliseconds();
  CubeInMemmoryUchar* cbimu = new CubeInMemmoryUchar(1536,3584,148);
  cbimu->loadFromFile("/media/neurons/n1/3d/1/N1.vl");
//   cbimu->loadFromFile("/home/ggonzale/mount/cvlabfiler/n1/3d/1/N1.vl");
  timeE = timer.getMilliseconds();
  printf("Time to load the uchar:  %u\n", timeE-timeS);
  fflush(stdout);
  exit(0);
  

  timeS = timer.getMilliseconds();
  CubeInMemmoryFloat* cbimf = new CubeInMemmoryFloat(cbimu->width,
                                                     cbimu->height,
                                                     cbimu->depth); 
  timeE = timer.getMilliseconds();
  printf("Time to allocate the float:  %u\n", timeE-timeS);

  vector< float > mask = Mask::gaussian_mask(2, 4, true);
  printf("Doing the convolution with mask of size %i ...\n", mask.size());

  timeS = timer.getMilliseconds();
  cbimu->convolve_horizontally(mask, cbimf);
  timeE = timer.getMilliseconds();
  printf("Time to do the convolution:  %u\n", timeE-timeS);

  timeS = timer.getMilliseconds();
  cbimf->saveToFile("/media/neurons/n1/3d/1/out.vl");
  timeE = timer.getMilliseconds();
  printf("Time to save the file:  %u\n", timeE-timeS);

  printf("Exiting ...\n");
}
