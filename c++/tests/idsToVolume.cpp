
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
// Contact <german.gonzalez@epfl.ch> for comments & bug reports        //
/////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "Cube.h"

using namespace std;

int main(int argc, char **argv) {

  if(argc != 3){
    printf("Usage: idsToVolume file.ids volumename\n");
    exit(0);
  }

  int width  = 1024;
  int height = 1024;
  int depth  = 61;
  string idsFile(argv[1]);
  string outName(argv[2]);
  Cube<float, double> * outCubeR 
    = new Cube<float,double>(width, height, depth, outName + "R", .1, .1, .8);
  Cube<float, double> * outCubeG 
    = new Cube<float,double>(width, height, depth, outName + "G", .1, .1, .8);
  Cube<float, double> * outCubeB 
    = new Cube<float,double>(width, height, depth, outName + "B", .1, .1, .8);


  int fildes = open64(idsFile.c_str(), O_RDWR);
  if(fildes == -1){ //The file does not exist{
      printf("The file %s does not exist.Aborting.\n", idsFile.c_str());
      exit(0);
  }
  void* mapped_file;
  mapped_file = mmap64(0,
                       width*height*depth*sizeof(float)*2*3,
                       PROT_READ|PROT_WRITE, MAP_SHARED, fildes, 0);
  if(mapped_file == MAP_FAILED)
    {
      printf("Cube<T,U>::load_volume_data: There is a bug here, volume not loaded. %s\n",
             idsFile.c_str());
      exit(0);
    }
  uchar* voxelsIds = (uchar*)mapped_file;

  printf("Computing the stuff [");
  for(int z = 0; z < depth; z++){
    for(int y = 0; y < height; y++)
      for(int x = 0; x < width; x++){
        outCubeR->put
          (x,y,z,
           voxelsIds[z*width*height*2*3 + y*2*3*width + x*2*3+0] +
           voxelsIds[z*width*height*2*3 + y*2*3*width + x*2*3+1]*256 //weird file format
           );
        outCubeG->put
          (x,y,z,
           voxelsIds[z*width*height*2*3 + y*2*3*width + x*2*3+2] +
           voxelsIds[z*width*height*2*3 + y*2*3*width + x*2*3+3]*256 //weird file format
           );
        outCubeB->put
          (x,y,z,
           voxelsIds[z*width*height*2*3 + y*2*3*width + x*2*3+4] +
           voxelsIds[z*width*height*2*3 + y*2*3*width + x*2*3+5]*256 //weird file format
           );
      }
    printf("#");
  }
  printf("\n");
  exit(0);

}
