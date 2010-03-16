
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
#include "tiffio.h"

using namespace std;

int main(int argc, char **argv) {

  if( argc!=9){
    printf("Usage: cubeFromImageStack-16bitTIFF directory image_format nlayer_b nlayer_e voxel_width voxel_height voxel_depth volume_name\n");
    exit(0);
  }

  string directory(argv[1]);
  string imageFormat(argv[2]);
  int    nlayerb = atoi(argv[3]);
  int    nlayere = atoi(argv[4]);
  float  voxelw  = atof(argv[5]);
  float  voxelh  = atof(argv[6]);
  float  voxeld  = atof(argv[7]);
  string volumeName(argv[8]);

  char format[1024];
  char imageName[1024];
  uint32 width, height, depth;
  uint16 bps;
  depth = nlayere - nlayerb + 1;
  sprintf(format, "%s/%s", directory.c_str(), imageFormat.c_str());
  sprintf(imageName, format, nlayerb);
  TIFF* tif = TIFFOpen(imageName, "r");
  TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
  TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
  TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bps);
  TIFFClose(tif);

  if(bps!= 16){
    printf("The image files do not have 16 bits, try other program\n");
    exit(0);
  }

  Cube<int, ulong>* dest =
    new Cube<int, ulong>(width, height, depth, volumeName,
                           voxelw, voxelh, voxeld);
  uint16 array[width*height];
  for(int i = nlayerb; i <= nlayere; i++){

    sprintf(imageName, format, i);
    TIFF* tif = TIFFOpen(imageName, "r");
    for (int j = 0; j < height; j++)
      TIFFReadScanline(tif, &array[j * width], j, 0);
    for(int y = 0; y < height; y++){
      for(int x = 0; x < width; x++){
        dest->put(x,y,i-nlayerb, (array[y*width+x]));
      }
    }
    TIFFClose(tif);
  }

  exit(0);
}
