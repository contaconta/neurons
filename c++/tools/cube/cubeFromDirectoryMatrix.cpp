
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

int main(int argc, char **argv) {

  if(argc!=12){
    printf("Usage: cubeFromirectoryMatrix <directory> <directory_format/image_format> <row_begin> <row_end> <col_begin> <col_end> <layer_begin> <layer_end> <voxel_width> <voxel_height> <voxel_depth>\n");
    exit(0);
  }

  string directory = argv[1];
  Cube<uchar, ulong>* source = new Cube<uchar,ulong>();
  source->create_cube_from_directory_matrix(
                                            argv[1], argv[2],
                                            atoi(argv[3]), atoi(argv[4]),
                                            atoi(argv[5]), atoi(argv[6]),
                                            atoi(argv[7]), atoi(argv[8]),
                                            atof(argv[9]), atof(argv[10]),
                                            atof(argv[11]) );
}
