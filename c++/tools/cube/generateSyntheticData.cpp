
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

void generate_lines(){

  Cube<uchar, ulong>* cube = new Cube<uchar,ulong>
    ("/media/neurons/synthetic/synthetic.nfo",false);
  cube->create_volume_file(cube->directory + cube->filenameVoxelData);
  cube->load_volume_data(cube->directory + cube->filenameVoxelData);
  cube->put_all(255);

  vector<int> idx1(3);
  idx1[0] = 20; idx1[1] = 20; idx1[2] = 20;
  vector<int> idx2(3);
  idx2[0] = 108; idx2[1] = 20; idx2[2] = 20;
  cube->render_cylinder(idx1, idx2, 5);

  idx2[0] = 20; idx2[1] = 108;
  cube->render_cylinder(idx1, idx2, 5);

  idx2[1] = 20; idx2[2] = 108;
  cube->render_cylinder(idx1, idx2, 5);

  idx2[0] = 108; idx2[1] = 108; idx2[2] = 108;
  cube->render_cylinder(idx1, idx2, 5);

  idx1[1] = 108; idx1[2] = 108;
  cube->render_cylinder(idx1, idx2, 5);

  idx1[0] = 108; idx1[1] = 20;
  cube->render_cylinder(idx1, idx2, 5);

  idx1[1] = 108; idx1[2] = 20;
  cube->render_cylinder(idx1, idx2, 5);

  idx1[0] = 20; idx1[1] = 20;
  cube->render_cylinder(idx1, idx2, 5);

  idx1[0] = 20; idx1[1] = 20; idx1[2] = 108;
  idx2[0] = 108; idx2[1] = 108; idx2[2] = 20;
  cube->render_cylinder(idx1, idx2, 2.5);

  idx1[0] = 108; idx1[1] = 20; idx1[2] = 108;
  idx2[0] = 64; idx2[1] = 64; idx2[2] = 64;
  cube->render_cylinder(idx1, idx2, 5);

  idx1[0] = 108; idx1[1] = 108; idx1[2] = 20;
  idx2[0] = 64; idx2[1] = 64; idx2[2] = 64 ;
  cube->render_cylinder(idx1, idx2, 5);

  idx1[0] = 20; idx1[1] = 20; idx1[2] = 20;
  idx2[0] = 108; idx2[1] = 20; idx2[2] = 108 ;
  cube->render_cylinder(idx1, idx2, 5);

  idx1[0] = 20; idx1[1] = 20; idx1[2] = 108;
  idx2[0] = 108; idx2[1] = 20; idx2[2] = 20 ;
  cube->render_cylinder(idx1, idx2, 5);



}



void generate_delta(){
  Cube<uchar, ulong>* cube = new Cube<uchar,ulong>
    ("/media/neurons/filter/filter.nfo",false);
  cube->create_volume_file(cube->directory + cube->filenameVoxelData);
  cube->load_volume_data(cube->directory + cube->filenameVoxelData);
  cube->put_all(0);
  cube->put(32,32,32,255);

  printf("Center: %u\n", cube->at(32,32,32));
  printf("EndL: %u\n", cube->at(32,32,64));
}


int main(int argc, char **argv) {
  generate_lines();
//   generate_delta();


}
