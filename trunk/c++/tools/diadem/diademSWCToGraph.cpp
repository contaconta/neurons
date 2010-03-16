
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
#include "CubeFactory.h"
#include "SWC.h"

using namespace std;

int main(int argc, char **argv) {

  if(argc!=4){
    printf("Usage: SWCtoGraph SWC.swc cubeToTranslate.nfo out.gr\n");
    exit(0);
  }

  SWC* swc     = new SWC(argv[1]);
  Cube_P* cube = CubeFactory::load(argv[2]);
  Graph<Point3Dw>* out = swc->toGraphInMicrometers(cube);

  out->saveToFile(argv[3]);

} 
