
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

using namespace std;

int main(int argc, char **argv) {

  if(argc!=3){
    printf("Usage: cibeEvaluateInPositiveNegatives cube mask\n");
    exit(0);
  }

  Cube<float, double>* segmented = new Cube<float, double>(argv[1]);
  Cube<uchar, ulong>* mask      = new Cube<uchar, ulong>(argv[2]);

  string cubeName = getNameFromPathWithoutExtension(argv[1]);
  Cube<uchar, ulong>* TP = mask->duplicate_clean(cubeName + "_TP");
  Cube<uchar, ulong>* FP = mask->duplicate_clean(cubeName + "_FP");
  Cube<uchar, ulong>* FN = mask->duplicate_clean(cubeName + "_FN");

  for(int z = 0; z < segmented->cubeDepth; z++)
    for(int y = 0; y < segmented->cubeHeight; y++)
      for(int x = 0; x < segmented->cubeWidth; x++){
        //TRUE POSITIVES
        if( (mask->at(x,y,z) < 100) && (segmented->at(x,y,z) > 100))
          TP->put(x,y,z,255);
        //FALSE POSITIVES
        if( (mask->at(x,y,z) > 100 ) && (segmented->at(x,y,z) > 100))
          FP->put(x,y,z,255);
        //FALSE NEGATIVES
        if( (mask->at(x,y,z) < 100) && (segmented->at(x,y,z) < 100 ))
          FN->put(x,y,z,255);

      }

}
