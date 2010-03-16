
////////////////////////////////////////////////////////////////////////////////
// This program is free software; you can redistribute it and/or              //
// modify it under the terms of the GNU General Public License                //
// version 2 as published by the Free Software Foundation.                    //
//                                                                            //
// This program is distributed in the hope that it will be useful, but        //
// WITHOUT ANY WARRANTY; without even the implied warranty of                 //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU          //
// General Public License for more details.                                   //
//                                                                            //
// Written and (C) by German Gonzalez Serrano                                 //
// Contact < german.gonzalez@epfl.ch > for comments & bug reports             //
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "Image.h"

using namespace std;

int main(int argc, char **argv) {
  if(argc!=6){
    printf("imageMergeAcrossScales img1.jpg img2.jpg img4.jpg out.jpg scale.jpg\n");
    exit(0);
  }

  Image< float >* orig_1 = new Image<float>(argv[1]);
  Image< float >* orig_2 = new Image<float>(argv[2]);
  Image< float >* orig_4 = new Image<float>(argv[3]);
  Image< float >* dest   = orig_1->create_blank_image_float(argv[4]);
  Image< float >* scale  = orig_1->create_blank_image_float(argv[5]);

  //No fancy stuff, just stupid non-interpolation
  for(int y = 0; y < dest->height; y++){
    for(int x = 0; x < dest->width; x++){
      dest->put(x,y,orig_1->at(x,y));
      scale->put(x,y,1);
      if(orig_2->at( min(x/2,orig_2->width-1),min(y/2,orig_2->height-1) ) > dest->at(x,y)){
        dest->put(x,y,orig_2->at(  min(x/2,orig_2->width-1),min(y/2,orig_2->height-1) ));
        scale->put(x,y,2);
      }
      /*
        if(orig_4->at(min(x/4,orig_4->width-1),min(y/4,orig_4->height-1))
        > dest->at(x,y)){
        dest->put(x,y,orig_4->at( min(x/4,orig_4->width-1),min(y/4,orig_4->height-1) ));
        scale->put(x,y,4);
      }
      */
    }
  }

  dest->save();
  scale->save();


}
