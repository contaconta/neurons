
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

#include "Mask.h"

using namespace std;

int main(int argc, char **argv) {

  printf("dfactorial of -1 should be  1 and is: %i\n", dfactorial_n(-1));
  printf("dfactorial of  0 should be  1 and is: %i\n", dfactorial_n( 0));
  printf("dfactorial of  1 should be  1 and is: %i\n", dfactorial_n( 1));
  printf("dfactorial of  2 should be  2 and is: %i\n", dfactorial_n( 2));
  printf("dfactorial of  3 should be  3 and is: %i\n", dfactorial_n( 3));
  printf("dfactorial of  4 should be  8 and is: %i\n", dfactorial_n( 4));
  printf("dfactorial of  5 should be 15 and is: %i\n", dfactorial_n( 5));

}
