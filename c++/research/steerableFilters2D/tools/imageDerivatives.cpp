
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
#include <string>
#include "Image.h"
#include "utils.h"
// #include <argp.h>






using namespace std;

int main(int argc, char **argv) {

  if(argc!= 4){
    printf("Usage: imageDerivatives <image> <order> <sigma>\n");
    exit(0);
  }

  string name = argv[1];
  int M = atoi(argv[2]);
  double sigma = atof(argv[3]);
  Image<float>* img = new Image<float>(name);
  Image<float>* result;


  ifstream inp;
  char buff[512];
  sprintf(buff, "_%02.02f.jpg",sigma);
  string directory = getDirectoryFromPath(name);
  string name_b = directory + "g" + buff;

  int k_idx = 0;
  for(int k = 1; k <= M; k++){
    for(int j = 0; j <= k; j++){
      string name = directory + "g_";
      for(int l = 0; l < k-j; l++)
        name = name + "x";
      for(int l = 0; l < j; l++)
        name = name + "y";
      name = name + buff;
      //Check for the existance of the file
      inp.open(name.c_str(), ifstream::in);
      if(inp.fail()){
        result = img->calculate_derivative(k-j,j,sigma, name);
        // delete result;
      }
      inp.close();
    }
    k_idx += (k+1);
  }




}
