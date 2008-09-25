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
#include "CubeFactory.h"
#include "Graph.h"
#include "Cloud.h"

using namespace std;



int main(int argc, char **argv) {

  // if(argc!=2)
    // {
      // printf("Usage: scratch cubeName\n");
      // exit(0);
    // }

  // string volume_str = argv[1];

  // Cube_P* pp = CubeFactory::load(volume_str);
  // vector<  vector< double > > pp = loadMatrix("cloud.cd");
  // printf("The size is: %i %i\n", pp.size(), pp[0].size());

  Graph* gr = new Graph("graph.gr");
  gr->save("graph2.gr");


}
