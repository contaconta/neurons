
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
#include "GraphFactory.h"
#include <iostream>

using namespace std;

int main(int argc, char **argv) {

  if(argc!=3){
    printf("Usage: cloudToDisplayList <Graph.gr> <display.lst>\n");
    exit(0);
  }

  std::ofstream out(argv[2]);
  char buff[1024];
  Graph<Point3D, EdgeW< Point3D> >* gr =
    new Graph<Point3D, EdgeW< Point3D> >(argv[1]);
  printf("The graph has %i edges\n", gr->eset.edges.size());

  for(int i =0; i < gr->eset.edges.size(); i++){
      if( gr->eset.edges[i]->p0 ==
          gr->eset.edges[i]->p1 )
        continue;
    sprintf(buff, "path_%04i_%04i.gr",
            gr->eset.edges[i]->p0,
            gr->eset.edges[i]->p1);
    out << buff << std::endl;
  }
  out.close();
}
