
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
#include "Graph.h"
#include "Cube.h"
#include "utils.h"

using namespace std;

int main(int argc, char **argv) {

  string directory = "/home/ggonzale/mount/bbpsg1scratch/n2/3d/tree/kmsts/";
  int kInit = 200;
  int kEnd  = 800;
  int kStep = 20;
  char name[2048];

  vector< vector< double > > toSave;
  EdgeW<Point3D>* ed;

  for(int k = kInit; k <=kEnd; k+=kStep){
    sprintf(name, "%s/kmst%i.gr", directory.c_str(), k);
    printf("%s\n", name);
    Graph< Point3D, EdgeW<Point3D> >* gr =
      new Graph< Point3D, EdgeW<Point3D> >(name);
    double val = 0;
    for(int i = 0; i < gr->eset.edges.size(); i++){
      ed = dynamic_cast< EdgeW<Point3D>*>( gr->eset.edges[i]);
      val += ed->w;
    }
    vector< double > it(2);
    it[0] = k; it[1] = val;
    toSave.push_back(it);
  }
  saveMatrix(toSave, directory + "costs.txt");

}
