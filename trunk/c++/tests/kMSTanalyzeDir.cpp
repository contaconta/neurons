
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

  if(argc!=5) {
    printf("Usage: kMSTanalyzeDir directory kInit kEnd kStep\n");
    exit(0);
  }

  //string directory = "/home/ggonzale/mount/bbpsg1scratch/n2/3d/tree/";
  string directory = argv[1];
  Cube<float, double>* svm = new Cube<float, double>
    (directory+ "/2/merged_248.nfo");
  int kInit = atoi(argv[2]);;
  int kEnd  = atoi(argv[3]);
  int kStep = atoi(argv[4]);
  char nameGraph[2048];
  char nameEdge[2048];

  vector< vector< double > > toSave;
  EdgeW<Point3D>* ed;

  for(int k = kInit; k <=kEnd; k+=kStep){
    sprintf(nameGraph, "%s/tree/kmsts/kmst%i.gr", directory.c_str(), k);
    if(!fileExists(nameGraph))
      continue;
    printf("%s\n", nameGraph);
    Graph< Point3D, EdgeW<Point3D> >* gr =
      new Graph< Point3D, EdgeW<Point3D> >(nameGraph);
    double val = 0;
    double loglikelihoodratio = 0;
    for(int i = 0; i < gr->eset.edges.size(); i++){
      ed = dynamic_cast< EdgeW<Point3D>*>( gr->eset.edges[i]);
      val += ed->w;
      //      loglikelihoodratio += log10(pow(10,-ed->w)/(1-pow(10,-ed->w)));
      sprintf(nameEdge, "%s/tree/paths/path_%04i_%04i.gr",
              directory.c_str(), ed->p0, ed->p1);
      if(!fileExists(nameEdge)){
        sprintf(nameEdge, "%s/tree/paths/path_%04i_%04i.gr",
                directory.c_str(), ed->p1, ed->p0);}
      Graph< Point3D, EdgeW<Point3D> >* path =
        new Graph< Point3D, EdgeW<Point3D> >(nameEdge);
      loglikelihoodratio +=
        svm->integralOverCloud(path->cloud);
    }
    vector< double > it(3);
    it[0] = k; it[1] = val; it[2] = loglikelihoodratio;
    toSave.push_back(it);
  }
  saveMatrix(toSave, directory + "/tree/kmsts/costs.txt");

}
