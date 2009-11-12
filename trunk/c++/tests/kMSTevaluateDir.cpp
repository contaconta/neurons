
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
#include "Neuron.h"

using namespace std;

void renderPointInCube
(Cube<uchar,ulong>* cube, Point3Dw* pt)
{
  int radious, k,kk, cx, cy, cz;
  cube->micrometersToIndexes3(pt->weight, 0, 0, k, cy, cz);
  cube->micrometersToIndexes3(2*pt->weight, 0, 0, kk,cy,cz);
  radious = kk-k;
  //printf("   the radious is %f -> %i\n", pt->weight, radious);
  cube->micrometersToIndexes3(pt->coords[0], pt->coords[1], pt->coords[2],
                              cx, cy, cz);
  int radious2 = radious*radious;
  if(radious < 1) radious = 1;
  int zStep = max(2, radious/4);
  for(int x = max(0, cx-radious); x < min(cx+radious, (int)cube->cubeWidth); x++)
    for(int y = max(0, cy-radious); y < min(cy+radious, (int)cube->cubeHeight); y++)
      for(int z = max(0, cz-zStep); z < min(cz+zStep, (int)cube->cubeDepth); z++){
        if( ((x-cx)*(x-cx)+(y-cy)*(y-cy)+(z-cz)*(z-cz)) < radious2){
          cube->put(x,y,z,0);
        }
      }
}

void renderGraphInCube
(Cube<uchar,ulong>* cube, Graph<Point3Dw, EdgeW<Point3Dw> >* gr)
{
  for(int i = 0; i < gr->cloud->points.size(); i++){
    Point3Dw* pt = dynamic_cast<Point3Dw*>( gr->cloud->points[i]);
    //printf("  pt %i / %i\n", i, gr->cloud->points.size());
    renderPointInCube(cube, pt);
  }
}

void integrateSegmentOverCube
(Neuron* neuron, NeuronSegment* segment,
 Cube<uchar, ulong>* cube, double& value, int& numberOfVoxels)
{
  for(int i = 0; i < segment->childs.size(); i++){
    integrateSegmentOverCube(neuron, segment->childs[i], cube, value, numberOfVoxels);
  }
  vector< float > neuronCoords1(3);
  vector< float > neuronCoords2(3);
  vector< float > micromCoords1(3);
  vector< float > micromCoords2(3);
  vector< int   > idx1(3);
  vector< int   > idx2(3);
  for(int i = 0; i < segment->points.size()-1; i++){
    neuronCoords1[0] = segment->points[i].coords[0];
    neuronCoords1[1] = segment->points[i].coords[1];
    neuronCoords1[2] = segment->points[i].coords[2];
    neuronCoords2[0] = segment->points[i+1].coords[0];
    neuronCoords2[1] = segment->points[i+1].coords[1];
    neuronCoords2[2] = segment->points[i+1].coords[2];
    neuron->neuronToMicrometers(neuronCoords1, micromCoords1);
    neuron->neuronToMicrometers(neuronCoords2, micromCoords2);
    cube->micrometersToIndexes(micromCoords1, idx1);
    cube->micrometersToIndexes(micromCoords2, idx2);
    //If the edge is in the cube, compute the integral
    if( (idx1[0] >= 0) && (idx1[1] >= 0) && (idx1[2] >= 0) &&
        (idx2[0] >= 0) && (idx2[1] >= 0) && (idx2[2] >= 0) &&
        (idx1[0] < cube->cubeWidth) && (idx1[1] < cube->cubeHeight) && (idx1[2] < cube->cubeDepth) &&
        (idx2[0] < cube->cubeWidth) && (idx2[1] < cube->cubeHeight) && (idx2[2] < cube->cubeDepth)
        ){
      double integral; int length;
      cube->integral_between
        (idx1[0], idx1[1], idx1[2],
         idx2[0], idx2[1], idx2[2],
         integral, length);
      value += integral;
      numberOfVoxels += length;
    }
  }
}


void integrateNeuronOverCube
(Neuron* neuron, Cube<uchar, ulong>* cube, double& value, int& numberOfVoxels)
{
  for(int i = 0; i < neuron->axon.size(); i++){
    integrateSegmentOverCube(neuron, neuron->axon[i], cube, value, numberOfVoxels);
  }
  for(int i = 0; i < neuron->dendrites.size(); i++){
    integrateSegmentOverCube(neuron, neuron->dendrites[i], cube, value, numberOfVoxels);
  }
}


void evaluateGraph
( string directory,
  Cube<uchar, ulong>* mask,
  Cube<uchar, ulong>* rendered,
  Neuron* neuron,
  Graph<Point3Dw, EdgeW<Point3Dw> >* gr,
  double& tp, double& fp, double& fn)
{
  tp = 0;
  fp = 0;
  rendered->put_all(255);
  EdgeW<Point3Dw>* ed;
  char nameEdge[2048];
  for(int i = 0; i < gr->eset.edges.size(); i++){
    ed = dynamic_cast< EdgeW<Point3Dw>*>( gr->eset.edges[i]);
    sprintf(nameEdge, "%s/tree/paths/path_%04i_%04iw.gr",
            directory.c_str(), ed->p0, ed->p1);
    if(!fileExists(nameEdge)){
      sprintf(nameEdge, "%s/tree/paths/path_%04i_%04iw.gr",
              directory.c_str(), ed->p1, ed->p0);}
    if(!fileExists(nameEdge)){
      printf("The path %s does not exist. Continuing\n");
      continue;
    }
    printf(" %s\n", nameEdge);
    Graph< Point3Dw, EdgeW<Point3Dw> >* path =
      new Graph< Point3Dw, EdgeW<Point3Dw> >(nameEdge);
    renderGraphInCube(rendered, path);
    int val = mask->integralOverCloud(path->cloud)/255;
    tp +=
      path->cloud->points.size() - val;
    fp += val;
  }
  double neuronIntegral = 0;
  int    neuronLength   = 0;
  integrateNeuronOverCube(neuron, rendered,
                          neuronIntegral, neuronLength);
  fn = neuronIntegral/255;
}




int main(int argc, char **argv) {

  if(argc!=7) {
    printf("Usage: kMSTanalyzeDir directory kInit kEnd kStep groundTruth.nfo pp.asc\n");
    exit(0);
  }

  //string directory = "/home/ggonzale/mount/bbpsg1scratch/n2/3d/tree/";
  string directory = argv[1];
  Cube<uchar, ulong>* mask = new Cube<uchar, ulong>(argv[5]);
  Cube<uchar, ulong>* rendered = new Cube<uchar, ulong>(argv[5], false);
  rendered->load_volume_data("", false);
  //Cube<uchar, ulong>* rendered = mask->duplicate_clean("rendered");
  int kInit = atoi(argv[2]);;
  int kEnd  = atoi(argv[3]);
  int kStep = atoi(argv[4]);
  Neuron* gtNeuron = new Neuron(argv[6]);
  char nameGraph[2048];

  vector< vector< double > > toSave;

  //Counts all the negative points in the ground truth
  int negativePoints = 0;
  for(int z = 0; z < mask->cubeDepth; z++)
    for(int y = 0; y < mask->cubeHeight; y++)
      for(int x = 0; x < mask->cubeWidth; x++)
        if(mask->at(x,y,z) == 255)
          negativePoints++;

  double tp = 0;
  double fp = 0;
  double fn = 0;

  if(0){
    for(int k = kInit; k <=kEnd; k+=kStep){
      tp = 0; fp = 0; fn = 0;
      sprintf(nameGraph, "%s/tree/kmsts/kmst%iw.gr", directory.c_str(), k);
      printf("%s\n", nameGraph);
      Graph< Point3Dw, EdgeW<Point3Dw> >* gr =
        new Graph< Point3Dw, EdgeW<Point3Dw> >(nameGraph);
      evaluateGraph(directory, mask, rendered, gtNeuron, gr, tp, fp, fn);
      vector< double > it(5);
      double tn = negativePoints - fp;
      it[0] = k; it[1] = tp; it[2] = fp; it[3] = tn; it[4] = fn;
      toSave.push_back(it);
    }
    saveMatrix(toSave, directory + "/tree/kmsts/evaluation.txt");
  }

  //Evaluatoin of the mst
  if(0){
    toSave.resize(0);
    sprintf(nameGraph, "%s/tree/mst/mstFromCptGraphw.gr", directory.c_str());
    printf("Evaluating the graph: %s\n", nameGraph);
    tp = 0; fp = 0; fn = 0;
    Graph< Point3Dw, EdgeW<Point3Dw> >* gr =
      new Graph< Point3Dw, EdgeW<Point3Dw> >(nameGraph);
    printf("Loaded the graph: %s\n", nameGraph);
    evaluateGraph(directory, mask, rendered, gtNeuron, gr, tp, fp, fn);
    vector< double > it(5);
    double tn = negativePoints - fp;
    it[0] = 0; it[1] = tp; it[2] = fp; it[3] = tn; it[4] = fn;
    toSave.push_back(it);
    saveMatrix(toSave, directory + "/tree/mst/mstFromCptGraphwE.txt");
  }

  if(1){
    toSave.resize(0);
    sprintf(nameGraph, "%s/tree/mst/mstFromCptGraphPrunedw.gr", directory.c_str());
    printf("Evaluating the graph: %s\n", nameGraph);
    tp = 0; fp = 0; fn = 0;
    Graph< Point3Dw, EdgeW<Point3Dw> >* gr =
      new Graph< Point3Dw, EdgeW<Point3Dw> >(nameGraph);
    printf("Loaded the graph: %s\n", nameGraph);
    evaluateGraph(directory, mask, rendered, gtNeuron, gr, tp, fp, fn);
    vector< double > it(5);
    double tn = negativePoints - fp;
    it[0] = 0; it[1] = tp; it[2] = fp; it[3] = tn; it[4] = fn;
    toSave.push_back(it);
    saveMatrix(toSave, directory + "/tree/mst/mstFromCptGraphwPrunedE.txt");
  }
}
