
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
#include "Graph.h"
#include "Cloud.h"
#include "CloudFactory.h"
#include "GraphFactory.h"


using namespace std;

Graph<Point3D, EdgeW<Point3D> >* gr;

vector<int> findEdgesThatTouchPoint(int nPoint)
{
  vector<int> toReturn;
  for(int i = 0; i < gr->eset.edges.size(); i++)
    if( (gr->eset.edges[i]->p0 == nPoint) ||
        (gr->eset.edges[i]->p1 == nPoint) )
      toReturn.push_back(i);
  return toReturn;
}

// nEdge should be passed as -1 in the initial call
void traceBackToSoma(int nPoint, int nEdge, vector<int>& visitedEdges)
{
  //If we are in the soma
  if(nPoint == 0){
    visitedEdges.push_back(nEdge);
    return;
  }

  int sizeNow = visitedEdges.size();
  vector<int> edgesThatTouch = findEdgesThatTouchPoint(nPoint);

  for(int i = 0; i < edgesThatTouch.size(); i++){
    if(edgesThatTouch[i] == nEdge) continue;
    int otherPoint = 0;
    if( gr->eset.edges[edgesThatTouch[i]]->p0 == nPoint)
      otherPoint = gr->eset.edges[edgesThatTouch[i]]->p1;
    else otherPoint = gr->eset.edges[edgesThatTouch[i]]->p0;
    traceBackToSoma(otherPoint, edgesThatTouch[i], visitedEdges);
  }

  int sizeAfterKids = visitedEdges.size();
  if((sizeAfterKids > sizeNow) && (nEdge!=-1)){
    visitedEdges.push_back(nEdge);
  }
}

int main(int argc, char **argv) {

  // if(argc!=3){
    // printf("graphPrune graph.gr graphPruned.gr\n");
    // exit(0);
  // }

  gr = new Graph<Point3D, EdgeW<Point3D> >
    ("/media/neurons/steerableFilters3D/tmp/cut2NegatedEuclideanAnisotropic/mstFromCptGraph.gr");

  printf("Finding the leaves\n");
  vector<int> leaves = gr->findLeaves();
  //Save the leaves as a cloud for visualization purposses
  vector<int> idx(3);
  vector<float> mic(3);
  Cloud<Point3D>* leavescl = new Cloud<Point3D>();
  for(int i = 0; i < leaves.size(); i++){
    leavescl->points.push_back
      (new Point3D(gr->cloud->points[leaves[i]]->coords[0],
                   gr->cloud->points[leaves[i]]->coords[1],
                   gr->cloud->points[leaves[i]]->coords[2]));
  }
  //Leaves are green
  leavescl->v_r = 0;
  leavescl->v_g = 1;
  leavescl->v_b = 0;
  leavescl->v_radius = 0.8;
  leavescl->saveToFile("/media/neurons/steerableFilters3D/tmp/cut2NegatedEuclideanAnisotropic/leaves.cl");



  //Check if the edge has already been visited in the tree (should be kept or removed)
  vector<int> edgesVisited(gr->eset.edges.size());
  for(vector<int>::iterator it = edgesVisited.begin();
      it != edgesVisited.end(); it++)
    *it = 0;

  // Point in which the edges should be elliminated
  float threshold = 0;

  vector<int> edgesTraced;
  for(int i = 0; i < leaves.size(); i++){
  // for(int i = 10; i < 11; i++){
    printf("  analizing leave %i\n", i);
    fflush(stdout);
    edgesTraced.resize(0);
    traceBackToSoma(leaves[i], -1, edgesTraced);
    float integral = 0;
    int split = edgesTraced.size(); //We keep all the points
    // for(vector<int>::iterator it = edgesTraced.end();
    // it != edgesTraced.begin(); it--){
    for(int it = edgesTraced.size() -1; it >= 0; it--){
      integral = integral  + log( (1-gr->eset.edges[edgesTraced[it]]->w)
                                /(gr->eset.edges[edgesTraced[it]]->w) );
      if(integral < threshold){
        split = it;
        integral = threshold;
      }
    }
    printf("    split = %i,  points %i\n", split, edgesTraced.size());
    //From the soma to split will be in the tree
    for(int i = 0; i < split; i++){
      edgesVisited[edgesTraced[i]] = 1;
    }
    //From the split to the end will not be on the tree
    for(int i = split; i < edgesTraced.size(); i++){
      if(edgesVisited[edgesTraced[i]] != 1){
        edgesVisited[edgesTraced[i]] = -1;
      }
    }
  }



  for(int i = 0; i < edgesVisited.size(); i++)
    printf("%i ", edgesVisited[i]);
  printf("\n");

  printf("Saving the pruned tree\n");
  Graph<Point3D, EdgeW<Point3D> >* pruned =
    new Graph<Point3D, EdgeW<Point3D> >();
  pruned->cloud = gr->cloud;
  for(int i = 0; i < edgesVisited.size(); i++){
    if(edgesVisited[i] > 0)
      pruned->eset.edges.push_back
        (new EdgeW<Point3D>(&pruned->cloud->points,
                            gr->eset.edges[i]->p0,
                            gr->eset.edges[i]->p1,
                            gr->eset.edges[i]->w));
  }
  pruned->saveToFile("/media/neurons/steerableFilters3D/tmp/cut2NegatedEuclideanAnisotropic/pruned.gr");


  std::ofstream out("/media/neurons/steerableFilters3D/tmp/cut2NegatedEuclideanAnisotropic/pruned.lst");
  std::ofstream out2("/media/neurons/steerableFilters3D/tmp/cut2NegatedEuclideanAnisotropic/pruned_out.lst");
  char buff[1024];
  for(int i = 0; i < edgesVisited.size(); i++){
  // for(int i = 0; i < 1; i++){
    if(edgesVisited[i] >= 0){
      sprintf(buff, "/media/neurons/steerableFilters3D/tmp/cut2NegatedEuclideanAnisotropic/path_%04i_%04i.gr",
              gr->eset.edges[i]->p0,
              gr->eset.edges[i]->p1);
      if(fileExists(buff)){
        out << buff << std::endl;
      }
      sprintf(buff, "/media/neurons/steerableFilters3D/tmp/cut2NegatedEuclideanAnisotropic/path_%04i_%04i.gr",
              gr->eset.edges[i]->p1,
              gr->eset.edges[i]->p0);
      if(fileExists(buff)){
        out << buff << std::endl;
      }
    } else {
      sprintf(buff, "/media/neurons/steerableFilters3D/tmp/cut2NegatedEuclideanAnisotropic/path_%04i_%04i.gr",
              gr->eset.edges[i]->p0,
              gr->eset.edges[i]->p1);
      if(fileExists(buff)){
        out2 << buff << std::endl;
      }
      sprintf(buff, "/media/neurons/steerableFilters3D/tmp/cut2NegatedEuclideanAnisotropic/path_%04i_%04i.gr",
              gr->eset.edges[i]->p1,
              gr->eset.edges[i]->p0);
      if(fileExists(buff)){
        out2 << buff << std::endl;
      }
    }
  }
  out.close();
  out2.close();
}
