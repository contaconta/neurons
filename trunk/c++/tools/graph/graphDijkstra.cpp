
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
// Contact <german.gonzalez@epfl.ch> for comments & bug reports        //
/////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "Graph.h"
#include <float.h>

using namespace std;

//Auxiliary class for the points in the dijkstra algorithm
class PD
{
public:
  int idx;
  int prev;
  PD(int _idx, int _prev){
    idx = _idx; prev = _prev;
  }
};


void computeAuxStructures
( Graph<Point3D, EdgeW<Point3D> >* gr,
  vector< vector< float > >& distanceMatrix,
  vector< vector< int   > >& neighbors)
{
  int nPoints = gr->cloud->points.size();
  distanceMatrix.resize(nPoints);
  for(int i = 0; i < nPoints; i++){
    distanceMatrix[i].resize(nPoints);
  }
  for(int i = 0; i < nPoints; i++)
    for(int j = 0; j < nPoints; j++)
      distanceMatrix[i][j] = FLT_MAX;
  neighbors.resize(nPoints);
  for(int i = 0; i < gr->eset.edges.size(); i++){
    neighbors[gr->eset.edges[i]->p0].push_back(gr->eset.edges[i]->p1);
    neighbors[gr->eset.edges[i]->p1].push_back(gr->eset.edges[i]->p0);
    EdgeW<Point3D> * edg = dynamic_cast<EdgeW<Point3D>* >(gr->eset.edges[i]);
    distanceMatrix[gr->eset.edges[i]->p0][gr->eset.edges[i]->p1] = edg->w;
    distanceMatrix[gr->eset.edges[i]->p1][gr->eset.edges[i]->p0] = edg->w;
  }
}


void runDijkstra
(Graph<Point3D, EdgeW<Point3D> >* gr,
 int sourceNode,
 vector< float >& distances,
 vector< int   >& previous,
 vector< vector< float > >& distanceMatrix,   //to speed up computation
 vector< vector< int   > >& neighbours)
{
  int nPoints = gr->cloud->points.size();
  distances.resize(nPoints);
  previous .resize(nPoints);
  vector<char> visited(nPoints);
  for(int i = 0; i < nPoints; i++){
    distances[i] = FLT_MAX;
    previous[i]  = -1;
    visited[i]   =  0;
  }
  multimap<float, PD> boundary; //keeps the priority queue
  boundary.insert(pair<float, PD>(0, PD(sourceNode, 0) ) );
  distances[sourceNode] = 0;
  previous [sourceNode] = 0;
  visited  [sourceNode] = 0;

  multimap< float, PD >::iterator itb;
  int pit; //point iteration
  int previt;
  float cit;
  int counter = 0;
  while(!boundary.empty()){
    itb = boundary.begin();  //pop
    cit = itb->first;
    PD tmp = itb->second;
    pit = tmp.idx;
    previt = tmp.prev;
    boundary.erase(itb);
    if(visited[pit]==1)
      continue; //the point is already evaluated
    visited  [pit] = 1;
    distances[pit] = cit;
    previous [pit] = previt;
    counter++;
    //And now expand the point
    for(int i = 0; i < neighbours[pit].size(); i++){
      if(!visited[neighbours[pit][i]]){
        boundary.insert(pair<float, PD>
                       (cit+distanceMatrix[pit][neighbours[pit][i]],
                        PD(neighbours[pit][i], pit)));
      }
    }
  }
  printf("At the end we have seen %i points of %i\n", counter, nPoints);
}

void traceBack
(int sourceNode,
 int nodeToStart,
 vector<int>& previous,
 vector<int>& path)
{
  int nodeT = nodeToStart;
  path.resize(0);
  path.push_back(nodeT);
  while(nodeT != sourceNode){
    nodeT = previous[nodeT];
    if(nodeT == -1){
      printf("There is something awfully wrong\n");
      break;
    }
    path.push_back(nodeT);
  }
}

void allShortestPaths
( Graph<Point3D, EdgeW<Point3D> >* gr,
  vector< vector< vector< int   > > >& paths,
  vector< vector< float > >& costs)
{
  // Temporal variables
  vector< vector< float > > distanceMatrix;
  vector< vector< int   > > neighbors;
  vector< float > distances;
  vector< int   > previous ;
  vector< int   > path;

  computeAuxStructures(gr, distanceMatrix, neighbors);

  // Output
  int nPoints = gr->cloud->points.size();
  paths.resize(nPoints); costs.resize(nPoints);
  for(int i = 0; i < nPoints; i++){
    paths[i].resize(nPoints); costs[i].resize(nPoints);
  }

  for(int i = 0; i < nPoints; i++){
    runDijkstra(gr, i, distances, previous, distanceMatrix, neighbors);
    for(int j = 0; j < nPoints; j++){
      traceBack(i, j, previous, path);
      for(int nE = 0; nE < path.size(); nE++){
        (paths[i][j]).push_back(path[nE]);
      }
      costs[i][j] = distances[j];
    }
  }

}

int main(int argc, char **argv) {

  Graph<Point3D, EdgeW<Point3D> >* gr =
    new Graph<Point3D, EdgeW<Point3D> >(argv[1]);

  vector< vector< vector< int   > > > paths;
  vector< vector< float > > costs;
  allShortestPaths(gr, paths, costs);
}
