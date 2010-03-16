
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
#include "Cube.h"
#include "CubeLiveWire.h"
#include "Graph.h"
#include <map>

using namespace std;

int main(int argc, char **argv) {

  vector<int>   idxs(3);
  vector<float> microm(3);

  Cube<uchar, ulong>* cube = new Cube<uchar, ulong>("/media/neurons/cut2/cut2.nfo");

  DistanceDijkstraColor* djkc = new DistanceDijkstraColor(cube);
  CubeLiveWire*      cubeLiveWire = new CubeLiveWire(cube, djkc);;

  printf("Computing the distances .... \n");
  cubeLiveWire->computeDistances(122, 85, 8);

  printf("Getting the list of the paths to all possible points [");
  fflush(stdout);

  multimap<float, Graph<Point3D, EdgeW<Point3D> >*> paths;

  // Graph<Point3D, EdgeW<Point3D> >* shortestPath =
    // cubeLiveWire->findShortestPathG(122,85,8,0,0,0);

  // shortestPath->saveToFile("out.gr");

  Cloud< Point3D>* cloud = new Cloud<Point3D>();
  idxs[0] = 122;
  idxs[1] = 85;
  idxs[2] = 8;
  cube->indexesToMicrometers(idxs,microm);
  cloud->addPoint(microm[0],microm[1],microm[2]);
  cloud->saveToFile("original.cl");

  float cost;
  for(int z = 0; z < cube->cubeDepth; z+=2){
    for(int y = 0; y < cube->cubeHeight; y+=2){
      for(int x = 0; x < cube->cubeWidth; x+=2){
          Graph<Point3D, EdgeW<Point3D> >* shortestPath =
            cubeLiveWire->findShortestPathG(122,85,8,x,y,z);
          cost = cube->integralOverCloud(shortestPath->cloud);
          paths.insert(pair<float, Graph<Point3D, EdgeW<Point3D> >*>
                       (cost/(shortestPath->cloud->points.size()), shortestPath));
      }
    }
    printf("#"); fflush(stdout);
  }
  printf("]\n");

  // Getting the first 100 paths
  multimap< float, Graph<Point3D, EdgeW<Point3D> >*>::iterator
    iter = paths.begin();

  int cubeLength = cube->cubeDepth*cube->cubeHeight*cube->cubeWidth;
  int* visited_orig   = (int*) malloc(cubeLength*sizeof(int));
  int*** visited;
  visited = (int***) malloc(cube->cubeDepth*sizeof(int**));
  for(int z = 0; z < cube->cubeDepth; z++){
    visited  [z] = (int**) malloc(cube->cubeHeight*sizeof(int*));
    for(int j = 0; j < cube->cubeHeight; j++){
      visited[z][j] =(int*) &visited_orig [(z*cube->cubeHeight + j)*cube->cubeWidth];
    }
  }
  for(int i = 0; i < cubeLength; i++)
    visited_orig[i] = 0;

  int limitEnd = 2000;
  int nGraphsAdded = 0;
  int nGraphsRejected = 0;
  while(nGraphsAdded < limitEnd){
    iter = paths.begin();
    char buff[512];
    Graph<Point3D, EdgeW<Point3D> >* gr = (*iter).second;
    int nPoint = 0;
    bool alreadyVisited = false;
    //Check if there is a visited part of the path
    for(nPoint = 0; nPoint < gr->cloud->points.size(); nPoint++){
      microm[0] = gr->cloud->points[nPoint]->coords[0];
      microm[1] = gr->cloud->points[nPoint]->coords[1];
      microm[2] = gr->cloud->points[nPoint]->coords[2];
      cube->micrometersToIndexes(microm, idxs);
      if(visited[idxs[2]][idxs[1]][idxs[0]] == true){
        alreadyVisited = true;
        break;  // we reached a visited voxel
      }
    }

    //If the path has been already visited, we better forget of the visited part
    // and add a new graph to the list
    if(alreadyVisited){
      Graph<Point3D, EdgeW<Point3D> >* toAdd = gr->subGraphToPoint(nPoint);
      // toAdd->cloud->v_r = 1.0;
      // toAdd->cloud->v_g = 1.0;
      // toAdd->cloud->v_b = 0.0;
      // toAdd->cloud->v_radius = 0.5;
      // toAdd->saveToFile("toAdd.gr");
      // (*iter).second->cloud->v_r = 0.0;
      // (*iter).second->cloud->v_g = 1.0;
      // (*iter).second->cloud->v_b = 0.0;
      // (*iter).second->cloud->v_radius = 0.5;
      // (*iter).second->saveToFile("orig.gr");
      // exit(0);
      cost = cube->integralOverCloud(toAdd->cloud);
      if(nGraphsRejected%100 == 0){
        printf("Rejected %i paths\r", nGraphsRejected++);
        fflush(stdout);
        // sprintf(buff, "pathRejected%04i.gr", nGraphsRejected);
        // (*iter).second->saveToFile(buff);
      }
      paths.erase(iter);
      paths.insert(pair<float, Graph<Point3D, EdgeW<Point3D> >*>
                   (cost/(toAdd->cloud->points.size()), toAdd));
    } else {
      for(int i = 0; i < gr->cloud->points.size(); i++){
        microm[0] = gr->cloud->points[i]->coords[0];
        microm[1] = gr->cloud->points[i]->coords[1];
        microm[2] = gr->cloud->points[i]->coords[2];
        cube->micrometersToIndexes(microm, idxs);
        visited[idxs[2]][idxs[1]][idxs[0]] = true;
        visited[8][85][122] = false;
      }
      gr->cloud->v_r = 1.0;
      gr->cloud->v_g = 1.0;
      gr->cloud->v_b = 0.0;
      gr->cloud->v_radius = 0.05;
      sprintf(buff, "path%03i.gr",nGraphsAdded);
      nGraphsAdded++;
      gr->saveToFile(buff);
      printf("Adding the %i path with size %i\n", nGraphsAdded, gr->cloud->points.size());
      paths.erase(iter);
    }
  }//while

}//main
