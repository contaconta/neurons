
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
#include "CloudFactory.h"
#include <gsl/gsl_rng.h>
#include <map>

using namespace std;

int main(int argc, char **argv) {

  vector<int>   idxs(3);
  vector<float> microm(3);

  // Cube<uchar, ulong>* cube = new Cube<uchar, ulong>("/media/neurons/cut2/cut2.nfo");
  // Cloud_P* decimatedCloud = CloudFactory::load("/media/neurons/cut2/decimated.cl");

  Cube<float, double>* cube = new Cube<float, double>
    ("/media/neurons/SteerableFilters3D/tmp/cut/cut.nfo");
  Cloud_P* decimatedCloud = CloudFactory::load
    ("/media/neurons/SteerableFilters3D/tmp/cut/cut.cl");
  Cloud<Point3D>* seedPointsSelected = new Cloud<Point3D>
    ();

  DistanceDijkstraColorNegated* djkc = new DistanceDijkstraColorNegated(cube);
  CubeLiveWire*      cubeLiveWire = new CubeLiveWire(cube, djkc);;

  printf("Computing the distances .... \n");
  // cubeLiveWire->computeDistances(122, 85, 8);
  cubeLiveWire->computeDistances(151,152,9);

  printf("Getting the list of the paths to all possible points [");
  fflush(stdout);

  multimap<float, Graph<Point3D, EdgeW<Point3D> >*> paths;

  // Graph<Point3D, EdgeW<Point3D> >* shortestPath =
    // cubeLiveWire->findShortestPathG(122,85,8,0,0,0);

  // shortestPath->saveToFile("out.gr");

  Cloud< Point3D>* cloud = new Cloud<Point3D>();
  idxs[0] = 151;
  idxs[1] = 152;
  idxs[2] = 9;
  cube->indexesToMicrometers(idxs,microm);
  cloud->addPoint(microm[0],microm[1],microm[2]);
  cloud->saveToFile("original.cl");

  float cost;

  for(int i = 0; i < decimatedCloud->points.size(); i++){
    microm[0] = decimatedCloud->points[i]->coords[0];
    microm[1] = decimatedCloud->points[i]->coords[1];
    microm[2] = decimatedCloud->points[i]->coords[2];
    cube->micrometersToIndexes(microm, idxs);
    Graph<Point3D, EdgeW<Point3D> >* shortestPath =
      cubeLiveWire->findShortestPathG(151,152,9,idxs[0],idxs[1],idxs[2]);
    cost = cube->integralOverCloud(shortestPath->cloud);
    paths.insert(pair<float, Graph<Point3D, EdgeW<Point3D> >*>
                 (cost/(shortestPath->cloud->points.size()), shortestPath));
  }

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

  // For the color
  const gsl_rng_type * T2;
  gsl_rng * r;
  gsl_rng_env_setup();
  T2 = gsl_rng_default;
  r = gsl_rng_alloc (T2);

  int limitEnd = decimatedCloud->points.size()-1;
  // int limitEnd = 200;
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
        if( (i==0) ||
            ((i == gr->cloud->points.size()-1)&&(i>0))
          )
          seedPointsSelected->points.push_back
            (new Point3D(microm[0],microm[1],microm[2]));
        cube->micrometersToIndexes(microm, idxs);
        visited[idxs[2]][idxs[1]][idxs[0]] = true;
        visited[9][152][151] = false;
      }
      gr->cloud->v_r = gsl_rng_uniform(r);
      gr->cloud->v_g = gsl_rng_uniform(r);
      gr->cloud->v_b = gsl_rng_uniform(r);
      // gr->cloud->v_r = cost;
      // gr->cloud->v_g = 0;
      // gr->cloud->v_b = 0;
      // cost = cube->integralOverCloud(gr->cloud)/gr->cloud->points.size();
      gr->cloud->v_radius = 0.5;
      sprintf(buff, "path%03i.gr",nGraphsAdded);
      nGraphsAdded++;
      gr->saveToFile(buff);
      printf("Adding the %i path with size %i\n", nGraphsAdded, gr->cloud->points.size());
      paths.erase(iter);
    }
  }//while
  seedPointsSelected->v_r = 1;
  seedPointsSelected->v_g = 1;
  seedPointsSelected->v_b = 0;
  seedPointsSelected->v_radius = 0.7;
  seedPointsSelected->saveToFile("/media/neurons/SteerableFilters3D/tmp/cut/seedPointsSelected.cl");
}//main
