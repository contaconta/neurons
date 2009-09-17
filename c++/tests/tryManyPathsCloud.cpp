
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
  char buff[1024];
  // For the color
  const gsl_rng_type * T2;
  gsl_rng * r;
  gsl_rng_env_setup();
  T2 = gsl_rng_default;
  r = gsl_rng_alloc (T2);


  // Cube<uchar, ulong>* cube = new Cube<uchar, ulong>("/media/neurons/cut2/cut2.nfo");
  // Cloud_P* decimatedCloud = CloudFactory::load("/media/neurons/cut2/decimated.cl");

  Cube<float, double>* cube = new Cube<float, double>
    ("/media/neurons/steerableFilters3D/tmp/cut/cut.nfo");
  Cloud_P* decimatedCloud = CloudFactory::load
    ("/media/neurons/steerableFilters3D/tmp/cut/dense.cl");
  Cloud<Point3D>* seedPointsSelected = new Cloud<Point3D>();

  DistanceDijkstraColorNegatedEuclidean* djkc
    = new DistanceDijkstraColorNegatedEuclidean(cube);
  CubeLiveWire* cubeLiveWire = new CubeLiveWire(cube, djkc);;


  vector<int> origIdxs(3);
  origIdxs[0] = 151;
  origIdxs[1] = 152;
  origIdxs[2] = 9;
  if(fileExists("/media/neurons/steerableFilters3D/tmp/cut/parents.nfo") &&
     fileExists("/media/neurons/steerableFilters3D/tmp/cut/distances.nfo")){
    printf("Loading the distances .... ");fflush(stdout);
    cubeLiveWire->loadParents("/media/neurons/steerableFilters3D/tmp/cut/parents.nfo");
    cubeLiveWire->loadDistances("/media/neurons/steerableFilters3D/tmp/cut/distances.nfo");
  } else{
    printf("Computing the distances .... ");fflush(stdout);
    cubeLiveWire->computeDistances(origIdxs[0],origIdxs[1],origIdxs[2]);
    cubeLiveWire->saveParents("parents");
    cubeLiveWire->saveDistances("distances");
  }
  printf(" distances done \n");


  multimap<float, Graph<Point3D, EdgeW<Point3D> >*> paths;

  Cloud< Point3D>* cloud = new Cloud<Point3D>();
  cube->indexesToMicrometers(origIdxs,microm);
  cloud->addPoint(microm[0],microm[1],microm[2]);
  cloud->v_g = 1.0;
  cloud->v_radius = 2.0;
  cloud->saveToFile("/media/neurons/steerableFilters3D/tmp/cut/original.cl");

  float cost;

  printf("Finding the shortest paths from all points to the soma[");fflush(stdout);
  for(int i = 0; i < decimatedCloud->points.size(); i++){
    cube->micrometersToIndexes3(decimatedCloud->points[i]->coords[0],
                                decimatedCloud->points[i]->coords[1],
                                decimatedCloud->points[i]->coords[2],
                                idxs[0],idxs[1],idxs[2]);

    Graph<Point3D, EdgeW<Point3D> >* shortestPath =
      cubeLiveWire->findShortestPathG(origIdxs[0],origIdxs[1],origIdxs[2],
                                      idxs[0],idxs[1],idxs[2]
                                      );
    cost = shortestPath->cloud->points.size() -
      cube->integralOverCloud(shortestPath->cloud);
    paths.insert(pair<float, Graph<Point3D, EdgeW<Point3D> >*>
                 ( exp(cost/(shortestPath->cloud->points.size())), shortestPath));
    if(i%100 == 0) printf("%02f]\r",
                          float(i*100)/decimatedCloud->points.size());fflush(stdout);
  }
  printf("\n");


  // Saves the paths according to the minimum mean distance
  if(1){
    int nGraphSaved = 0;
    for(multimap< float, Graph<Point3D, EdgeW<Point3D> >*>::iterator
          iter = paths.begin();
        iter != paths.end();
        iter++){
      Graph<Point3D, EdgeW<Point3D> >* gr = (*iter).second;
      sprintf(buff, "/media/neurons/steerableFilters3D/tmp/cut/opath%04i.gr",
              nGraphSaved++);
      gr->v_r = gsl_rng_uniform(r);
      gr->v_g = gsl_rng_uniform(r);
      gr->v_b = gsl_rng_uniform(r);
      gr->v_radius = 0.3;
      gr->cloud->v_r = gsl_rng_uniform(r);
      gr->cloud->v_g = gsl_rng_uniform(r);
      gr->cloud->v_b = gsl_rng_uniform(r);
      gr->cloud->v_radius = 0.3;
      gr->saveToFile(buff);
      if(nGraphSaved%100 == 0)
        printf("Saving the %i path with size %i\n", nGraphSaved, gr->cloud->points.size());
    }
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




  int limitEnd = decimatedCloud->points.size()-1;
  // int limitEnd = 500 ;
  int nGraphsAdded = 0;
  int nGraphsRejected = 0;
  while(nGraphsAdded < limitEnd){
    iter = paths.begin();
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
      cost = toAdd->cloud->points.size() -
        cube->integralOverCloud(toAdd->cloud);
      if(nGraphsRejected%100 == 0)
        printf("Rejected %i paths\n", nGraphsRejected++);
      fflush(stdout);
      nGraphsRejected++;
      paths.erase(paths.begin());
      paths.insert(pair<float, Graph<Point3D, EdgeW<Point3D> >*>
                   (exp(cost/(toAdd->cloud->points.size())), toAdd));
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
        visited[origIdxs[2]][origIdxs[1]][origIdxs[0]] = false;
      }
      gr->v_r = gsl_rng_uniform(r);
      gr->v_g = gsl_rng_uniform(r);
      gr->v_b = gsl_rng_uniform(r);
      gr->v_radius = 0.3;
      gr->cloud->v_r = gsl_rng_uniform(r);
      gr->cloud->v_g = gsl_rng_uniform(r);
      gr->cloud->v_b = gsl_rng_uniform(r);
      gr->cloud->v_radius = 0.3;

      // cost = cube->integralOverCloud(gr->cloud)/gr->cloud->points.size();

      sprintf(buff, "/media/neurons/steerableFilters3D/tmp/cut/path%04i.gr",nGraphsAdded);
      nGraphsAdded++;
      gr->saveToFile(buff);
      if(nGraphsAdded%100 == 0)
        printf("Adding the %i path with size %i\n", nGraphsAdded, gr->cloud->points.size());
      paths.erase(paths.begin());
    }

  }//while
  seedPointsSelected->v_r = 1;
  seedPointsSelected->v_g = 0;
  seedPointsSelected->v_b = 1;
  seedPointsSelected->v_radius = 0.7;
  seedPointsSelected->saveToFile("/media/neurons/steerableFilters3D/tmp/cut/seedPointsSelected.cl");

  // Saves the visited points into a cube

  Cube<int, long>* visitedC =
    new Cube<int, long>(cube->cubeWidth, cube->cubeHeight, cube->cubeDepth,
                        cube->directory + "visited",
                        cube->voxelWidth, cube->voxelHeight, cube->voxelDepth);
  for(int x = 0; x < cube->cubeWidth; x++)
    for(int y = 0; y < cube->cubeHeight; y++)
      for(int z = 0; z < cube->cubeDepth; z++){
        visitedC->put(x,y,z, visited[z][y][x]);
      }


}//main
