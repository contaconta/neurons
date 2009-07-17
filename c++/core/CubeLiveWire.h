#ifndef CUBELIVEWIRE_H_
#define CUBELIVEWIRE_H_

#include "Cube_P.h"
#include "Cube.h"
#include "Cloud.h"
#include "Graph.h"
#include "CubeDijkstra.h"
#include <pthread.h>
#include "float.h"

/** Implementation of the Dijkstra algorithm for a cube. The distance
    between pixels is determined by the class Distance Dijkstra

Each voxel is surrounded by 26 neighbors. The [x,y,z] coordinates of
the cube follow the scheme of neseg. For the neighbors we follow the
following schma:

View with Z in depth:

Layer 1:         Layer 2:         Layer 3:

0------>X
|  0 ----------3      ----------     1 ----------2
|  | 1 | 2 | 3 |    | 10| 11| 12|    | 18| 19| 20|
|  |--- --- ---|    |--- --- ---|    |--- --- ---|
|  | 4 | 5 | 6 |    | 13| X | 14|    | 21| 22| 23|
Y  |-----------|    |-----------|    |-----------|
   | 7 | 8 | 9 |    | 15| 16| 17|    | 24| 25| 26|
   4-----------7     -----------     5-----------6

*/

struct thread_data{
  int x; int y; int z;
};


class CubeLiveWire : public CubeDijkstra
{

public:

  // Structures for the algorithm
  int***   previous;
  bool***  visited;
  float*** distances;
  int*     previous_orig;
  bool*    visited_orig;
  float*   distance_orig;
  multimap<float, int> boundary; //Ordered by the distances

  // For multi-thread programming: we need the point of start
  int  xS, yS, zS;
  bool computingDistances;
  pthread_t          thread;

  CubeLiveWire(Cube_P* cube, DistanceDijkstra* distance) : CubeDijkstra(cube, distance)
  {
    computingDistances = false;
    int cubeLength = cube->cubeDepth*cube->cubeHeight*cube->cubeWidth;
    previous_orig  = (int*)  malloc(cubeLength*sizeof(float));
    visited_orig   = (bool*) malloc(cubeLength*sizeof(bool));
    distance_orig  = (float*)malloc(cubeLength*sizeof(float));

    //Initialization of the 3D pointers in the cube
    //Initialization of the first array?
    previous  = (int***)  malloc(cube->cubeDepth*sizeof(int** ));
    visited   = (bool***) malloc(cube->cubeDepth*sizeof(bool**));
    distances = (float***)malloc(cube->cubeDepth*sizeof(float**));
    for(int z = 0; z < cube->cubeDepth; z++){
      previous [z] = (int**)  malloc(cube->cubeHeight*sizeof(int*));
      visited  [z] = (bool**) malloc(cube->cubeHeight*sizeof(bool*));
      distances[z] = (float**)malloc(cube->cubeHeight*sizeof(float*));
      for(int j = 0; j < cube->cubeHeight; j++){
        previous[z][j]=(int*)  &previous_orig[(z*cube->cubeHeight + j)*cube->cubeWidth];
        visited[z][j] =(bool*) &visited_orig [(z*cube->cubeHeight + j)*cube->cubeWidth];
        distances[z][j]=(float*)&distance_orig[(z*cube->cubeHeight + j)*cube->cubeWidth];
      }
    }
  }

  //Computes the distances to all points in the cube starting in x0, y0, z0 using dijkstra
  void* computeDistances(int x0, int y0, int z0);


  Cloud<Point3D>* findShortestPath(int x0, int y0, int z0,
                                   int x1, int y1, int z1
                                   );

  Graph<Point3D, EdgeW<Point3D> >* findShortestPathG(int x0, int y0, int z0,
                                   int x1, int y1, int z1
                                   );

  Cube<float, double>* goThroughBorders(string cubeName);

  vector<vector< int > > findShortestPathIdx
  (int x0, int y0, int z0, int x1, int y1, int z1);

  vector< Cloud< Point3D >*> goThroughBordersCloud(int nClouds);

  float integralOverPath(vector< vector< int > >& path);

};

#endif
