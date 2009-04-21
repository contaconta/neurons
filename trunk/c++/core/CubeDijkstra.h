#ifndef CUBEDIJKSTRA_H_
#define CUBEDIJKSTRA_H_

#include "Cube_P.h"
#include "Cube.h"
#include "Cloud.h"

// Mock-up class. Need to have it done with an abstract class and derivates
class DistanceDijkstra{
public:
  DistanceDijkstra(){}
  virtual float distance(int x0, int y0, int z0, int x1, int y1, int z1) = 0;
};

class DistanceDijkstraColor : public DistanceDijkstra {
public:
  Cube_P* cube;

  DistanceDijkstraColor(Cube_P* cube){
    if (cube->type == "uchar"){
      this->cube = dynamic_cast<Cube< uchar, ulong>* >(cube);
    } else if (cube->type == "float"){
      this->cube = dynamic_cast<Cube< float, double>* >(cube);
    }

  }
  float distance(int x0, int y0, int z0, int x1, int y1, int z1){
    return fabs(cube->at(x0,y0,z0)-cube->at(x1,y1,z1));
  }
}

/** Container for an integer with the parent and a float with the
    distance to the original point.*/
class PointDijkstra
{
public:
  int previous;
  float distance;
  PointDijkstra(int p, float d){
    previous = p; distance = d;
  }
};



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

class CubeDijkstra
{

public:

  Cube_P* cube;
  DistanceDijkstra* distance;

  /** Table that codes the relationship between the indexes previously
      and the [x,y,z] indexes in the cube. Initialized in the
      constructor*/
  int nbrToIdx[27][3];



  /* From x,y,z to idx.*/
  int toLinearIndex(int x, int y, int z, Cube_P* cube){
    return (z * cube->cubeHeight + y) * cube->cubeWidth + x;
  }

  /** From idx to x,y,z */
  void toCubeIndex(int idx, int& x, int& y, int& z, Cube_P* cube){
    z = idx / (cube->cubeHeight * cube->cubeWidth);
    y = (idx - z*(cube->cubeHeight * cube->cubeWidth))/cube->cubeWidth;
    x = idx - z*(cube->cubeHeight * cube->cubeWidth) - y*cube->cubeWidth;
  }


  CubeDijkstra(Cube_P* cube, DistanceDijkstra* distance)
  {
    this->cube = cube;
    this->distance = distance;
    // This hack sucks, but it is the simplest manner I can find
    int nbrToIdxLocal[27][3] = {
      { 0, 0, 0}, //0
      {-1,-1,-1}, //Layer 1
      { 0,-1,-1},
      { 1,-1,-1},
      {-1, 0,-1},
      { 0, 0,-1}, //5
      { 1, 0,-1},
      {-1, 1,-1},
      { 0, 1,-1},
      { 1, 1,-1},
      {-1,-1, 0}, // 10 - Layer 2
      { 0,-1, 0},
      { 1,-1, 0},
      {-1, 0, 0},
      { 1, 0, 0},
      {-1, 1, 0}, //15
      { 0, 1, 0},
      { 1, 1, 0},
      {-1,-1, 1}, // Layer 3
      { 0,-1, 1},
      { 1,-1, 1}, //20
      {-1, 0, 1},
      { 0, 0, 1},
      { 1, 0, 1},
      {-1, 1, 1},
      { 0, 1, 1}, //25
      { 1, 1, 1}
    };
    for(int i = 0; i < 27; i++)
      for(int j = 0; j < 3; j++){
        nbrToIdx[i][j] = nbrToIdxLocal[i][j];
      }


  }


  Cloud<Point3D>* findShortestPath(int x0, int y0, int z0,
                                   int x1, int y1, int z1);

};





#endif
