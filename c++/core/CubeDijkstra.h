#ifndef CUBEDIJKSTRA_H_
#define CUBEDIJKSTRA_H_

#include "Cube_P.h"
#include "Cube.h"
#include "Cloud.h"
#include "CubeFactory.h"
#include <pthread.h>

// Mock-up class. Need to have it done with an abstract class and derivates
class DistanceDijkstra{
public:
  DistanceDijkstra(){}
  virtual float distance(int x0, int y0, int z0, int x1, int y1, int z1) = 0;
};

class DistanceDijkstraColor : public DistanceDijkstra {
public:
  Cube_P* cube;
  Cube<uchar, ulong>*  cubeUchar;
  Cube<float, double>* cubeFloat;
  int     cubeType; // 0 for uchar, 1 for float, 2 for int - lazy to typedef structs

  DistanceDijkstraColor(Cube_P* cube){
    printf("DistanceDijkstraColor created with the cube %s\n", cube->filenameParameters.c_str());
    if (cube->type == "uchar"){
      this->cubeUchar = dynamic_cast<Cube< uchar, ulong>* >(cube);
      cubeType = 0;
    } else if (cube->type == "float"){
      this->cubeFloat = dynamic_cast<Cube< float, double>* >(cube);
      cubeType = 1;
    }

  }
  float distance(int x0, int y0, int z0, int x1, int y1, int z1){
    float ds = sqrt( (double) (x0-x1)*(x0-x1) + (y0-y1)*(y0-y1) + (z0-z1)*(z0-z1));
    switch(cubeType){
    case 0:
      // return fabs(cubeUchar->at(x0,y0,z0)-cubeUchar->at(x1,y1,z1));
      return (float)cubeUchar->at(x0,y0,z0)*ds;
      break;
    case 1:
      return cubeFloat->at(x0,y0,z0)*ds;
      // return fabs(cubeFloat->at(x0,y0,z0)-cubeFloat->at(x1,y1,z1));
      break;
    };
    return 0.0;
  }
};

class DistanceDijkstraColorTruncated : public DistanceDijkstra {
public:
  Cube_P* cube;
  Cube<uchar, ulong>*  cubeUchar;
  Cube<float, double>* cubeFloat;
  int     cubeType; // 0 for uchar, 1 for float, 2 for int - lazy to typedef structs

  DistanceDijkstraColorTruncated(Cube_P* cube){
    printf("DistanceDijkstraColor created with the cube %s\n", cube->filenameParameters.c_str());
    this->cube = cube;
    if (cube->type == "uchar"){
      this->cubeUchar = dynamic_cast<Cube< uchar, ulong>* >(cube);
      cubeType = 0;
    } else if (cube->type == "float"){
      this->cubeFloat = dynamic_cast<Cube< float, double>* >(cube);
      cubeType = 1;
    }

  }
  float distance(int x0, int y0, int z0, int x1, int y1, int z1){
    //Check if both points are on the border of the cube
    if( ((x0==0)&&(x1==0)) ||
        ((y0==0)&&(y1==0)) ||
        ((z0==0)&&(z1==0)) ||
        ((x0==cube->cubeWidth-1)&&(x1==cube->cubeWidth-1)) ||
        ((y0==cube->cubeHeight-1)&&(y1==cube->cubeHeight-1)) ||
        ((z0==cube->cubeDepth-1)&&(z1==cube->cubeDepth-1)) )
      return 50.0;

    float module = sqrt((double) (x0-x1)*(x0-x1) + (y0-y1)*(y0-y1) + (z0-z1)*(z0-z1));

    switch(cubeType){
    case 0:
      // return fabs(cubeUchar->at(x0,y0,z0)-cubeUchar->at(x1,y1,z1));
      return (float)cubeUchar->at(x0,y0,z0)*module;
      break;
    case 1:
      return cubeFloat->at(x0,y0,z0)*module;
      // return fabs(cubeFloat->at(x0,y0,z0)-cubeFloat->at(x1,y1,z1));
      break;
    };
    return 0.0;
  }
};



class DistanceDijkstraColorInverse : public DistanceDijkstra {
public:
  Cube_P* cube;
  Cube<uchar, ulong>*  cubeUchar;
  Cube<float, double>* cubeFloat;
  int     cubeType; // 0 for uchar, 1 for float, 2 for int - lazy to typedef structs

  DistanceDijkstraColorInverse(Cube_P* cube){
    if (cube->type == "uchar"){
      this->cubeUchar = dynamic_cast<Cube< uchar, ulong>* >(cube);
      cubeType = 0;
    } else if (cube->type == "float"){
      this->cubeFloat = dynamic_cast<Cube< float, double>* >(cube);
      cubeType = 1;
    }

  }
  float distance(int x0, int y0, int z0, int x1, int y1, int z1){
    switch(cubeType){
    case 0:
      // return fabs(cubeUchar->at(x0,y0,z0)-cubeUchar->at(x1,y1,z1));
      return 1.0/cubeUchar->at(x1,y1,z1);
      break;
    case 1:
      return 1.0/cubeFloat->at(x1,y1,z1);
      // return fabs(cubeFloat->at(x0,y0,z0)-cubeFloat->at(x1,y1,z1));
      break;
    };
    return 0.0;
  }
};


class DistanceDijkstraColorNegated : public DistanceDijkstra {
public:
  Cube_P* cube;
  Cube<uchar, ulong>*  cubeUchar;
  Cube<float, double>* cubeFloat;
  int     cubeType; // 0 for uchar, 1 for float, 2 for int - lazy to typedef structs

  DistanceDijkstraColorNegated(Cube_P* cube){
    if (cube->type == "uchar"){
      this->cubeUchar = dynamic_cast<Cube< uchar, ulong>* >(cube);
      cubeType = 0;
    } else if (cube->type == "float"){
      this->cubeFloat = dynamic_cast<Cube< float, double>* >(cube);
      cubeType = 1;
    }

  }
  float distance(int x0, int y0, int z0, int x1, int y1, int z1){
    switch(cubeType){
    case 0:
      return 1.0-(float)(cubeUchar->at(x1,y1,z1))/255;
      break;
    case 1:
      return 1.0-cubeFloat->at(x1,y1,z1);
      break;
    };
    return 0.0;
  }
};

class DistanceDijkstraColorNegatedEuclidean : public DistanceDijkstra {
public:
  Cube_P* cube;
  Cube<uchar, ulong>*  cubeUchar;
  Cube<float, double>* cubeFloat;
  int     cubeType; // 0 for uchar, 1 for float, 2 for int - lazy to typedef structs

  DistanceDijkstraColorNegatedEuclidean(Cube_P* cube){
    if (cube->type == "uchar"){
      this->cubeUchar = dynamic_cast<Cube< uchar, ulong>* >(cube);
      cubeType = 0;
    } else if (cube->type == "float"){
      this->cubeFloat = dynamic_cast<Cube< float, double>* >(cube);
      cubeType = 1;
    }

  }
  float distance(int x0, int y0, int z0, int x1, int y1, int z1){
    double dist;
    switch(cubeType){
    case 0:
      return 1.0-(float)(cubeUchar->at(x1,y1,z1))/255;
      break;
    case 1:
      dist = sqrt((double) (x0-x1)*(x0-x1) + (y0-y1)*(y0-y1) + (z0-z1)*(z0-z1));
      return 1.0-cubeFloat->at(x1,y1,z1)/dist;
      break;
    };
    return 0.0;
  }
};


class DistanceDijkstraColorNegatedEuclideanAnisotropic : public DistanceDijkstra {
public:
  Cube_P* cube;
  Cube<uchar, ulong>*  cubeUchar;
  Cube<float, double>* cubeFloat;
  int     cubeType; // 0 for uchar, 1 for float, 2 for int - lazy to typedef structs
  double ratioY, ratioZ;

  DistanceDijkstraColorNegatedEuclideanAnisotropic(Cube_P* cube){
    if (cube->type == "uchar"){
      this->cubeUchar = dynamic_cast<Cube< uchar, ulong>* >(cube);
      cubeType = 0;
    } else if (cube->type == "float"){
      this->cubeFloat = dynamic_cast<Cube< float, double>* >(cube);
      cubeType = 1;
    }
    ratioZ = cube->voxelDepth/cube->voxelWidth;
    ratioY = cube->voxelHeight/cube->voxelWidth;
  }
  float distance(int x0, int y0, int z0, int x1, int y1, int z1){
    double dist;
    switch(cubeType){
    case 0:
      return 1.0-(float)(cubeUchar->at(x1,y1,z1))/255;
      break;
    case 1:
      dist = sqrt((double) (x0-x1)*(x0-x1) + ratioY*(y0-y1)*(y0-y1) + ratioZ*(z0-z1)*(z0-z1));
      return 1.0-cubeFloat->at(x1,y1,z1)/dist;
      break;
    };
    return 0.0;
  }
};



class DistanceDijkstraColorAngle : public DistanceDijkstra{
public:

  Cube<float, double>* c_measure;
  Cube<float, double>* c_theta;
  Cube<float, double>* c_phi;

  DistanceDijkstraColorAngle
  (Cube<float, double>* _measure, Cube<float, double>* _theta,
   Cube<float, double>* _phi){
    c_measure = _measure;
    c_theta   = _theta;
    c_phi     = _phi;
  }


  float distance(int x0, int y0, int z0, int x1, int y1, int z1){
    float theta0 = c_theta->at(x0,y0,z0);
    float theta1 = c_theta->at(x1,y1,z1);
    float phi0   = c_phi->at(x0,y0,z0);
    float phi1   = c_phi->at(x1,y1,z1);
    float measure0 = c_measure->at(x0,y0,z0);
    float measure1 = c_measure->at(x1,y1,z1);

    float dist = cos(theta0)*sin(theta0)*cos(theta1)*sin(theta1) + //x component
      sin(theta0)*sin(theta0)*sin (theta1)*sin(theta1) + //y component
      cos(phi0)*cos(phi1);  // z componenet
    dist = fabs(dist)/measure1; //to elliminate anti-directions
    return dist;
  }
};


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

  Cube<int, long>* previous_idx;
  int x0,y0,z0;

  bool pathFound;

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
    pathFound = false;
    // This hack sucks, but it is the simplest manner I can find
    initializenbrToIdxLocal();
  }

  CubeDijkstra(Cube_P* cube, DistanceDijkstra* distance, string cubePrevious)
  {
    this->cube = cube;
    this->distance = distance;
    pathFound = false;
    // This hack sucks, but it is the simplest manner I can find
    initializenbrToIdxLocal();
    if(fileExists(cubePrevious)){
      previous_idx = (Cube<int, long>*)CubeFactory::load(cubePrevious);
    }else {
      previous_idx = new Cube<int, long>(cube->cubeWidth, cube->cubeHeight, cube->cubeDepth,
                                         getNameFromPathWithoutExtension(cubePrevious),
                                         cube->voxelWidth, cube->voxelHeight,
                                         cube->voxelDepth);
    }
  }

  void initializenbrToIdxLocal(){
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
                                   int x1, int y1, int z1,
                                   Cloud<Point3D>& boundary,
                                   pthread_mutex_t& mutex);

  void initializeCubePrevious(int x0, int y0, int z0);

  Cloud<Point3D>* traceBack(int x1, int y1, int z1);

};

#endif
