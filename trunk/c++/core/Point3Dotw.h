#ifndef POINT3DOTW_H_
#define POINT3DOTW_H_

#include "neseg.h"

#include <vector>
#include <fstream>

#include "Point3Dot.h"

using namespace std;

/** A point in 3D*/
class Point3Dotw : public Point3Dot
{
public:

  double weight;

  Point3Dotw() : Point3Dot(){
    weight = 0;
  }

  Point3Dotw(float x, float y, float z,  float theta=0, float phi = 0,
             int type = -1, double _weight = 0.0) :
    Point3Dot(x,y,z,theta,phi,type)
  {
    weight = _weight;
  }

  virtual string className(){
    return "Point3Dotw";
  }
};

#endif
