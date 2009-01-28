#ifndef POINT3DO_H_
#define POINT3DO_H_

#include "neseg.h"

#include <vector>
#include <fstream>

#include "Point.h"

using namespace std;

/** A point in 3D*/
class Point3Do : public Point
{
public:

  // vector< float > coords;

  float theta;

  float phi;


  Point3Do();

  Point3Do(float x, float y, float z,  float theta=0, float phi = 0);

  void draw();

  void draw(float width);

  bool load(istream &in);

  void save(ostream &out);

  double distanceTo(Point* p);

  static string className(){
    return "Point3Do";
  }
};

#endif
