#ifndef POINT3D_H_
#define POINT3D_H_

#include "neseg.h"

#include <vector>
#include <fstream>

#include "Point.h"

using namespace std;

/** A point in 3D*/
class Point3D : public Point
{
public:

  vector< float > coords;

  Point3D();

  Point3D(float x, float y, float z=0);

  void draw();

  void draw(float width);

  bool load(istream &in);

  void save(ostream &out);

  double distanceTo(Point* p);

  static string className(){
    return "Point3D";
  }
};

#endif
