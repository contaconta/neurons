#ifndef POINT3DT_H_
#define POINT3DT_H_

#include "neseg.h"

#include <vector>
#include <fstream>

#include "Point.h"

using namespace std;

/** A point in 3D*/
class Point3Dt : public Point
{
public:

  // vector< float > coords;

  int type;

  Point3Dt();

  Point3Dt(float x, float y, float z=0, int type = 0);

  void draw();

  void draw(float width);

  bool load(istream &in);

  void save(ostream &out);

  double distanceTo(Point* p);

  static string className(){
    return "Point3Dt";
  }
};

#endif
