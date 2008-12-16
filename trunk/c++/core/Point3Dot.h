#ifndef POINT3DOT_H_
#define POINT3DOT_H_

#include "neseg.h"

#include <vector>
#include <fstream>

#include "Point3Do.h"

using namespace std;

/** A point in 3D*/
class Point3Dot : public Point3Do
{
public:

  int type; //-1 or 1 according to the class

  Point3Dot();

  Point3Dot(float x, float y, float z,  float theta=0, float phi = 0, int type = -1);

  void draw();

  void draw(float width);

  bool load(istream &in);

  void save(ostream &out);

  double distanceTo(Point* p);

  static string className(){
    return "Point3Dot";
  }
};

#endif
