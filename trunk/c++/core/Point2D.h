#ifndef POINT2D_H_
#define POINT2D_H_

#include "neseg.h"

#include <vector>
#include <fstream>

#include "Point.h"

using namespace std;

/** A point in 2D*/
class Point2D : public Point
{
public:

  // vector< float > coords;

  Point2D();

  Point2D(float x, float y);

  void draw();

  void draw(float width);

  bool load(istream &in);

  void save(ostream &out);

  double distanceTo(Point* p);

  static string className(){
    return "Point2D";
  }

};

#endif
