#ifndef POINT2DO_H_
#define POINT2DO_H_


#include "Point.h"

class Point2Do : public Point
{
public:

  //In radians!
  float theta;

  Point2Do();

  Point2Do(float x, float y, float theta);

  void draw();

  void draw(float width);

  bool load(istream &in);

  void save(ostream &out);

  double distanceTo(Point* p);

  virtual string className(){
    return "Point2Do";
  }

};


#endif
