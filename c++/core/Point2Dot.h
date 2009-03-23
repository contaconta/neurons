#ifndef POINT2DOT_H_
#define POINT2DOT_H_

#include "Point2Do.h"


class Point2Dot : public Point2Do
{
public:

  int type; //-1 or 1 according to the class

  Point2Dot();

  Point2Dot(float x, float y, float theta, int type);

  void draw();

  void draw(float width);

  bool load(istream &in);

  void save(ostream &out);

  virtual string className(){
    return "Point2Dot";
  }


};

#endif
