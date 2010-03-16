#ifndef POINT2DOTW_H_
#define POINT2DOTW_H_

#include "Point2Dot.h"


class Point2Dotw : public Point2Dot
{
public:

  float w; //width (in micrometers) of the point

  Point2Dotw();

  Point2Dotw(float x, float y, float theta, int type, float w);

  void draw();

  void draw(float width);

  bool load(istream &in);

  void save(ostream &out);

  string className();

};

#endif
