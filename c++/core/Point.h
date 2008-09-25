#ifndef POINT_H_
#define POINT_H_

#include "neseg.h"

#include <vector>
#include <fstream>
#include <math.h>

#include "Visible.h"

using namespace std;

/** A point in 3D*/
class Point : public Visible
{
public:

  vector< float > coords;

  // Point(float x, float y, float z);

  virtual double distanceTo(Point* p)=0;

  virtual void draw()=0;

  virtual void draw(float width)=0;

};

#endif
