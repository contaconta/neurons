/** Class Visible
 *  Simple class that defines a visible object of the neseg library.
 *
 *  German Gonzalez
 *  20080624
 */



#ifndef VISIBLE_H_
#define VISIBLE_H_

// #include "neseg.h"

#include "Object.h"
#include <sstream>
#include <iostream>

using namespace std;

class Visible : public Object
{
public:
  virtual void draw()=0;
};
#endif

