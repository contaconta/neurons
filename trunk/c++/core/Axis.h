#ifndef AXIS_H_
#define AXIS_H_


#include "VisibleE.h"

class Axis : public VisibleE
{
public:

  Axis() : VisibleE() {}

  void draw();

 virtual string className(){
   return "Axis";
 }

};


#endif
