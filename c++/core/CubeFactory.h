/** Simple class to load cubes.*/


#ifndef CUBE_FACTORY_H_
#define CUBE_FACTORY_H_

#include "Cube_P.h"
// class Cube_P;

class CubeFactory
{
public:

  static Cube_P* load(string volume_name);
};



#endif
