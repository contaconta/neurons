/** Simple class to load cubes.*/


#ifndef CUBE_FACTORY_H_
#define CUBE_FACTORY_H_

#include "Cube.h"

class CubeFactory
{
public:

  static Cube_P* load(string volume_name){
    Cube_P* cube;
    cube = new Cube<uchar,ulong>(volume_name,false);
    if(cube->type == "uchar"){
      delete cube;
      cube = new Cube<uchar,ulong>(volume_name);
    }
    if(cube->type == "float"){
      cube = new Cube<float,double>(volume_name);
    }
    return cube;
  }

};



#endif
