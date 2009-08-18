#include "CubeFactory.h"
#include "Cube.h"

Cube_P* CubeFactory::load(string volume_name){
  Cube_P* cube;
  cube = new Cube<uchar,ulong>(volume_name,false);
  if(cube->type == "uchar"){
    delete cube;
    cube = new Cube<uchar,ulong>(volume_name);
  } else
    if(cube->type == "float"){
      cube = new Cube<float,double>(volume_name);
    } else
      if(cube->type == "int"){
        cube = new Cube<int,long>(volume_name);
      } else {
        printf("CubeFactory::load no idea what %s is\n", cube->type.c_str());
        exit(0);
      }
  return cube;
}


