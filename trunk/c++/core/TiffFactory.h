#ifndef FACTORYTIFF_H_
#define FACTORYTIFF_H_

#include "tiffio.h"
#include "Image.h"
#include "CubeFactory.h"
#include "Cube_C.h"

class TiffFactory
{
public:

  static int getNumberOfLayers(string filename);
  static int getNumberOfBPS(string filename);
  static VisibleE* load(string filename);
  static void getInfo(string filename);


};


#endif
