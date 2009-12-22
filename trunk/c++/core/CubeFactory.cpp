#include "CubeFactory.h"
#include "Cube.h"
#include "Cube_C.h"
#include "Cube_T.h"

Cube_P* CubeFactory::load(string volume_name){

  string extension = getExtension(volume_name);

  Cube_P* cube;

  if(extension == "nfo" ||
     extension == "nfc"){
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
        } else
          if(cube->type == "color"){
            cube = new Cube_C(volume_name);
          } else {
            printf("CubeFactory::load no idea what %s is\n", cube->type.c_str());
            exit(0);
          }
  }
  if(extension == "cbt"){
    cube = new Cube_T(volume_name);
  }

  if ( (extension == "tiff") ||
       (extension == "TIFF") ||
       (extension == "tif")  ||
       (extension == "TIF")
       )
    {
      TIFF* tif = TIFFOpen(volume_name.c_str(), "r");
      uint16 samplesPerPixel;
      uint16 bitsPerSample;
      TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
      TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
      TIFFClose(tif);
      //bw image
      if(samplesPerPixel == 1){
        if(bitsPerSample == 16){
          cube =  new Cube<int,long>   (volume_name,0);
        } else {
          cube =  new Cube<uchar,ulong>(volume_name,0);
        }
      } else {
        cube = new Cube_C(volume_name);
      }
  }

  return cube;
}


