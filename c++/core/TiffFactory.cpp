#include "TiffFactory.h"

static int TiffFactory::getNumberOfLayers(string filename)
{
  int dircount = 0;
  uint16 bps;
  TIFF* tif = TIFFOpen(filename.c_str(), "r");
  if(!tif){
    printf("TiffFactory::getNumberOfLayers::Error getting the tiff image.\n");
    exit(0);
  } else {
    do{
        dircount++;
        TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bps);
      } while (TIFFReadDirectory(tif));
  }
  TIFFClose(tif);
  return dircount;
}

static int TiffFactory::getNumberOfBPS(string filename)
{
  uint16 bps;
  TIFF* tif = TIFFOpen(filename.c_str(), "r");
  if(!tif){
    printf("TiffFactory::getNumberOfBPS::Error getting the tiff image.\n");
    exit(0);
  } else {
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bps);
  }
  return (int)bps;
}


static void TiffFactory::getInfo(string filename)
{
  uint32 w, h, depth;
  uint16 bps, samplesPerPixel, tPhotoMetric, planarConfig;
  TIFF* tif = TIFFOpen(filename.c_str(), "r");
  int directories = getNumberOfLayers(filename);
  TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
  TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
  TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bps);
  TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &tPhotoMetric);
  TIFFGetField(tif, TIFFTAG_PLANARCONFIG, &planarConfig);
  TIFFGetField(tif, TIFFTAG_IMAGEDEPTH, &depth);
  TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);


  printf("TIFFImage: %s\n"
         " -> directories    = %i\n"
         " -> pwidth          = %i\n"
         " -> height         = %i\n"
         " -> bitesPerSample = %i\n"
         " -> photometric    = %i\n"
         " -> planarConfig   = %i\n"
         " -> imageDepth     = %i\n"
         " -> samplesPerPixel= %i\n"
         " -> bytesPerLine   = %i\n"
         , filename.c_str(),
         directories, w, h, bps,
         tPhotoMetric, planarConfig,
         depth, samplesPerPixel,
         (int)TIFFScanlineSize(tif)
         );
}

static VisibleE* TiffFactory::load(string filename){

  int directories = getNumberOfLayers(filename);
  uint16 bps, photometric;

  TIFF* tif = TIFFOpen(filename.c_str(), "r");

  TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bps);
  TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photometric);

  // if(directories == 1){
  if(0){
    printf("Interpreting as an image\n");
    return new Image<float>(filename, 0);
  }
  else {
    printf("Interpreting as a cube\n");
    if(photometric == 1){
      if(bps == 8){
        printf(" ->t he cube is char\n");
      } else if (bps == 16) {
        printf(" ->t he cube is int\n");
      } else if (bps == 32) {
        printf(" ->t he cube is float\n");
      }
      return CubeFactory::load(filename);
    } //bw
    else if (photometric >= 2) {
      printf("Interpreting as a colorcube\n");
      if(bps == 8){
        printf(" -> the cube is char\n");
      } else if (bps == 16) {
        printf(" ->t he cube is int\n");
      } else if (bps == 32) {
        printf(" ->t he cube is float\n");
      }
      // return new Cube_C(filename); //!!!! TIFFF NO COLOR!!!!
      return CubeFactory::load(filename);
    }//RGB or ColorMap
  }

  return new VisibleE();

}
