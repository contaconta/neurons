#ifndef INTEGRALIMAGE_H_
#define INTEGRALIMAGE_H_

#include "Image.h"

class IntegralImage : public Image<float>
{

public:
  Image<float>* orig;

  /** Filename should be the integral image alrady computed.*/
  IntegralImage( string filename);

  IntegralImage( int width, int height);

  IntegralImage( Image<float>* img);

  void computeIntegralImage(Image<float>* img);

  float integral(int x0, int y0, int x1, int y1);

};

IntegralImage::IntegralImage(string filename) : Image<float>(filename)
{
  printf("Should be creating an integral image\n");
}

IntegralImage::IntegralImage( int width, int height) : Image<float>(width, height)
{
}

IntegralImage::IntegralImage( Image<float>* img) : Image<float>(img->width, img->height)
{
  computeIntegralImage(img);
}


void IntegralImage::computeIntegralImage(Image<float>* img)
{
  orig = img;
  put_all(0);
  float accumulator = 0;
  //might not be the most efficient implementation, but it works
  for(int col = 0; col < width; col++){
    accumulator += orig->at(0,col);
    put(col,0 ,accumulator);
  }
  accumulator = at(0,0);
  for(int row = 1; row < height; row++){
    accumulator += at(0, row);
    put(0, row, accumulator);
  }
  for(int row = 1; row < height; row++)
    for(int col = 1; col < width; col++)
      put(col, row,
          at(col-1, row) + at(col, row-1)
          - at(col-1,row-1) + orig->at(col, row));
}

float IntegralImage::integral(int x0, int y0, int x1, int y1)
{
  return at(x1,y1) + at(x0,y0) - at(x0,y1) - at(x1, y0);
}

#endif
