#include "Image.h"

using namespace std;


int main(int argc, char ** argv)
{

  Image< float >* img = new Image<float>();
  img->height = 512;
  img->width  = 256;

  printf("Created an image of 512x256\n");
  vector< int > indexes(3);
  vector< float > micrometers(3);

  indexes[0] = 0;
  indexes[1] = 0;
  indexes[2] = 0;

  img->indexesToMicrometers(indexes, micrometers);

  printf("Indexes [%i,%i,%i] correspond to micrometers [%f,%f,%f]\n",
         indexes[0], indexes[1], indexes[2], micrometers[0], micrometers[1], micrometers[2]);

  img->micrometersToIndexes(micrometers, indexes);

  printf("Micrometers [%f,%f,%f] correspond to indexes [%i,%i,%i]\n",
         micrometers[0], micrometers[1], micrometers[2], indexes[0], indexes[1], indexes[2]);


  indexes[0] = 255;
  indexes[1] = 511;
  indexes[2] = 0;

  img->indexesToMicrometers(indexes, micrometers);

  printf("Indexes [%i,%i,%i] correspond to micrometers [%f,%f,%f]\n",
         indexes[0], indexes[1], indexes[2], micrometers[0], micrometers[1], micrometers[2]);

  img->micrometersToIndexes(micrometers, indexes);

  printf("Micrometers [%f,%f,%f] correspond to indexes [%i,%i,%i]\n",
         micrometers[0], micrometers[1], micrometers[2], indexes[0], indexes[1], indexes[2]);


}
