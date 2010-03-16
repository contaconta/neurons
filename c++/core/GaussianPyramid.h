/** Create a pyramid of images

*/

#ifndef GAUSSIAN_PYR_H_
#define GAUSSIAN_PYR_H_

#include "VisibleE.h"
#include <sstream>

using namespace std;

class GaussianPyramid : public VisibleE
{
public:

  IplImage** imgPyr;

  int sizePyr;

  string directory;
  string name;

  GaussianPyramid(string filename, int a_sizePyr);

  ~GaussianPyramid();

  void save();

  virtual string className(){
    return "GaussianPyramid";
  }

 private:
  void calculate_pyramid();
};

GaussianPyramid::~GaussianPyramid()
{
  for (int i = 0; i < sizePyr; i++)
    cvReleaseImage(&imgPyr[i]);
}

GaussianPyramid::GaussianPyramid(string filename, int a_sizePyr) : VisibleE()
{
  sizePyr = a_sizePyr;
  directory = "";
  name = "";

  imgPyr = new IplImage*[sizePyr];
  imgPyr[0] = cvLoadImage(filename.c_str(),0);
  if(imgPyr[0] == NULL){
    cout << "Error getting the image " << filename << std::endl;
    exit(0);
  }
  directory = getDirectoryFromPath(filename);
  name = getNameFromPath(filename);

  //cout << "Pyramid 0 " << ": " << imgPyr[0]->width << " " << imgPyr[0]->height << endl;

  calculate_pyramid();
}

void GaussianPyramid::calculate_pyramid()
{
  bool ok = true;
  int width = imgPyr[0]->width;
  int height = imgPyr[0]->height;
  //imgPyr = new IplImage*[sizePyr];
  for (int i = 1; i < sizePyr; i++){
    if (width%2 != 0 || height%2 != 0)
      cout << "Warning : width%2 != 0 || height%2 != 0\n";
    //width = (int)( (float)imgPyr[0]->width/(float)((pow((float)2,i+1))) );
    //height = (int)( (float)imgPyr[0]->height/(float)((pow((float)2,i+1))) );

    width /= 2;
    height /= 2;
    cout << "Pyramid " << i << ": " << width << " " << height << endl;
    imgPyr[i] = cvCreateImage(cvSize(width, height),imgPyr[0]->depth, imgPyr[0]->nChannels);
  }

  // calculate gaussian pyramid
  for (int i = 0; i < sizePyr-1; i++)
    {
      cout << "Downsampling image " << i << std::endl;
      cvPyrDown(imgPyr[i], imgPyr[i+1]);
    }
}

void GaussianPyramid::save()
{
  for (int i = 0; i < sizePyr; i++)
    {
      std::stringstream out;
      out << i;

      string filename("Pyr_"+out.str());
      filename += name;
      cout << "Saving file " << filename << std::endl;
      cvSaveImage(filename.c_str(),imgPyr[i]);
    }
}

#endif //GAUSSIAN_PYR
