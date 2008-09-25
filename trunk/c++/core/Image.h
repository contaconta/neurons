/** Wrapper of IplImage to add several methods. Faster than learning to use opencv.

The coordinates of the image are as usual:

------->X
|
|
|
V
Y

*/





#ifndef IMAGE_H_
#define IMAGE_H_

#include "neseg.h"

#include "polynomial.h"
#include "VisibleE.h"
#include "utils.h"

using namespace std;


template< class T>
class Image : public VisibleE
{
public:

  IplImage* img;

  int width;
  int height;
  int widthStep;

  int nChannels;

  string directory;
  string name;
  string name_raw;

  bool texture_loaded;
  GLuint  texture;

  int fildes;

  T* pixels_origin;
  T** pixels;

  Image();

  ~Image();

  Image(string filename, bool subtract_mean = false);

//   void calculateIntegralImage(string filename);

  /** Calculates the dx derivative in x and ny derivative in y of the image*/
  Image<float>* calculate_derivative(int nx, int ny, double sigma,  string filename = "");

  /** Calculate the hermite coefficients for a given order.*/
  Image<float>* calculate_hermite(int dx, int dy,double sigma,  string filename = "");

  /** Convolves with an 'orthogonal gaussian mask'*/
  Image<float>* calculate_gaussian_orthogonal(int dx, int dy, double sigma, string filename = "");

  /** Convolves with a normalized gaussian mask (energy = 1)*/
  Image<float>* calculate_gaussian_normalized(int dx, int dy, double sigma, string filename = "");

  Image<float>* create_blank_image_float(string name);

  Image<T>*     create_blank_image(string name);

  void convolve_horizontally(vector<float>& mask, Image<float>* output);

  void convolve_vertically  (vector<float>& mask, Image<float>* output);

  void micrometersToIndexes(vector<float>& micrometers, vector< int >& indexes);

  void indexesToMicrometers(vector< int >& indexes, vector< float >& micrometers);

  /** Threshold the image.
      If the filename is defined, it creates a new image with that name and
         stores the thresholded image there.
      If upper = true, it is the values of the image greater than the threshold
          the ones put to the value.
      The toValue field is defined in case we want to change the number that
           thresholded pixels are put to.
   */
  void threshold(T threshold, string filename = "", T toValueLow = 0, T toValueUp = 255);

  /** Applies a mask to this image. The pixels are put to value. The selected pixels are: if(high_mask) > mean(mask), otherwhise.*/
  void applyMask(Image<float>* mask, T value, bool high_mask = true);

  void histogram(int nbins, vector<int>& n_points, vector<float>& range, bool ignoreLowerValue = false);

  double getMean();

  T at(int x, int y);

  T max();

  void put(int x, int y, T value);

  void put_all(T value);

  /** Saves the image as an image. The float values will be kept.*/
  void save(); // Needed to write into the HD the changes

  /** The image will be drawn between (0,0) and (width,height).*/
  void draw();
};

template<class T>
Image<T>::Image() : VisibleE()
{

  img = NULL;
  texture_loaded = false;
}

template<class T>
Image<T>::~Image()
{
  if(fildes != -1){
    munmap(pixels_origin, width*height*sizeof(T));
    close(fildes);
  }
  save();
  // cvReleaseImage(&img);
}

template<class T>
Image<T>::Image(string filename, bool subtract_mean) : VisibleE()
{

  texture_loaded = false;
  directory = "";
  name = "";
  img = cvLoadImage(filename.c_str(),0);
  if(img == NULL){
    printf("Error getting the image %s\n", filename.c_str());
    exit(0);
  }
  directory = filename.substr(0,filename.find_last_of("/\\")+1);
  name = filename.substr(filename.find_last_of("/\\")+1);
  name_raw = directory + name.substr(0,name.find_last_of(".")) + ".raw";

  width = img->width;
  height = img->height;
  widthStep = img->widthStep;
  nChannels = img->nChannels;

  double mean  = 0;
  if(subtract_mean)
    for(int y = 0; y < height; y++)
      for(int x = 0; x < width; x++)
        mean += ((uchar *)(img->imageData + y*img->widthStep))[x];

  mean = mean/(width*height);

  //Creates the mapping of the raw data.
  fildes = open64(name_raw.c_str(), O_RDWR);

  if(fildes == -1) //The file does not exist create it
    {
      printf("The file %s does not exist. Creating it.\n", name_raw.c_str());
      FILE* fp = fopen(name_raw.c_str(), "w");
      int line_length = width;
      //FIXME
      T buff[line_length];
      float sum = 0;
      CvScalar s;
      for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){
          if(subtract_mean){
            buff[x] = ((T)((uchar *)(img->imageData + y*img->widthStep))[x]-mean)/255;
          }
          else{
            buff[x] = (T)((uchar *)(img->imageData + y*img->widthStep))[x];
          }
        }
        fwrite(buff, sizeof(T), line_length, fp);
      }
      fclose(fp);
      fildes = open64(name_raw.c_str(), O_RDWR);
    }

  void* mapped_file = mmap64(0,
                             width*height*sizeof(T),
                             PROT_READ|PROT_WRITE, MAP_SHARED, fildes, 0);

  if(mapped_file == MAP_FAILED)
    {
      printf("Image. There is a bug here, volume not loaded. %s\n", 
             name_raw.c_str());
      exit(0);
    }

  pixels_origin = (T*) mapped_file;
  pixels = (T**)malloc(height*sizeof(T*));

  //Initializes the pointer structure to acces quickly to the voxels
  for(int j = 0; j < height; j++){
    pixels[j]=(T*)&pixels_origin[j*width];
  }

//   name = filename;
}

template<class T>
T Image<T>::at(int x, int y){
  return pixels[y][x];
}

template<class T>
void Image<T>::put(int x, int y, T value){
  pixels[y][x] = value;
}

template<class T>
void Image<T>::put_all(T value){
  for(int x = 0; x < width; x++)
    for(int y = 0; y < height; y++)
      pixels[y][x] = value;
}

template<class T>
void Image<T>::convolve_horizontally(vector<float>& mask, Image<float>* output)
{
  int mask_side = mask.size()/2;
  int mask_size = mask.size();

//   printf("Cube<T,U>::convolve_horizontally [");
  #ifdef WITH_OPENMP
  #pragma omp parallel for
  #endif
  for(int y = 0; y < height; y++){
    float result = 0;
    int q = 0;
    int x = 0;
    //Beginning of the line
    for(x = 0; x < mask_size; x++){
      result = 0;
      for(q = -mask_side; q <=mask_side; q++){
        if(x+q<0)
          result+=this->at(0,y)*mask[mask_side + q];
        else if (x+q>width)
          result+=this->at(width-1,y)*mask[mask_side + q];
        else
          result += this->at(x+q,y)*mask[mask_side + q];
      }
      output->put(x,y,result);
    }

    //Middle of the line
    for(x = mask_size; x <= width-mask_size-1; x++){
      result = 0;
      for(q = -mask_side; q <=mask_side; q++)
        result += this->at(x+q,y)*mask[mask_side + q];
      output->put(x,y,result);
    }
    //End of the line
    for(x = width-mask_size; x < width; x++){
      result = 0;
      for(q = -mask_side; q <=mask_side; q++){
        if(x+q >= width)
          result+=this->at(width-1,y)*mask[mask_side + q];
        else if (x+q<0)
          result+=this->at(0,y)*mask[mask_side + q];
        else
          result += this->at(x+q,y)*mask[mask_side + q];
      }
      output->put(x,y,result);
    }
  }
//   printf("]\n");
}

template<class T>
void Image<T>::convolve_vertically(vector<float>& mask, Image<float>* output)
{
  int mask_side = mask.size()/2;
  int mask_size = mask.size();

//   printf("Cube<T,U>::convolve_vertically [");
  #ifdef WITH_OPENMP
  #pragma omp parallel for
  #endif
  for(int x = 0; x < width; x++){
    float result = 0;
    int q = 0;
    int y = 0;
//     Beginning of the line
    for(y = 0; y < mask_size; y++){
      result = 0;
      for(q = -mask_side; q <=mask_side; q++){
        if(y+q<0)
          result+=this->at(x,0)*mask[mask_side + q];
        else if (y+q>height)
          result+=this->at(x,height-1)*mask[mask_side + q];
        else
          result += this->at(x,y+q)*mask[mask_side + q];
      }
      output->put(x,y,result);
    }
    //Middle of the line
    for(y = mask_size; y <= height-mask_size-1; y++){
      result = 0;
      for(q = -mask_side; q <=mask_side; q++)
        result += this->at(x,y+q)*mask[mask_side + q];
      output->put(x,y,result);
    }
    //End of the line
    for(y = height-mask_size; y < height; y++){
      result = 0;
      for(q = -mask_side; q <=mask_side; q++){
        if(y+q >= height)
          result+=this->at(x,height-1)*mask[mask_side + q];
        else if (y+q < 0)
          result+=this->at(x,0)*mask[mask_side + q];
        else
          result += this->at(x,y+q)*mask[mask_side + q];
      }
      output->put(x,y,result);
    }
  }
//   printf("]\n");
}

template<class T>
Image<float>* Image<T>::calculate_derivative(int dx, int dy,double sigma,  string filename)
{
  vector< float > m_x = Mask::gaussian_mask(dx,sigma,true);
  vector< float > m_y = Mask::gaussian_mask(dy,sigma,true);

  string tmp_name = directory + "tmp.jpg";

  Image<float>* tmp;
  ifstream inp;
  inp.open(tmp_name.c_str(), ifstream::in);
  if(inp.fail())
    tmp = create_blank_image_float(tmp_name);
  else
    tmp = new Image<float>(tmp_name);
  inp.close();

  Image<float>* result = create_blank_image_float(filename);

  this->convolve_horizontally(m_x,tmp);
  tmp->convolve_vertically(m_y, result);

  result->save();
  return result;
}

template<class T>
Image<float>* Image<T>::calculate_hermite(int dx, int dy,double sigma,  string filename)
{
  vector< float > m_x = Mask::hermitian_mask(dx,sigma,true);
  vector< float > m_y = Mask::hermitian_mask(dy,sigma,true);

  string tmp_name = directory + "tmp.jpg";

  Image<float>* tmp;
  ifstream inp;
  inp.open(tmp_name.c_str(), ifstream::in);
  if(inp.fail())
    tmp = create_blank_image_float(tmp_name);
  else
    tmp = new Image<float>(tmp_name);
  inp.close();

  Image<float>* result = create_blank_image_float(filename);

  this->convolve_horizontally(m_x,tmp);
  tmp->convolve_vertically(m_y, result);

  result->save();
  return result;
}

template<class T>
Image<float>* Image<T>::calculate_gaussian_orthogonal(int dx, int dy, double sigma, string filename)
{
  vector< float > m_x = Mask::gaussian_mask_orthogonal(dx,sigma,true);
  vector< float > m_y = Mask::gaussian_mask_orthogonal(dy,sigma,true);

  string tmp_name = directory + "tmp.jpg";

  Image<float>* tmp;
  ifstream inp;
  inp.open(tmp_name.c_str(), ifstream::in);
  if(inp.fail())
    tmp = create_blank_image_float(tmp_name);
  else
    tmp = new Image<float>(tmp_name);
  inp.close();

  Image<float>* result = create_blank_image_float(filename);

  this->convolve_horizontally(m_x,tmp);
  tmp->convolve_vertically(m_y, result);

  result->save();
  return result;

}


template<class T>
Image<float>* Image<T>::calculate_gaussian_normalized(int dx, int dy, double sigma, string filename)
{
  vector< float > m_x = Mask::gaussian_mask(dx,sigma,true);
  vector< float > m_y = Mask::gaussian_mask(dy,sigma,true);

  // The sqrt is needed because the 2D mask is computed as an outer product.
  double en = sqrt(Mask::energy2DGaussianMask(dx, dy, sigma));
  for(int i = 0; i < m_x.size(); i++)
    m_x[i] = m_x[i]/en;
  for(int i = 0; i < m_y.size(); i++)
    m_y[i] = m_y[i]/en;

  string tmp_name = directory + "tmp.jpg";

  Image<float>* tmp;
  ifstream inp;
  inp.open(tmp_name.c_str(), ifstream::in);
  if(inp.fail())
    tmp = create_blank_image_float(tmp_name);
  else
    tmp = new Image<float>(tmp_name);
  inp.close();

  Image<float>* result = create_blank_image_float(filename);

  this->convolve_horizontally(m_x,tmp);
  tmp->convolve_vertically(m_y, result);

  result->save();
  return result;

}



template<class T>
Image<float>* Image<T>::create_blank_image_float(string name)
{
//   printf("Name: %s %i %i\n", name.c_str(), width, height);
  IplImage* img = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 3);
  cvSaveImage(name.c_str(),img);
  Image<float>* imag = new Image<float>(name);
  return imag;
}


template<class T>
Image<T>* Image<T>::create_blank_image(string name)
{
//   printf("Name: %s %i %i\n", name.c_str(), width, height);
  IplImage* img = cvCreateImage(cvSize(width,height), IPL_DEPTH_8U, 3);
  cvSaveImage(name.c_str(),img);
  Image<T>* imag = new Image<T>(name);
  return imag;
}


template<class T>
T Image<T>::max()
{
  T max = -1e6;
  for(int x = 0; x < width; x++){
    for(int y = 0; y < height; y++){
      if(at(x,y)>max)
        max = at(x,y);
    }
  }
  return max;
}

template<class T>
void Image<T>::save()
{
  //Gets the max and the min of the image
  T max;
  T min;
  if(sizeof(T)==1){
    max = 0; min = 255;
  } else{
    max = -1e4; min = 1e4;
  }

  for(int x= 0; x < width; x++)
    for(int y = 0; y < height; y++){
      if(at(x,y)>max)
        max = at(x,y);
      if(at(x,y) < min)
        min = at(x,y);}

  for(int x= 0; x < width; x++)
    for(int y = 0; y < height; y++)
      for(int n = 0; n < nChannels; n++)
        img->imageData[x*nChannels + n + y*img->widthStep] =
          (uchar)255*(float(at(x,y) - min)/(max-min));

  string full_name = directory + name;
  cvSaveImage(full_name.c_str(),img);
//   printf("Saving in %s\n", full_name.c_str());
}

template<class T>
void Image<T>::micrometersToIndexes(vector<float>& micrometers, vector< int >& indexes)
{
  indexes[0] = (int)micrometers[0];
  indexes[1] = (int)(height -0.001 - micrometers[1]);
  if((micrometers.size()==3) && (indexes.size() == 3))
    indexes[2] = (int)0;
}

template<class T>
void Image<T>::indexesToMicrometers(vector< int >& indexes, vector< float >& micrometers)
{
  assert(indexes.size() == micrometers.size());
  //The index is in the middle of the pixel
  micrometers[0] = indexes[0] + 0.5;
  micrometers[1] = height - 1 - indexes[1] + 0.5;
  if(micrometers.size() == 3)
    micrometers[2] = 0;
}

template<class T>
void Image<T>::threshold
(T threshold, string filename,
 T toValueLow, T toValueUp
 )
{

  Image<T>* toThreshold = this;
  if(filename!=""){
    toThreshold = create_blank_image(filename);
  }

    for(int x = 0; x < width; x++)
      for(int y = 0; y < height; y++)
        if(at(x,y) < threshold)
          toThreshold->put(x,y,toValueLow);
        else
          toThreshold->put(x,y,toValueUp);

  toThreshold->save();
}



template<class T>
void Image<T>::draw()
{
  // Loads the texture
  if(!texture_loaded){
    T min_val, max_val;
    if(sizeof(T)==1){
      min_val = 255;
      max_val = 0;
    } else{
      min_val = 1e6;
      max_val = -1e6;
    }
    for(int x = 0; x < width; x++)
      for(int y = 0; y < height; y++){
        if(at(x,y) < min_val)
          min_val = at(x,y);
        if(at(x,y) > max_val)
          max_val = at(x,y);
      }
    T texels[width*height];

    for(int x = 0; x < width; x++)
      for(int y = 0; y < height; y++){
        texels[x + y*width] = (at(x,y) - min_val)/
          (max_val-min_val);
      }

    glGenTextures(1,&texture);
    glEnable(GL_TEXTURE);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D,  GL_TEXTURE_PRIORITY, 1.0);
    GLfloat border_color[4];
    for(int i = 0; i < 4; i++)
      border_color[i] = 1.0;
    glTexParameterfv(GL_TEXTURE, GL_TEXTURE_BORDER_COLOR, border_color);

    if(sizeof(T)==4)
      glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE,
                   width, height, 0, GL_LUMINANCE,
                   GL_FLOAT, texels);
    else
      glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE,
                   width, height, 0, GL_LUMINANCE,
                   GL_UNSIGNED_BYTE, texels);
    texture_loaded = true;
//     free texels;
  }

  glColor3f(1.0,1.0,1.0);

  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, texture);

  if(1){
    glBegin(GL_QUADS);
    glTexCoord2f(0.0,0.0);
    glVertex3f(0.0,height,0.0);
    glTexCoord2f(1.0,0.0);
    glVertex3f(width,height,0.0);
    glTexCoord2f(1.0,1.0);
    glVertex3f(width,0.0,0.0);
    glTexCoord2f(0.0,1.0);
    glVertex3f(0.0,0.0,0.0);
    glEnd();
  }

  if(0){
    float w_s = 0.2*float(width)/2;
    float h_s = 0.2*float(height)/2;

    glBegin(GL_QUADS);
    glTexCoord2f(0.0,0.0);
    glVertex3f(-w_s,h_s,0.0);
    glTexCoord2f(1.0,0.0);
    glVertex3f(w_s,h_s,0.0);
    glTexCoord2f(1.0,1.0);
    glVertex3f(w_s,-h_s,0.0);
    glTexCoord2f(0.0,1.0);
    glVertex3f(-w_s,-h_s,0.0);
    glEnd();
  }


  glBegin(GL_LINE_STRIP);
  glVertex3f(0.0,height,0.0);
  glVertex3f(width,height,0.0);
  glVertex3f(width,0.0,0.0);
  glVertex3f(0.0,0.0,0.0);
  glVertex3f(0.0,height,0.0);
  glEnd();

  glDisable(GL_TEXTURE_2D);
}

template< class T>
void Image<T>::histogram(int nbins, vector<int>& boxes, vector<float>& rangev, bool ignoreLowerValue)
{

  float max = -1e12;
  float min = 1e12;
  for(int y = 0; y < height; y++)
    for(int x = 0; x < width; x++)
      {
        if(pixels[y][x] > max)
          max = pixels[y][x];
        if(pixels[y][x] < min)
            min = pixels[y][x];
      }
  printf("Image<T>, the max is %f and the min is %f\n", max, min);

  // printf("Image<T>::histogram ["); fflush(stdout);

  float range = max - min;

  boxes.resize(nbins);
  for(int i = 0; i < nbins; i++)
    boxes[i] = 0;

  for(int y = 0; y < height; y++){
    for(int x = 0; x < width; x++){
      if(ignoreLowerValue && (this->at(x,y)==min))
        continue;
      boxes[(int)(floor(nbins*float((this->at(x,y)-min))/range))] += 1;
    }
    // printf("%02i\r", y*100/height ); fflush(stdout);
  }
  // printf("]\n");

  // if(!(*rangev==NULL)){
  rangev.resize(nbins);
  for(int i = 0; i < nbins; i++)
    rangev[i] = min + range*i/nbins;
  // }
}



template< class T>
double Image<T>::getMean()
{
  double result = 0;
  double line = 0;
  for(int y = 0; y < height; y++){
    line = 0;
    for(int x = 0; x < width; x++)
      line += at(x,y);
    result += line/(width*height);
  }
  return result;
}


template<class T>
void Image<T>::applyMask(Image<float>* mask, T value, bool high_mask)
{
  double m_m = mask->getMean();
  for(int y = 0; y < height; y++){
    for(int x = 0; x < width; x++){
      if(high_mask && (mask->at(x,y) > m_m))
        put(x,y,value);
      if(!high_mask && (mask->at(x,y) < m_m))
        put(x,y,value);
    }
  }
}


#endif
