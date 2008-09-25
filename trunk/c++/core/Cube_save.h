/** Cube.cpp
 * defines a cube of voxels. Each voxel would be represented as an array of unsigned bytes. It will be needed to improve it at some point.
The coordinate system would be as in openCV (i.e. top-left). The z direction goes towards the screen. The integral volume that can also
be loaded will share the same coordinate system as the others

Cube/texture coordinates    Cube vertex        World Coordinates
  Z t                     1-------------2      Y
 /                       /|            /|      |  Z
/                       / |           / |      | /
0------ X r            0-------------3  |      |/
|                      |  |          |  |      O-----X
|                      |  5--------- |- 6
|                      | /           | /       Centered in the center of the cube
Y s                    |/            |/
                       4-------------7



*/


#ifndef CUBE_H_
#define CUBE_H_

// #define ulong unsigned long long

#include <GL/glew.h>
#include <GL/glut.h>
#include <string>
#include "cv.h"
#include "highgui.h"
// #include "../ascParser/Neuron.h"
// class Neuron;
// class NeuronPoint;

#include <math.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <fcntl.h>
#include <iostream>
#include <fstream>
#include <stdarg.h>
#include <vector>
#include <algorithm>
#include <map>
#include <set>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>


#define CUBE_MAX_X 10000
#define CUBE_MAX_Y 10000
#define CUBE_MAX_Z 1000

using std::string;
using std::vector;
using namespace std;


template <class T, class U>
class Cube
{

public:
  // Dimension of the voxels in micrometers
  float voxelWidth;
  float voxelHeight;
  float voxelDepth;

  // Dimensions of the cube in voxels. Defined as long to allow for big cubes
  long cubeWidth;
  long cubeHeight;
  long cubeDepth;

  //Dimensions of the supercube it bellongs to (if it is not a subcube, then it will be the same as cubeWidth....)
  long parentCubeWidth;
  long parentCubeHeight;
  long parentCubeDepth;

  //Offset where to draw it. This will be to a global cube. In the case of a subcube, there is
  // an extra offset for the rowOffset and colOffset
  float x_offset;
  float y_offset;
  float z_offset;

  //If it is a subcube, the row and column number that they represent
  int rowOffset;
  int colOffset;

  //More variables, to know the last time it was drawn between two points
  int x0_old;
  int y0_old;
  int z0_old;
  int x1_old;
  int y1_old;
  int z1_old;
  float threshold_old;

  //Keeps pointers to the ordered (for fast indexing). it goes as voxels[z][y][x]
  T*** voxels;
  U*** voxels_integral;

  //Pointer to the volume data
  T* voxels_origin;
  U* voxels_integral_origin;


  unsigned int wholeTexture;
  unsigned int wholeTextureTrue;
  GLuint wholeTextureDepth;

  int nColToDraw;
  int nRowToDraw;

  int fildes;

  string filenameVoxelData;

  Cube(string filenameParams, string filenameVoxelData);
  Cube(string filenameParams, string filenameVoxelData, string filenameIntegralData);
  Cube(string dirName);
  Cube();
  ~Cube();

  void load_parameters(string filenameParams);
  void load_volume_data(string filenameVoxelData);
  void load_integral_volume(string filename);
  void create_volume_file(string filename);
  void create_integral_cube(string filename);
  void create_cube_from_kevin_images(
                                     string directory, string format, int layer_init, int layer_end,
            float voxelWidth, float voxelHeight, float voxelDepth);
  void create_cube_from_directory(string directory, string format, int layer_init, int layer_end,
                                     float voxelWidth, float voxelHeight, float voxelDepth);
  void create_cube_from_directory_matrix
  (
   string directory, string format,
   int row_begin, int row_end,
   int col_begin, int col_end,
   int layer_begin, int layer_end,
   float voxelWidth, float voxelHeight, float voxelDepth
   );
  void save_as_image_stack(string filename = "");
  void createMIPImage(string filename = "");
  void micrometersToIndexes(vector<float>& micrometers, vector< int >& indexes);
  void indexesToMicrometers(vector< int >& indexes, vector< float >& micrometers);
  void print_statistics(string filename = "");
  void histogram(string filename = "");
  void histogram_ignoring_zeros(string filename = "");
  void subsampleMean(string dir_name);
  void subsampleMinimum();
  void apply_mask(string mask_nfo, string mask_vl, string output_nfo, string output_vl);
  double integral_between(int x0, int y0, int z0, int x1, int y1, int z1);

  /** Prints the parameters*/
  void print();


  T at(int x, int y, int z);
  T at2(int x, int y, int z);
  void put(int x, int y, int z, T value);
  void put2(int x, int y, int z, T value);
  ulong integral_volume_at(int x, int y, int z);

  //Representation functions in Cube_draw.cpp
  void load_whole_texture();
  void draw_layers_parallel();                               //Draws the cube as a set of overlayed layers
  void draw(float rotx, float roty, float nPlanes = 200.0, int min_max = 0);  //Draws the cube following the 3D texture approach
  void draw_whole(float rotx, float roty, float nPlanes, int min_max);
  void draw(int x0, int y0, int z0, int x1, int y1, int z1, float rotx, float roty, float nPlanes, int min_max, float threshold = -1e6);
  void load_texture_brick(int row, int col);                 //Col and row start at 0
  void load_thresholded_texture_brick(int row, int col, float threshold);
  void load_thresholded_texture_brick_float(int row, int col, float threshold);
  void load_thresholded_maxmin_texture_brick_float(int row, int col, float threshold_low = -1e6, float threshold_high = 1e6);
  void draw_layer_tile_XY(float depth, int color = 0); //Depth starts at 0 - N-1
  void draw_layer_tile_XZ(float depth); //Depth starts at 0 - N-1
  void draw_layer_tile_YZ(float depth); //Depth starts at 0 - N-1
  void draw_orientation_grid(bool include_split = true);
  void render_string(const char* format, ...);


  //Math stuff in Cube_math.cpp
  // Inverts a matrix in the opengl format. Returns also opengl format.
  GLfloat* invert_matrix(GLfloat* a);
  GLfloat* matrix_vector_product(GLfloat* matrix, GLfloat* vector);
  GLfloat* get_matrix_angles(GLfloat* m);
  GLfloat* create_vector(float x, float y, float z, float w);
  vector< T > sort_values();
  int find_value_in_ordered_vector(vector< T >& vector, T value);
  vector< vector< int > > decimate(float threshold, int window_xy = 8, int window_z = 3, string filemane = "", bool save_boosting_response = false);
  vector< vector< int > > decimate_log(float threshold, int window_xy = 8, int window_z = 3, string filemane = "", bool save_boosting_response = false);

  //Operations over the cube
  int gaussian_mask(float sigma, vector< float >& Mask0, vector< float >& Mask1); //Taken from libvision
  void convolve_horizontally(vector< float >& Mask, Cube<float,double>* output);
  void convolve_vertically(vector< float >& Mask, Cube<float,double>* output);
  void convolve_depth(vector< float >& Mask, Cube<float,double>* output);
  void gradient_x(float sigma, Cube< float,double >* output, Cube<float,double>* tmp);
  void gradient_y(float sigma, Cube<float,double>* output, Cube<float,double>* tmp);
  void gradient_z(float sigma, Cube<float,double>* output, Cube<float,double>* tmp);
  void calculate_eigen_values(string directory_name);
  void calculate_eigen_values(float sigma, string directory_name);
  void calculate_eigen_vector_lower_eigenvalue(string directory_name);
  void calculate_f_measure(string directory_name);
  void norm_cube(Cube<float,double>* c1, Cube<float,double>* c2, Cube<float,double>* output);
  void norm_cube(string volume_nfo, string volume_1, string volume_2, string volume_3, string volume_output);
  void get_ROC_curve(string volume_nfo, string volume_positive, string volume_negative, string output_file = "ROC.txt", int nPoints = 100);
//   void create_cube_from_neuron(string neuron_name, string volume_nfo, string volume_vl);

  //Boosting related code. In Cube_boosting.cpp. OBSOLETE
  void apply_boosting_to_brick(string name_classifier, int row, int col, Cube* output);
  GLuint classify_voxel(vector< float*>& global_classifier, int x, int y, int z);
  vector< float* > load_boosting_haar_classifier(string name_classifier);

  //Initialization and boosting stuff
};


//################################################
// PUT HERE FOR THE TEMPLATE INSTANTIALIZATION
//################################################



//######### CUBE MAIN #####################


template <class T, class U>
Cube<T,U>::Cube()
{
  filenameVoxelData = "";
}
template <class T, class U>
Cube<T,U>::~Cube()
{
  munmap(voxels_origin, cubeWidth*cubeHeight*cubeDepth*sizeof(T));
  close(fildes);
}
template <class T, class U>
Cube<T,U>::Cube(string filenameParams, string _filenameVoxelData)
{
  filenameVoxelData = "";

  if (filenameParams!= "")
    load_parameters(filenameParams);

  if (_filenameVoxelData!= "")
    load_volume_data(filenameVoxelData);

  nColToDraw = -1;
  nRowToDraw = -1;
  glGenTextures(1, &wholeTexture);
  glGenTextures(1, &wholeTextureTrue);


}

template <class T, class U>
Cube<T,U>::Cube(string filenameParams, string filenameVoxelData, string filenameIntegralData)
{
  if (filenameParams!= "")
    load_parameters(filenameParams);

  if (filenameVoxelData!= "")
    load_volume_data(filenameVoxelData);

  if(filenameIntegralData != "")
    load_integral_volume(filenameIntegralData);

  nColToDraw = -1;
  nRowToDraw = -1;
  glGenTextures(1, &wholeTexture);
  glGenTextures(1, &wholeTextureTrue);

}

template <class T, class U>
Cube<T,U>::Cube(string dirName)
{
//   string filenameParams = dirName + "/volume.nfo";
//   string filenameVoxelData = dirName + "/volume.vl";
//   string filenameIntegralData = dirName + "/volume.iv";

//   if (filenameParams!= "")
//     load_parameters(filenameParams);

//   if (filenameVoxelData!= "")
//     load_volume_data(filenameVoxelData);

//   if(filenameIntegralData != "")
//     load_integral_volume(filenameIntegralData);

//   nColToDraw = -1;
//   nRowToDraw = -1;
//   glGenTextures(1, &wholeTexture);
//   glGenTextures(1, &wholeTextureTrue);

  load_parameters(dirName);
  nColToDraw = -1;
  nRowToDraw = -1;
  glGenTextures(1, &wholeTexture);
  glGenTextures(1, &wholeTextureTrue);

}




template <class T, class U>
T Cube<T,U>::at(int x, int y, int z) {return voxels[z][y-rowOffset*512][x-colOffset*512];}

template <class T, class U>
T Cube<T,U>::at2(int x, int y, int z) {return voxels[z][y][x];}

template <class T, class U>
void Cube<T,U>::put(int x, int y, int z, T value) {voxels[z][y-rowOffset*512][x-colOffset*512] = value;}

template <class T, class U>
void Cube<T,U>::put2(int x, int y, int z, T value) {voxels[z][y][x] = value;}

template <class T, class U>
ulong Cube<T,U>::integral_volume_at(int x, int y, int z) {return voxels_integral[z][y][x];}


template <class T, class U>
void Cube<T,U>::subsampleMean(string dirname)
{

  // // Subsampling /6 /6
  char buff_nfo[1024];
  char buff_vl [1024];

  sprintf(buff_nfo, "%s/volume_subsampled.nfo",dirname.c_str());
  std::ofstream out(buff_nfo);
  out << "cubeWidth " << this->cubeWidth/6 << std::endl;
  out << "cubeHeight " << this->cubeHeight/6 << std::endl;
  out << "cubeDepth " << this->cubeDepth << std::endl;
  out << "parentCubeWidth " << this->cubeWidth/6 << std::endl;
  out << "parentCubeHeight " << this->cubeHeight/6 << std::endl;
  out << "parentCubeDepth " << this->cubeDepth << std::endl;
  out << "voxelWidth " << this->voxelWidth*6 << std::endl;
  out << "voxelHeight " << this->voxelHeight*6 << std::endl;
  out << "voxelDepth " << this->voxelDepth << std::endl;
  out << "rowOffset 0\n";
  out << "colOffset 0\n";
  out << "x_offset 0\n";
  out << "y_offset 0\n";
  out << "z_offset 0\n";
  out.close();

  sprintf(buff_vl, "%s/volume_subsampled.vl", dirname.c_str());
  printf("Creating volume file in %s\n", buff_vl);
  FILE* fp = fopen(buff_vl, "w");
  int line_length = floor((int)cubeWidth/6);
  T buff[line_length];
  for(int i = 0; i < line_length; i++)
    buff[i] = 0;
  for(int i = 0; i < floor((int)cubeHeight/6)*(int)cubeDepth; i++)
    {
      int err = fwrite(buff, sizeof(T), line_length, fp);
      if(err == 0)
        printf("Cube::create_volume_file(%s): error writing the layer %i\n", buff_vl, i);
    }
  fclose(fp);




  Cube<uchar,ulong>* pepe = new Cube<uchar,ulong>(buff_nfo, buff_vl);

  printf("Subsampling mean[");
  //Outer loop, for all the pixels
  for(int z = 0; z < cubeDepth; z++)
    {
      for(int y = 0; y < cubeHeight-6-1; y+=6)
        {
          for(int x = 0; x < cubeWidth-6-1; x+=6)
            {
              int value = 0;
                for(int y2 = 0; y2 < 6; y2++)
                  for(int x2 = 0; x2 < 6; x2++)
                    value+= this->at(x+x2, y+y2, z);
              pepe->put2(x/6,y/6,z, (uchar)(value/36));
            }
        }
      printf("#"); fflush(stdout);
    }
  printf("]\n");

//Subsampling /12 /12 /2
//   string filename = "/media/neurons/neuron1/volume_subsampled_12.vl";
//   printf("Creating volume file in %s\n", filename.c_str());
//   FILE* fp = fopen(filename.c_str(), "w");
//   int line_length = floor((int)cubeWidth/12);
//   T buff[line_length];
//   for(int i = 0; i < line_length; i++)
//     buff[i] = 0;
//   for(int i = 0; i < floor((int)cubeHeight/12)*floor((int)cubeDepth/2); i++)
//     {
//       int err = fwrite(buff, sizeof(T), line_length, fp);
//       if(err == 0)
//         printf("Cube::create_volume_file(%s): error writing the layer %i\n", filename.c_str(), i);
//     }
//   fclose(fp);
//   Cube< uchar >* pepe = new Cube< uchar >("/media/neurons/neuron1/volume_subsampled_12.nfo",
//                                          "/media/neurons/neuron1/volume_subsampled_12.vl");

//   printf("Subsampling mean[");
//   fflush(stdout);
//   //Outer loop, for all the pixels
//   for(int z = 0; z < cubeDepth-3; z+=2)
//     {
//       for(int y = 0; y < cubeHeight-12; y+=12)
//         {
//           for(int x = 0; x < cubeWidth-12; x+=12)
//             {
//               int value = 0;
//               for(int z2=0; z2<2; z2++)
//                 for(int y2 = 0; y2 < 12; y2++)
//                   for(int x2 = 0; x2 < 12; x2++){
// //                     printf("%i %i %i\n", x+x2, y+y2, z+z2);
//                     value+= this->at(x+x2, y+y2, z+z2);
//                   }
//               pepe->put2(x/12,y/12,z/2, (uchar)(value/288));
//             }
//         }
//       printf("#"); fflush(stdout);
//     }
//   printf("]\n");


}

template <class T, class U>
void Cube<T,U>::subsampleMinimum()
{
  string filename = "/media/neurons/neuron1/volume_subsampled_minimum.vl";
  printf("Creating volume file in %s\n", filename.c_str());
  FILE* fp = fopen(filename.c_str(), "w");
  int line_length = floor((int)cubeWidth/6);
  T buff[line_length];
  for(int i = 0; i < line_length; i++)
    buff[i] = 0;
  for(int i = 0; i < floor((int)cubeHeight/6)*cubeDepth; i++)
    {
      int err = fwrite(buff, sizeof(T), line_length, fp);
      if(err == 0)
        printf("Cube::create_volume_file(%s): error writing the layer %i\n", filename.c_str(), i);
    }
  fclose(fp);
  Cube< uchar,ulong >* pepe = new Cube< uchar,ulong >("/media/neurons/neuron1/volume_subsampled.nfo",
                                         "/media/neurons/neuron1/volume_subsampled_minimum.vl");

  printf("Subsampling minimum[");
  //Outer loop, for all the pixels
  for(int z = 0; z < cubeDepth; z++)
    {
      for(int y = 0; y < cubeHeight; y+=6)
        {
          for(int x = 0; x < cubeWidth; x+=6)
            {
              uchar value = 255;
              for(int y2 = 0; y2 < 6; y2++)
                for(int x2 = 0; x2 < 6; x2++)
                  if (this->at2(x+x2, y+y2, z)< value)
                    value = this->at2(x+x2, y+y2, z)   ;
              pepe->put2(x/6,y/6,z, (uchar)(value));
            }
        }
      printf("#"); fflush(stdout);
    }
  printf("]\n");
}

/** The dimensions are on the voxels of the cube*/
template <class T, class U>
double Cube<T,U>::integral_between(int x0, int y0, int z0, int x1, int y1, int z1)
{
  // First step is to calculate in which dimension we have the greatest distance.

  double value_to_return = 0;

  //Case it is in the x
  if ( (abs(x1-x0) >= abs(y1-y0)) &&
       (abs(x1-x0) >= abs(z1-z0)) )
  {
    float my = ((float)(y1-y0))/(x1-x0);
    float mz = ((float)(z1-z0))/(x1-x0);
    float y = 0;
    float z = 0;
    if(min(x0,x1) == x0)
      {
        y = y0;
        z = z0;
      }
    if(min(x0,x1) == x1)
      {
        y = y1;
        z = z1;
      }
    for(int x = min(x0,x1); x <= max(x0,x1); x++)
      {
//         printf("%i %i %i\n", (int)roundf(x),
//                (int)roundf(y),
//                (int)roundf(z));
        value_to_return += this->at2((int)roundf(x),(int)roundf(y),(int)roundf(z));
        y += my;
        z += mz;
      }
    value_to_return = value_to_return / (abs(x1-x0)+1);
  }

  //Case it is in the y
  if ( (abs(y1-y0) >= abs(x1-x0)) &&
       (abs(y1-y0) >= abs(z1-z0)) )
  {
    float mx = ((float)(x1-x0))/(y1-y0);
    float mz = ((float)(z1-z0))/(y1-y0);
    float x = 0;
    float z = 0;
    if(min(y0,y1) == y0)
      {
        x = x0;
        z = z0;
      }
    if(min(y0,y1) == y1)
      {
        x = x1;
        z = z1;
      }
    for(int y = min(y0,y1); y <= max(y0,y1); y++)
      {
//         printf("%i %i %i\n", (int)roundf(x),
//                (int)roundf(y),
//                (int)roundf(z));
        value_to_return += this->at2((int)roundf(x),(int)roundf(y),(int)roundf(z));
        x += mx;
        z += mz;
      }
    value_to_return = value_to_return / (abs(y1-y0)+1);
  }

  //Case it is in the z
  if ( (abs(z1-z0) >= abs(y1-y0)) &&
       (abs(z1-z0) >= abs(x1-x0)) )
  {
    float my = ((float)(y1-y0))/(z1-z0);
    float mx = ((float)(x1-x0))/(z1-z0);
    float y = 0;
    float x = 0;
    if(min(z0,z1) == z0)
      {
        y = y0;
        x = x0;
      }
    if(min(z0,z1) == z1)
      {
        y = y1;
        x = x1;
      }

    for(int z = min(z0,z1); z <= max(z0,z1); z++)
      {
//         printf("%i %i %i\n", (int)roundf(x),
//                (int)roundf(y),
//                (int)roundf(z));
        value_to_return += this->at2((int)roundf(x),(int)roundf(y),(int)roundf(z));
        y += my;
        x += mx;
      }
    value_to_return = value_to_return / (abs(z1-z0)+1);
  }

  return value_to_return;

}

template <class T, class U>
void Cube<T,U>::apply_mask(string mask_nfo, string mask_vl, string output_nfo, string output_vl)
{
  Cube<uchar,ulong>* mask = new Cube<uchar, ulong>(mask_nfo, mask_vl);
  Cube<T,U>*     output = new Cube<T,U>(output_nfo, output_vl);

  printf("Cube<T,U>::apply_mask %s %s %s %s\n[", mask_nfo.c_str(), mask_vl.c_str(), output_nfo.c_str(), output_vl.c_str());
  for(int z = 0; z < mask->cubeDepth; z++){
    for(int y = 0; y < mask->cubeHeight; y++){
      for(int x = 0; x < mask->cubeWidth; x++){
        if(mask->at2(x,y,z) == 255)
          output->put2(x,y,z,this->at2(x,y,z));
        else
          output->put2(x,y,z,0);
      }
    }
    printf("#"); fflush(stdout);
  }
  printf("]\n");
  delete mask;
  delete output;
}


template <class T, class U>
void Cube<T,U>::load_volume_data(string filenameVoxelData)
{
 fildes = open64(filenameVoxelData.c_str(), O_RDWR);

 if(fildes == -1) //The file does not exist
   {
     create_volume_file(filenameVoxelData);
     fildes = open64(filenameVoxelData.c_str(), O_RDWR);
   }

 void* mapped_file = mmap64(0,cubeWidth*cubeHeight*cubeDepth*sizeof(T), PROT_READ|PROT_WRITE, MAP_SHARED, fildes, 0);

 if(mapped_file == MAP_FAILED)
    {
      printf("Cube<T,U>::load_volume_data: There is a bug here, volume not loaded\n");
      exit(0);
    }
  voxels_origin = (T*) mapped_file;
  voxels = (T***)malloc(cubeDepth*sizeof(T**));

  //Initializes the pointer structure to acces quickly to the voxels
  for(int z = 0; z < cubeDepth; z++)
    {
      voxels[z] = (T**)malloc(cubeHeight*sizeof(T*));
      for(int j = 0; j < cubeHeight; j++)
        {
          voxels[z][j]=(T*)&voxels_origin[z*cubeWidth*cubeHeight + j*cubeWidth];
        }
    }
}

//FIXME
template <class T, class U>
void Cube<T,U>::create_volume_file(string filename)
{

  printf("Creating volume file in %s\n", filename.c_str());
  FILE* fp = fopen(filename.c_str(), "w");
  int line_length = 0;
//   if ((colOffset == 0) && (rowOffset == 0))
//     line_length = cubeWidth;
//   else
  line_length = cubeWidth;
  //FIXME
  T buff[line_length];
  for(int i = 0; i < line_length; i++)
    buff[i] = 0;

  for(int i = 0; i < cubeHeight*cubeDepth; i++)
    {
      int err = fwrite(buff, sizeof(T), line_length, fp);
      if(err == 0)
        printf("Cube::create_volume_file(%s): error writing the layer %i\n", filename.c_str(), i);
    }
  fclose(fp);
}


template <class T, class U>
void Cube<T,U>::load_integral_volume(string filename)
{
 int fildes = open64(filename.c_str(), O_RDWR);
 void* mapped_file = mmap64(0,cubeWidth*cubeHeight*cubeDepth*sizeof(U), PROT_READ, MAP_PRIVATE, fildes, 0);
 if(mapped_file == MAP_FAILED)
    {
      printf("Cube<T,U>::load_integral_volume(%s): Volume not loaded\n", filename.c_str());
      exit(0);
    }
  voxels_integral_origin = (U*) mapped_file;
  voxels_integral = (U***)malloc(cubeDepth*sizeof(U**));

  //Initializes the pointer structure to acces quickly to the voxels
  for(int z = 0; z < cubeDepth; z++)
    {
      voxels_integral[z] = (U**)malloc(cubeHeight*sizeof(U*));
      for(int j = 0; j < cubeHeight; j++)
        {
          voxels_integral[z][j]=(U*)&voxels_integral_origin[z*cubeWidth*cubeHeight + j*cubeWidth];
        }
    }
}

template <class T, class U>
void Cube<T,U>::load_parameters(string filenameParams)
 {
   filenameVoxelData = "";

   std::ifstream file(filenameParams.c_str());
   if(!file.good())
     printf("Cube<T,U>::load_parameters: error loading the file %s\n", filenameParams.c_str());

   string name;
   string attribute;
   while(file.good())
     {
       file >> name;
       file >> attribute;
       if(!strcmp(name.c_str(), "voxelDepth"))
         voxelDepth = atof(attribute.c_str());
       else if(!strcmp(name.c_str(), "voxelHeight"))
         voxelHeight = atof(attribute.c_str());
       else if(!strcmp(name.c_str(), "voxelWidth"))
         voxelWidth =  atof(attribute.c_str());
       else if(!strcmp(name.c_str(), "cubeDepth"))
         cubeDepth = atoi(attribute.c_str());
       else if(!strcmp(name.c_str(), "cubeHeight"))
         cubeHeight = atoi(attribute.c_str());
       else if(!strcmp(name.c_str(), "cubeWidth"))
         cubeWidth = atoi(attribute.c_str());
       else if(!strcmp(name.c_str(), "parentCubeDepth"))
         parentCubeDepth = atoi(attribute.c_str());
       else if(!strcmp(name.c_str(), "parentCubeHeight"))
         parentCubeHeight = atoi(attribute.c_str());
       else if(!strcmp(name.c_str(), "parentCubeWidth"))
         parentCubeWidth = atoi(attribute.c_str());
       else if(!strcmp(name.c_str(), "x_offset"))
         x_offset =  atof(attribute.c_str());
       else if(!strcmp(name.c_str(), "y_offset"))
         y_offset =  atof(attribute.c_str());
       else if(!strcmp(name.c_str(), "z_offset"))
         z_offset =  atof(attribute.c_str());
       else if(!strcmp(name.c_str(), "rowOffset"))
         rowOffset = atoi(attribute.c_str());
       else if(!strcmp(name.c_str(), "colOffset"))
         colOffset = atoi(attribute.c_str());
       else if(!strcmp(name.c_str(), "cubeFile")) 
         filenameVoxelData = attribute;
       else
         printf("Cube<T,U>::load_parameters: Attribute %s and value %s not known\n", name.c_str(), attribute.c_str());
     }

   if(filenameVoxelData != ""){
     string directory = filenameParams.substr(0,filenameParams.find_last_of("/\\")+1);
     load_volume_data(directory + filenameVoxelData);

   }

   //  #if debug
   printf("Cube parameters:\n");
   printf("  cubeWidth %i cubeHeight %i cubeDepth %i\n", cubeWidth, cubeHeight, cubeDepth);
   printf("  voxelWidth %f voxelHeight %f voxelDepth %f\n", voxelWidth, voxelHeight, voxelDepth);
   //  #endif
 }

template <class T, class U>
void Cube<T,U>::create_integral_cube(string filename)
{
  char buff[1024];
  sprintf(buff, "touch %s", filename.c_str());
  system(buff);
  printf("Cube::create_integral_cube(): creating the file %s\n", filename.c_str());
  int fp = open64(filename.c_str(), O_WRONLY || O_SYNC || O_LARGEFILE );
  if(fp == -1)
    {
      printf("Cube::create_integral_cube(): error openning the file %s\n", filename.c_str());
      exit(0);
    }

  ulong* temporal_layer_integral = (ulong*)malloc(cubeWidth*cubeHeight*sizeof(ulong));
  for(int i = 0; i < cubeHeight; i++)
    for(int j = 0; j < cubeWidth; j++)
        temporal_layer_integral[i*cubeWidth + j] = 0;

  int* temporal_layer_depth = (int*)malloc(cubeWidth*cubeHeight*sizeof(int));
  for(int i = 0; i < cubeHeight; i++)
    for(int j = 0; j < cubeWidth; j++)
        temporal_layer_depth[i*cubeWidth + j] = 0;

  printf("Calculating the integral cube: [");
  ulong accumulator = 0;

  for(int depth = 0; depth < cubeDepth; depth++)
    {
      //Calculates the first row
      accumulator = 0;
      for(int col = 0; col < cubeWidth; col++){
        accumulator += voxels[depth][0][col];
        temporal_layer_integral[col] += accumulator;
        temporal_layer_depth[col] += voxels[depth][0][col];
      }

      accumulator = voxels[depth][0][0];
      for(int row = 1; row < cubeHeight; row++){
        accumulator += voxels[depth][row][0];
        temporal_layer_integral[row*cubeWidth] += accumulator;
        temporal_layer_depth[row*cubeWidth] += voxels[depth][row][0];
      }

      //Calculates the rest of the rows
      for(int row = 1; row < cubeHeight; row++){
        for(int col = 1; col < cubeWidth; col++){
          temporal_layer_depth[col + row*cubeWidth] += voxels[depth][row][col];
          temporal_layer_integral[col + row*cubeWidth] =
            temporal_layer_depth[col + row*cubeWidth] +
            temporal_layer_integral[col + (row-1)*cubeWidth] -
            temporal_layer_integral[col-1 + (row-1)*cubeWidth] +
            temporal_layer_integral[col-1 + row*cubeWidth];
        }
      }
      printf("#");
      fflush(stdout);
//       printf("%lu\n", temporal_layer_integral[cubeWidth*cubeHeight-1]);

      //Saves the first image in the volum2
      int err = write(fp, temporal_layer_integral, cubeHeight*cubeWidth*sizeof(ulong));
      if(err == -1)
        printf("Cube::create_integral_cube(%s): error writing the layer %i\n", filename.c_str(), depth);
    }
  printf("]\n");
  close(fp);
}

// template <class T, class U>
// void Cube<T,U>::create_integral_cube_float(string filename)
// {
//   printf("Cube::create_integral_cube(): creating the file %s\n", filename.c_str());
//   int fp = open64(filename.c_str(), O_WRONLY || O_SYNC || O_LARGEFILE );
//   if(fp == -1)
//     {
//       printf("Cube::create_integral_cube(): error openning the file %s\n", filename.c_str());
//       exit(0);
//     }

//   double* temporal_layer_integral = (double*)malloc(cubeWidth*cubeHeight*sizeof(double));
//   for(int i = 0; i < cubeHeight; i++)
//     for(int j = 0; j < cubeWidth; j++)
//         temporal_layer_integral[i*cubeWidth + j] = 0;

//   double* temporal_layer_depth = (double*)malloc(cubeWidth*cubeHeight*sizeof(double));
//   for(int i = 0; i < cubeHeight; i++)
//     for(int j = 0; j < cubeWidth; j++)
//         temporal_layer_depth[i*cubeWidth + j] = 0;

//   printf("Calculating the integral cube: [");
//   double accumulator = 0;

//   for(int depth = 0; depth < cubeDepth; depth++)
//     {
//       //Calculates the first row
//       accumulator = 0;
//       for(int col = 0; col < cubeWidth; col++){
//         accumulator += voxels[depth][0][col];
//         temporal_layer_integral[col] += accumulator;
//         temporal_layer_depth[col] += voxels[depth][0][col];
//       }

//       accumulator = voxels[depth][0][0];
//       for(int row = 1; row < cubeHeight; row++){
//         accumulator += voxels[depth][row][0];
//         temporal_layer_integral[row*cubeWidth] += accumulator;
//         temporal_layer_depth[row*cubeWidth] += voxels[depth][row][0];
//       }

//       //Calculates the rest of the rows
//       for(int row = 1; row < cubeHeight; row++){
//         for(int col = 1; col < cubeWidth; col++){
//           temporal_layer_depth[col + row*cubeWidth] += voxels[depth][row][col];
//           temporal_layer_integral[col + row*cubeWidth] =
//             temporal_layer_depth[col + row*cubeWidth] +
//             temporal_layer_integral[col + (row-1)*cubeWidth] -
//             temporal_layer_integral[col-1 + (row-1)*cubeWidth] +
//             temporal_layer_integral[col-1 + row*cubeWidth];
//         }
//       }
//       printf("#");
//       fflush(stdout);
// //       printf("%lu\n", temporal_layer_integral[cubeWidth*cubeHeight-1]);

//       //Saves the first image in the volum2
//       int err = write(fp, temporal_layer_integral, cubeHeight*cubeWidth*sizeof(double));
//       if(err == -1)
//         printf("Cube::create_integral_cube(%s): error writing the layer %i\n", filename.c_str(), depth);
//     }
//   printf("]\n");
//   close(fp);
// }

template <class T, class U>
void Cube<T,U>::print_statistics(string filename)
{
  printf("%f %f %f\n", voxels[0][0][0], voxels[112][511][511], voxels[30][400][40]);

  //Will find the mean and the variance and print it. Also the max and the min
  float max = 1e-12;
  float min = 1e12;
  float mean = 0;
  for(int z = 0; z < cubeDepth; z++)
    for(int y = 0; y < cubeHeight; y++)
      for(int x = 0; x < cubeWidth; x++)
        {
          mean += voxels[z][y][x];
          if(voxels[z][y][x] > max)
            max = voxels[z][y][x];
          if(voxels[z][y][x] < min)
            min = voxels[z][y][x];
        }

  mean = mean / (cubeDepth*cubeHeight*cubeWidth);
  printf("Cube mean value is %06.015f, max = %06.015f, min = %06.015f\n", mean, max, min);
}

template <class T, class U>
void Cube<T,U>::histogram(string filename)
{

  printf("Cube<T,U>::histogram [");
  float max = 1e-12;
  float min = 1e12;
  float mean = 0;
  for(int z = 0; z < cubeDepth; z++)
    for(int y = 0; y < cubeHeight; y++)
      for(int x = 0; x < cubeWidth; x++)
        {
          mean += voxels[z][y][x];
          if(voxels[z][y][x] > max)
            max = voxels[z][y][x];
          if(voxels[z][y][x] < min)
            min = voxels[z][y][x];
        }

  float range = max - min;

  vector< int > boxes(100);
  for(int i = 0; i < 100; i++)
    boxes[i] = 0;

  for(int z = 0; z < cubeDepth; z++){
    for(int y = 0; y < cubeHeight; y++){
      for(int x = 0; x < cubeWidth; x++){
        boxes[floor(100*(this->at2(x,y,z)-min)/range)] += 1;
      }
    }
    printf("#"); fflush(stdout);
  }
  printf("]\n");

  if(filename == ""){
    for(int i =0; i < boxes.size(); i++)
      printf("%i ", boxes[i]);
    printf("\n");
  }
  else{
    std::ofstream out(filename.c_str());
    out << min << std::endl;
    out << max << std::endl;
    for(int i = 0; i < boxes.size(); i++)
      out << boxes[i] << std::endl;
    out.close();
  }
}



template <class T, class U>
void Cube<T,U>::histogram_ignoring_zeros(string filename)
{
  printf("Cube<T,U>::histogram [");
  float max = 1e-12;
  float min = 1e12;
  float mean = 0;
  for(int z = 0; z < cubeDepth; z++)
    for(int y = 0; y < cubeHeight; y++)
      for(int x = 0; x < cubeWidth; x++)
        {
          mean += voxels[z][y][x];
          if(voxels[z][y][x] > max)
            max = voxels[z][y][x];
          if(voxels[z][y][x] < min)
            min = voxels[z][y][x];
        }

  float range = max - min;

  vector< int > boxes(100);
  for(int i = 0; i < 100; i++)
    boxes[i] = 0;

  for(int z = 0; z < cubeDepth; z++){
    for(int y = 0; y < cubeHeight; y++){
      for(int x = 0; x < cubeWidth; x++){
        if(this->at(x,y,z) == 0.0)
          continue;
        boxes[floor(100*(this->at2(x,y,z)-min)/range)] += 1;
      }
    }
    printf("#"); fflush(stdout);
  }
  printf("]\n");

  if(filename == ""){
    for(int i =0; i < boxes.size(); i++)
      printf("%i ", boxes[i]);
    printf("\n");
  }
  else{
    std::ofstream out(filename.c_str());
    out << min << std::endl;
    out << max << std::endl;
    for(int i = 0; i < boxes.size(); i++)
      out << boxes[i] << std::endl;
    out.close();
  }
}


template <class T, class U>
void Cube<T,U>::create_cube_from_kevin_images(
        string directory, string format, int begin, int end,
        float voxelWidth_p, float voxelHeight_p, float voxelDepth_p)
{
  string name = directory + "/volume.vl";
  FILE *fp = fopen(name.c_str(), "w+");
  char image_format[1024];
  sprintf(image_format, "%s/%s", directory.c_str(), format.c_str());
  char buff[1024];
  printf("Generating the cube: [");
//   cvNamedWindow("pepe",1);
  int cubeWidth_p = 0;
  int cubeHeight_p = 0;
  for(int z = begin; z <= end; z++)
    {
      printf("#");
      fflush(stdout);
      sprintf(buff, image_format, z);
      IplImage* pepe = cvLoadImage(buff,0);
      cubeWidth_p = pepe->width;
      cubeHeight_p = pepe->height;
      IplImage* pepe_low = cvCreateImage(cvSize(pepe->width, pepe->height), IPL_DEPTH_8U, 1);
      for(int y = 0; y < pepe_low->height; y++)
        for(int x = 0; x < pepe_low->width; x++)
          pepe_low->imageData[x + y*pepe_low->widthStep] = (T)pepe->imageData[x + y*pepe->widthStep]*32;
//       cvShowImage("pepe", pepe_low);
//       cvWaitKey(1000);
//       if(z == 20)
//         cvSaveImage("kevin.jpg", pepe_low);
      for(int y = 0; y < pepe_low->height; y++)
        fwrite( ((T*)(pepe_low->imageData + pepe_low->widthStep*y )), sizeof(T), pepe_low->width, fp);
      cvReleaseImage(&pepe);
      cvReleaseImage(&pepe_low);
    }

  string parameters_file = directory + "/volume.nfo";
  std::ofstream out_w(parameters_file.c_str());
  out_w << "parentCubeWidth " << cubeWidth_p << std::endl;
  out_w << "parentCubeHeight " << cubeHeight_p << std::endl;
  out_w << "parentCubeDepth " << end - begin + 1 << std::endl;
  out_w << "cubeWidth " << cubeWidth_p << std::endl;
  out_w << "cubeHeight " << cubeHeight_p << std::endl;
  out_w << "cubeDepth " << end - begin + 1 << std::endl;
  out_w << "voxelWidth " << voxelWidth_p << std::endl;
  out_w << "voxelHeight " << voxelHeight_p << std::endl;
  out_w << "voxelDepth " << voxelDepth_p << std::endl;
  out_w << "rowOffset  0" << std::endl;
  out_w << "colOffset  0" << std::endl;
  out_w << "x_offset  0" << std::endl;
  out_w << "y_offset  0" << std::endl;
  out_w << "z_offset  0" << std::endl;
  out_w.close();

  this->cubeWidth   = cubeWidth_p     ;
  this->cubeHeight  = cubeHeight_p    ;
  this->cubeDepth   = end - begin + 1 ;
  this->voxelWidth  = voxelWidth_p    ;
  this->voxelHeight = voxelHeight_p   ;
  this->voxelDepth  = voxelDepth_p    ;

  printf("]\n");
  fclose(fp);
  load_volume_data(name);
}

template <class T, class U>
void Cube<T,U>::create_cube_from_directory(string directory, string format, int layer_begin, int layer_end, float voxelWidth, float voxelHeight, float voxelDepth)
{

  char image_format[1024];
  sprintf(image_format, "%s/%s", directory.c_str(), format.c_str());
  char buff[1024];

  sprintf(buff, image_format, layer_begin);
  IplImage* pepe = cvLoadImage(buff,0);

  string parameters_file = directory + "/volume.nfo";
  std::ofstream out_w(parameters_file.c_str());
  out_w << "parentCubeWidth " << pepe->width << std::endl;
  out_w << "parentCubeHeight " << pepe->height << std::endl;
  out_w << "parentCubeDepth " << layer_end - layer_begin + 1 << std::endl;
  out_w << "cubeWidth " << pepe->width << std::endl;
  out_w << "cubeHeight " << pepe->height << std::endl;
  out_w << "cubeDepth " << layer_end - layer_begin + 1 << std::endl;
  out_w << "voxelWidth " << voxelWidth << std::endl;
  out_w << "voxelHeight " << voxelHeight << std::endl;
  out_w << "voxelDepth " << voxelDepth << std::endl;
  out_w << "rowOffset  0" << std::endl;
  out_w << "colOffset  0" << std::endl;
  out_w << "x_offset  0" << std::endl;
  out_w << "y_offset  0" << std::endl;
  out_w << "z_offset  0" << std::endl;
  out_w.close();

  this->cubeWidth   = pepe->width     ;
  this->cubeHeight  = pepe->height    ;
  this->cubeDepth   = layer_end - layer_begin + 1 ;
  this->voxelWidth  = voxelWidth    ;
  this->voxelHeight = voxelHeight   ;
  this->voxelDepth  = voxelDepth    ;

  string name = directory + "/volume.vl";
  FILE *fp = fopen(name.c_str(), "w+");
  printf("Generating the cube: [");
//   cvNamedWindow("pepe",1);
  for(int z = layer_begin; z <= layer_end; z++)
    {
      printf("#");
      fflush(stdout);
      sprintf(buff, image_format, z);
      IplImage* pepe = cvLoadImage(buff,0);
      IplImage* pepe_low = cvCreateImage(cvSize(pepe->width, pepe->height), IPL_DEPTH_8U, 1);
      for(int y = 0; y < pepe_low->height; y++)
        for(int x = 0; x < pepe_low->width; x++)
          pepe_low->imageData[x + y*pepe_low->widthStep] = 255-(T)pepe->imageData[x + y*pepe->widthStep];
//       cvShowImage("pepe", pepe_low);
//       cvWaitKey(1000);
//       if(z == 20)
//         cvSaveImage("kevin.jpg", pepe_low);
      for(int y = 0; y < pepe_low->height; y++)
        fwrite( ((T*)(pepe_low->imageData + pepe_low->widthStep*y )), sizeof(T), pepe_low->width, fp);
      cvReleaseImage(&pepe);
      cvReleaseImage(&pepe_low);
    }
  printf("]\n");
  fclose(fp);
  load_volume_data(name);
}

template <class T, class U>
void Cube<T,U>::create_cube_from_directory_matrix
(
 string directory, string format,
 int row_begin, int row_end,
 int col_begin, int col_end,
 int layer_begin, int layer_end,
 float voxelWidth, float voxelHeight, float voxelDepth
)
{

  char image_format[1024];
  sprintf(image_format, "%s/%s", directory.c_str(), format.c_str());
  char buff[1024];
  sprintf(buff, image_format, layer_begin);

  sprintf(buff, image_format, row_begin, col_begin, layer_begin);
  IplImage* pepe = cvLoadImage(buff,0);

  string parameters_file = directory + "/volume.nfo";
  std::ofstream out_w(parameters_file.c_str());
  out_w << "parentCubeWidth " << pepe->width*(col_end - col_begin +1) << std::endl;
  out_w << "parentCubeHeight " << pepe->height*(row_end - row_begin +1) << std::endl;
  out_w << "parentCubeDepth " << layer_end - layer_begin + 1 << std::endl;
  out_w << "cubeWidth " << pepe->width*(col_end - col_begin +1) << std::endl;
  out_w << "cubeHeight " << pepe->height*(row_end - row_begin +1) << std::endl;
  out_w << "cubeDepth " << layer_end - layer_begin + 1 << std::endl;
  out_w << "voxelWidth " << voxelWidth << std::endl;
  out_w << "voxelHeight " << voxelHeight << std::endl;
  out_w << "voxelDepth " << voxelDepth << std::endl;
  out_w << "rowOffset  0" << std::endl;
  out_w << "colOffset  0" << std::endl;
  out_w << "x_offset  0" << std::endl;
  out_w << "y_offset  0" << std::endl;
  out_w << "z_offset  0" << std::endl;
  out_w.close();

  this->cubeWidth   = pepe->width*(col_end - col_begin +1)     ;
  this->cubeHeight  = pepe->height*(row_end - row_begin +1)    ;
  this->cubeDepth   = layer_end - layer_begin + 1 ;
  this->voxelWidth  = voxelWidth    ;
  this->voxelHeight = voxelHeight   ;
  this->voxelDepth  = voxelDepth    ;


  string name = directory + "/volume.vl";
  FILE *fp = fopen(name.c_str(), "w+");
  printf("Generating the cube: [");
//   cvNamedWindow("pepe",1);
  for(int z = layer_begin; z <= layer_end; z++)
    {
      for(int y = row_begin; y <= row_end; y++)
        {
          IplImage* row_imgs[col_end - col_begin + 1];
          for(int x = 0; x < col_end - col_begin + 1; x++)
            {
              sprintf(buff, image_format, y,x+col_begin,z);
              row_imgs[x] = cvLoadImage(buff,0);
            }
          for(int row = 0; row < row_imgs[0]->height; row++)
            for(int col = 0; col < col_end - col_begin + 1; col ++)
              fwrite( ((T*)(row_imgs[col]->imageData + row_imgs[col]->widthStep*row )), sizeof(T), row_imgs[col]->width, fp);

          for(int x = 0; x < col_end - col_begin + 1; x++)
            cvReleaseImage(&row_imgs[x]);
        }
      printf("#");
      fflush(stdout);
    }
  printf("]\n");
  fclose(fp);
  load_volume_data(name);
}


template <class T, class U>
void Cube<T,U>::save_as_image_stack(string dirname)
{
  printf("Cube<T,U>::save_as_image_stack saving images in %s [", dirname.c_str());
  char image_name[1024];
  for(int z = 0; z < cubeDepth; z++)
    {
      sprintf(image_name, "%s/%03i.png", dirname.c_str(), z);
      IplImage* toSave = cvCreateImage(cvSize(cubeWidth, cubeHeight), IPL_DEPTH_8U, 1);
      for(int y = 0; y < cubeHeight; y++)
        for(int x = 0; x < cubeWidth; x++)
          toSave->imageData[x + y*toSave->widthStep] = this->at2(x,y,z);
      cvSaveImage(image_name, toSave);
      printf("#"); fflush(stdout);
    }
  printf("]\n");
}

template <class T, class U>
void Cube<T,U>::createMIPImage(string filename)
{
  if(filename == "")
    filename = "MIP.jpg";
  IplImage* output = cvCreateImage(cvSize(cubeWidth, cubeHeight), IPL_DEPTH_8U, 1);
  uint minimum_intensity = 255;
  printf("creatingMIPImage [");
  for(int y = 0; y < cubeHeight; y++){
    for(int x = 0; x < cubeWidth; x++){
      minimum_intensity = 255;
      for(int z = 0; z < cubeDepth; z++)
        {
          if(this->at2(x,y,z) < minimum_intensity)
            minimum_intensity = this->at2(x,y,z);
        }
      output->imageData[y*output->widthStep + x] = minimum_intensity;
    }
    if(y%100 == 0){
      printf("#");
      fflush(stdout);
    }
  }
  printf("]\n");
  cvSaveImage(filename.c_str(), output);
}

template <class T, class U>
void Cube<T,U>::micrometersToIndexes(vector< float >& micrometers, vector< int >& indexes)
{
  indexes[0] = (int)(float(cubeWidth)/2 + micrometers[0]/voxelWidth);
  indexes[1] = (int)(float(cubeHeight)/2 - micrometers[1]/voxelHeight);
  indexes[2] = (int)(float(cubeDepth)/2 + micrometers[2]/voxelDepth);
}

template <class T, class U>
void Cube<T,U>::indexesToMicrometers(vector< int >& indexes, vector< float >& micrometers)
{
  micrometers[0] = (float)(-parentCubeWidth*voxelWidth/2 + indexes[0]*voxelWidth + colOffset*cubeWidth*voxelWidth);
  micrometers[1] = (float)(parentCubeHeight*voxelHeight/2 - indexes[1]*voxelHeight - rowOffset*cubeHeight*voxelHeight);
  micrometers[2] = (float)(-parentCubeDepth*voxelDepth/2 + indexes[2]*voxelDepth);
}

template <class T, class U>
void Cube<T,U>::print()
{
  printf("Cube parameters:\n");
  printf("  indexes: %llu %llu %llu\n", cubeWidth, cubeHeight, cubeDepth);
  printf("  voxels : %f %f %f\n", voxelWidth, voxelHeight, voxelDepth);
}

//########## CUBE DRAWING ####################


template <class T, class U>
void Cube<T,U>::load_whole_texture()
{

  nColToDraw = -1;
  nRowToDraw = -1;

  GLint max_texture_size = 0;
  glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_texture_size);
  #if debug
  printf("Cube::load_whole_texture() max_texture_size = %i\n", max_texture_size);
  printf("Cube::load_whole_texture() creating the texture buffer\n");
  #endif

  printf("Loading the whole texture[");

  //Creates the array with the texture. Coded to avoid float multiplications
  wholeTextureDepth = min(max_texture_size, (int)cubeDepth);
  T* texels =(T*)( malloc(max_texture_size*max_texture_size*wholeTextureDepth*sizeof(T)));
  for(int i = 0; i < max_texture_size*max_texture_size*wholeTextureDepth; i++)
    texels[i] = 0;
  float scale_x = float(cubeWidth)/max_texture_size;
  float scale_y = float(cubeHeight)/max_texture_size;
  float scale_z = float(cubeDepth)/wholeTextureDepth;
  float temp_x = 0; float temp_y = 0; float temp_z = 0;
  int temp_x_i = 0; int temp_y_i = 0; int temp_z_i = 0;
  for(int z = 0; z < wholeTextureDepth; z++)
    {
      temp_z_i = (int)temp_z;
      temp_y = 0;
      for(int y = 0; y < max_texture_size; y++)
        {
          temp_y_i = (int)temp_y;
          temp_x = 0;
          for(int x = 0; x < max_texture_size; x++)
            {
              temp_x_i = (int)temp_x;
              T voxel = voxels[temp_z_i][temp_y_i][temp_x_i];
//               if(voxel < 128)
                texels[z*max_texture_size*max_texture_size + y*max_texture_size + x ] = voxel;
//               else
//                 texels[z*max_texture_size*max_texture_size + y*max_texture_size + x ] = 255;
              temp_x = temp_x + scale_x;
            }
          temp_y = temp_y + scale_y;
        }
      temp_z = temp_z + scale_z;
      printf("#");
      fflush(stdout);
    }
  printf("]\n");
  #if debug
  printf("Cube::load_whole_texture() created the texture buffer\n");
  #endif

  glEnable(GL_TEXTURE_3D);
  glBindTexture(GL_TEXTURE_3D, wholeTexture);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
//   glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//   glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP);
  glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_PRIORITY, 1.0);
  GLfloat border_color[4];
  for(int i = 0; i < 4; i++)
    border_color[i] = 1.0;
  glTexParameterfv(GL_TEXTURE_3D, GL_TEXTURE_BORDER_COLOR, border_color);
  glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE8, max_texture_size, max_texture_size, wholeTextureDepth, 0, GL_LUMINANCE,
             GL_UNSIGNED_BYTE, texels);
  GLclampf priority = 1;
  glPrioritizeTextures(1, &wholeTexture, &priority);
}


template <class T, class U>
void Cube<T,U>::load_texture_brick(int row, int col)
{
  nColToDraw = col;
  nRowToDraw = row;
  GLint max_texture_size = 0;
  glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_texture_size);

  #if debug
  printf("Cube::load_texture_brick() max_texture_size = %i\n", max_texture_size);
  printf("Cube::load_texture_brick() creating the texture buffer\n");
  #endif

  int limit_x = min((int)cubeWidth, min((int)max_texture_size, (int)cubeWidth - (nColToDraw*max_texture_size)));
  int limit_y = min((int)cubeHeight,  min((int)max_texture_size, (int)cubeHeight - (nRowToDraw*max_texture_size)));
  if( (limit_x<0) || (limit_y<0))
    {
      printf("Cube<T,U>::load_texture_brick requested col %i row %i out of range, loading 0,0\n", nColToDraw, nRowToDraw);
      nColToDraw = 0;
      nRowToDraw = 0;
      limit_x = max_texture_size;
      limit_y = max_texture_size;
    }

  glEnable(GL_TEXTURE_3D);
  glBindTexture(GL_TEXTURE_3D, wholeTexture);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP);
  glTexParameterf(GL_TEXTURE_3D,  GL_TEXTURE_PRIORITY, 1.0);
  GLfloat border_color[4];
  for(int i = 0; i < 4; i++)
    border_color[i] = 1.0;
  glTexParameterfv(GL_TEXTURE_3D, GL_TEXTURE_BORDER_COLOR, border_color);

  //   Creates the array with the texture. Coded to avoid float multiplications
  wholeTextureDepth = min(max_texture_size, (int)cubeDepth);


  printf("Loading texture brick %i %i [", row, col);

  if(sizeof(T) == 1)
    {
      uchar* texels =(uchar*)( malloc(limit_x*limit_y*wholeTextureDepth*sizeof(uchar)));
      int col_offset = (col-colOffset)*max_texture_size;
      int row_offset = (row-rowOffset)*max_texture_size;

      for(int z = 0; z < wholeTextureDepth; z++)
        {
          int depth_z = z*limit_x*limit_y;
          for(int y = 0; y < limit_y; y++)
            {
              int depth_y = y*limit_x;
              for(int x = 0; x < limit_x; x++)
                {
                  texels[depth_z + depth_y + x] = voxels[z][row_offset+y][col_offset+x];
                }
            }
          printf("#");
          fflush(stdout);
        }
      printf("]\n");
      glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE8, limit_x, limit_y, wholeTextureDepth, 0, GL_LUMINANCE,
             GL_UNSIGNED_BYTE, texels);
      free(texels);
    }

  if(sizeof(T) == 4)
    {
      int col_offset = col*max_texture_size;
      int row_offset = row*max_texture_size;
      float* texels =(float*)( malloc(limit_x*limit_y*wholeTextureDepth*sizeof(float)));
      printf("Cube::load_texture_brick() creating texture from floats\n");
      float max_texture = -10e6;
      float min_texture = 10e6;
      for(int z = 0; z < wholeTextureDepth; z++)
        {
          int depth_z = z*limit_y*limit_x;
          for(int y = 20; y < limit_y; y++)
            {
              int depth_y = y*limit_x;
              for(int x = 0; x < limit_x; x++)
                {
                  if(max_texture <  this->at(col_offset+x,row_offset+y,z))
                    max_texture = this->at(col_offset+x,row_offset+y,z);
                  if(min_texture >  this->at(col_offset+x,row_offset+y,z))
                    min_texture = this->at(col_offset+x,row_offset+y,z);
                }
            }
        }
      printf("Cube::load_texture_brick(): max=%f and min=%f\n", (float)max_texture, (float)min_texture);
      printf("Loading texture brick %i %i [", row, col);
      for(int z = 0; z < wholeTextureDepth; z++)
        {
          int depth_z = z*limit_y*limit_x;
          for(int y = 0; y < limit_y; y++)
            {
              int depth_y = y*limit_x;
              for(int x = 0; x < limit_x; x++)
                {
                  if((y<20) || (z>86) || (z<10) ){
                    texels[depth_z + depth_y + x] = 0;
                  }
                  else{
                    texels[depth_z + depth_y + x] = (this->at(col_offset+x,row_offset+y,z) - min_texture)
                      / (max_texture - min_texture);
                  }
                }
            }
          printf("#");
          fflush(stdout);
        }
      printf("]\n");
      glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE, limit_x, limit_y, wholeTextureDepth, 0, GL_LUMINANCE,
                   GL_FLOAT, texels);

      free(texels);
    }

  #if debug
  printf("Cube::load_whole_texture() created the texture buffer\n");
  #endif


//   glBindTexture(GL_TEXTURE_3D, wholeTexture);
}

template <class T, class U>
void Cube<T,U>::load_thresholded_texture_brick(int row, int col, float threshold)
{
  nColToDraw = col;
  nRowToDraw = row;
  GLint max_texture_size = 0;
  glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_texture_size);

  #if debug
  printf("Cube::load_texture_brick() max_texture_size = %i\n", max_texture_size);
  printf("Cube::load_texture_brick() creating the texture buffer\n");
  #endif

  int limit_x = min((int)max_texture_size, (int)cubeWidth - (nColToDraw*max_texture_size));
  int limit_y = min((int)max_texture_size, (int)cubeHeight - (nRowToDraw*max_texture_size));
  if( (limit_x<0) || (limit_y<0))
    {
      printf("Cube<T,U>::load__thresholded_texture_brick_fload requested col %i row %i out of range, loading 0,0\n", nColToDraw, nRowToDraw);
      nColToDraw = 0;
      nRowToDraw = 0;
      limit_x = max_texture_size;
      limit_y = max_texture_size;
    }


//   if(sizeof(T) == 1) {
//     printf("Cube::load_thresholded_texture_brick called when it is a uchar cube\n");
//     return;
//   }

  //   Creates the array with the texture. Coded to avoid float multiplications
  wholeTextureDepth = min(max_texture_size, (int)cubeDepth);
  uchar* texels =(uchar*)( malloc(wholeTextureDepth*limit_x*limit_y*sizeof(uchar)));

  printf("Loading thresholded texture brick %i %i %f [", row, col, threshold);

  int col_offset = col*max_texture_size;
  int row_offset = row*max_texture_size;

  //       printf("Cube::load_texture_brick(): max=%f and min=%f\n", max_texture, min_texture);
  for(int z = 0; z < wholeTextureDepth; z++)
    {
      int depth_z = z*limit_y*limit_x;
      for(int y = 0; y < limit_y; y++)
        {
          int depth_y = y*limit_x;
          for(int x = 0; x < limit_x; x++)
            {
              if(this->at(col_offset+x,row_offset+y,z) > threshold)
                texels[depth_z + depth_y + x] = 255;
              else
                texels[depth_z + depth_y + x] = this->at(col_offset+x,row_offset+y,z);
            }
        }
      printf("#");
      fflush(stdout);
    }
  printf("]\n");

  #if debug
  printf("Cube::load_whole_texture() created the texture buffer\n");
  #endif

  glEnable(GL_TEXTURE_3D);
  glBindTexture(GL_TEXTURE_3D, wholeTexture);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP);
  glTexParameterf(GL_TEXTURE_3D,  GL_TEXTURE_PRIORITY, 1.0);
  GLfloat border_color[4];
  for(int i = 0; i < 4; i++)
    border_color[i] = 1.0;
  glTexParameterfv(GL_TEXTURE_3D, GL_TEXTURE_BORDER_COLOR, border_color);

  glBindTexture(GL_TEXTURE_3D, wholeTexture);

  glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE, limit_x, limit_y, wholeTextureDepth, 0, GL_LUMINANCE,
               GL_UNSIGNED_BYTE, texels);

}


template <class T, class U>
void Cube<T,U>::load_thresholded_maxmin_texture_brick_float(int row, int col, float threshold_low, float threshold_high)
{
  nColToDraw = col;
  nRowToDraw = row;
  GLint max_texture_size = 0;
  glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_texture_size);

  int limit_x = min((int)max_texture_size, (int)cubeWidth - (nColToDraw*max_texture_size));
  int limit_y = min((int)max_texture_size, (int)cubeHeight - (nRowToDraw*max_texture_size));
  if( (limit_x<0) || (limit_y<0))
    {
      printf("Cube<T,U>::load__thresholded_texture_brick_fload requested col %i row %i out of range, loading 0,0\n", nColToDraw, nRowToDraw);
      nColToDraw = 0;
      nRowToDraw = 0;
      limit_x = max_texture_size;
      limit_y = max_texture_size;
    }


  //   Creates the array with the texture. Coded to avoid float multiplications
  wholeTextureDepth = min(max_texture_size, (int)cubeDepth);
  float* texels =(float*)( malloc(limit_x*limit_y*wholeTextureDepth*sizeof(float)));

  printf("Loading texture brick %i %i [", row, col);
  int col_offset = col*max_texture_size;
  int row_offset = row*max_texture_size;

//   printf("Cube::load_texture_brick() creating texture from floats\n");
//   float max_texture = -10e6;
//   float min_texture = 10e6;
//   for(int z = 0; z < wholeTextureDepth; z++)
//     {
//       int depth_z = z*limit_y*limit_x;
//       for(int y = 0; y < limit_y; y++)
//         {
//           int depth_y = y*limit_x;
//           for(int x = 0; x < limit_x; x++)
//             {
//               if(max_texture <  this->at(col_offset+x,row_offset+y,z))
//                 max_texture = this->at(col_offset+x,row_offset+y,z);
//               if(min_texture >  this->at(col_offset+x,row_offset+y,z))
//                 min_texture = this->at(col_offset+x,row_offset+y,z);
//             }
//         }
//     }
//   min_texture = threshold;

  //       printf("Cube::load_texture_brick(): max=%f and min=%f\n", max_texture, min_texture);
  int points_over = 0;
  int points_under = 0;
  int points_middle = 0;
  printf("Loading texture brick float %i %i [", row, col);
  for(int z = 0; z < wholeTextureDepth; z++)
    {
      int depth_z = z*limit_y*limit_x;
      for(int y = 0; y < limit_y; y++)
        {
          int depth_y = y*limit_x;
          for(int x = 0; x < limit_x; x++)
            {
              if(this->at(col_offset+x,row_offset+y,z) > threshold_high){
                texels[depth_z + depth_y + x] = 0;
                points_over++;
                continue;
              }
              if(this->at(col_offset+x, row_offset+y, z) < threshold_low){
                texels[depth_z + depth_y + x] = 0;
                points_under++;
                continue;
              }
              texels[depth_z + depth_y + x] = (this->at(col_offset+x,row_offset+y,z) - threshold_low)
                  / (threshold_high - threshold_low);
              points_middle++;
//               else
//                 texels[depth_z + depth_y +z] = 0;
            }
        }
      printf("#");
      fflush(stdout);
    }
  printf("] %i %i %i\n", points_over, points_under, points_middle);

  #if debug
  printf("Cube::load_texture_brick_float() created the texture buffer\n");
  #endif

  glEnable(GL_TEXTURE_3D);
  glBindTexture(GL_TEXTURE_3D, wholeTexture);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP);
  glTexParameterf(GL_TEXTURE_3D,  GL_TEXTURE_PRIORITY, 1.0);
  GLfloat border_color[4];
  for(int i = 0; i < 4; i++)
    border_color[i] = 1.0;
  glTexParameterfv(GL_TEXTURE_3D, GL_TEXTURE_BORDER_COLOR, border_color);

//   printf("Blah2 %i %i %i\n", limit_x, limit_y, wholeTextureDepth);

  glBindTexture(GL_TEXTURE_3D, wholeTexture);

  glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE, limit_x, limit_y, wholeTextureDepth, 0, GL_LUMINANCE,
               GL_FLOAT, texels);

  free(texels);
}

template <class T, class U>
void Cube<T,U>::load_thresholded_texture_brick_float(int row, int col, float threshold)
{
  nColToDraw = col;
  nRowToDraw = row;
  GLint max_texture_size = 0;
  glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_texture_size);

  int limit_x = min((int)max_texture_size, (int)cubeWidth - (nColToDraw*max_texture_size));
  int limit_y = min((int)max_texture_size, (int)cubeHeight - (nRowToDraw*max_texture_size));
  if( (limit_x<0) || (limit_y<0))
    {
      printf("Cube<T,U>::load__thresholded_texture_brick_fload requested col %i row %i out of range, loading 0,0\n", nColToDraw, nRowToDraw);
      nColToDraw = 0;
      nRowToDraw = 0;
      limit_x = max_texture_size;
      limit_y = max_texture_size;
    }


  wholeTextureDepth = min(max_texture_size, (int)cubeDepth);
  float* texels =(float*)( malloc(limit_x*limit_y*wholeTextureDepth*sizeof(float)));

  printf("Loading texture brick %i %i [", row, col);
  int col_offset = col*max_texture_size;
  int row_offset = row*max_texture_size;

  printf("Cube::load_texture_brick() creating texture from floats\n");
  float max_texture = -10e6;
  float min_texture = 10e6;
  for(int z = 10; z < wholeTextureDepth-10; z++)
    {
      int depth_z = z*limit_y*limit_x;
      for(int y = 0; y < limit_y; y++)
        {
          int depth_y = y*limit_x;
          for(int x = 0; x < limit_x; x++)
            {
              if(max_texture <  this->at(col_offset+x,row_offset+y,z))
                max_texture = this->at(col_offset+x,row_offset+y,z);
              if(min_texture >  this->at(col_offset+x,row_offset+y,z))
                min_texture = this->at(col_offset+x,row_offset+y,z);
            }
        }
    }

  //       printf("Cube::load_texture_brick(): max=%f and min=%f\n", max_texture, min_texture);
  int points_over = 0;
  int points_under = 0;
  int points_middle = 0;
  printf("Loading texture brick float %i %i [", row, col);
  for(int z = 10; z < wholeTextureDepth-10; z++)
    {
      int depth_z = z*limit_y*limit_x;
      for(int y = 0; y < limit_y; y++)
        {
          int depth_y = y*limit_x;
          for(int x = 0; x < limit_x; x++)
            {
              if(sizeof(T)==4){
                min_texture = threshold;
                if(this->at(col_offset+x, row_offset+y, z) < threshold){
                  texels[depth_z + depth_y + x] = 0;
                }else
                  texels[depth_z + depth_y + x] = (this->at(col_offset+x,row_offset+y,z) - min_texture)
                    / (max_texture - min_texture);
              }
              if(sizeof(T)==1){
                uchar value = 255 - this->at(col_offset+x, row_offset+y, z);
                if(value > threshold){
                  texels[depth_z + depth_y + x] = float(value - threshold)/(255-min_texture - threshold);
//                   printf("%f\n", texels[depth_z + depth_y + x]);
                }else
                  texels[depth_z + depth_y + x] =
                    0;
              }
            }
        }
      printf("#");
      fflush(stdout);
    }
  printf("] %i %i %i\n");

  #if debug
  printf("Cube::load_texture_brick_float() created the texture buffer\n");
  #endif

  glEnable(GL_TEXTURE_3D);
  glBindTexture(GL_TEXTURE_3D, wholeTexture);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP);
  glTexParameterf(GL_TEXTURE_3D,  GL_TEXTURE_PRIORITY, 1.0);
  GLfloat border_color[4];
  for(int i = 0; i < 4; i++)
    border_color[i] = 1.0;
  glTexParameterfv(GL_TEXTURE_3D, GL_TEXTURE_BORDER_COLOR, border_color);

//   printf("Blah2 %i %i %i\n", limit_x, limit_y, wholeTextureDepth);

  glBindTexture(GL_TEXTURE_3D, wholeTexture);

  glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE, limit_x, limit_y, wholeTextureDepth, 0, GL_LUMINANCE,
               GL_FLOAT, texels);

  free(texels);
}

template <class T, class U>
void Cube<T,U>::draw_layers_parallel()
{

  if(true) //draws some axes
    {
      glColor3f(1.0, 1.0, 1.0);
      glutSolidSphere(1.0, 20, 20);
      glPushMatrix();
      //Draw The Z axis
      glColor3f(0.0, 0.0, 1.0);
      glRotatef(180, 1.0, 0, 0);
      glutSolidCone(1.0, 10, 20, 20);
      glBegin(GL_LINES);
      glVertex3f(0,0,-100000);
      glVertex3f(0,0, 100000);
      glEnd();
      //Draw the x axis
      glColor3f(1.0, 0.0, 0.0);
      glRotatef(90, 0.0, 1.0, 0.0);
      glutSolidCone(1.0, 10, 20, 20);
      glBegin(GL_LINES);
      glVertex3f(0,0,-100000);
      glVertex3f(0,0, 100000);
      glEnd();
      //Draw the y axis
      glColor3f(0.0, 1.0, 0.0);
      glRotatef(90, 1.0, 0.0, 0.0);
      glutSolidCone(1.0, 10, 20, 20);
      glBegin(GL_LINES);
      glVertex3f(0,0,-100000);
      glVertex3f(0,0, 100000);
      glEnd();
      glPopMatrix();
      glColor3f(1.0, 1.0, 1.0);
    }

  glEnable(GL_BLEND);
  glBlendEquation(GL_MIN);
  glEnable(GL_TEXTURE_3D);
  glBindTexture(GL_TEXTURE_3D, wholeTexture);

  for(float z = wholeTextureDepth-1; z >= 0; z-=1)
    {
      glBegin(GL_QUADS);
      //Down left corner
      glTexCoord3f(0, 1, z/wholeTextureDepth);
      glVertex3f(-voxelWidth*cubeWidth/2, -voxelHeight*cubeHeight/2,z*voxelDepth );
      //Top left corner
      glTexCoord3f(0, 0,z/wholeTextureDepth);
      glVertex3f(-voxelWidth*cubeWidth/2, voxelHeight*cubeHeight/2,z*voxelDepth );
      //Top right corner
      glTexCoord3f(1, 0, z/wholeTextureDepth);
      glVertex3f(voxelWidth*cubeWidth/2, voxelHeight*cubeHeight/2,z*voxelDepth );
      //Bottom right corner
      glTexCoord3f(1, 1, z/wholeTextureDepth);
      glVertex3f(voxelWidth*cubeWidth/2, -voxelHeight*cubeHeight/2,z*voxelDepth );

      glEnd();
    }

  glDisable(GL_TEXTURE_3D);
  glDisable(GL_BLEND);
}

template <class T, class U>
void Cube<T,U>::draw(float rotx, float roty, float nPlanes, int min_max)
{

//   GLboolean resident[1];
//   GLboolean pepe = glAreTexturesResident(1, &wholeTextureDepth, resident);
//   if(resident[0] == GL_TRUE)
//     printf("Texture resident\n");
//   else
//     printf("Texture NOT resident\n");

  GLint max_texture_size = 0;
  glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_texture_size);

  draw_orientation_grid();

  //In object coordinates, we will draw the cube
  GLfloat widthStep = float(parentCubeWidth)*voxelWidth/2;
  GLfloat heightStep = float(parentCubeHeight)*voxelHeight/2;
  GLfloat depthStep = float(parentCubeDepth)*voxelDepth/2;

  int nColTotal = nColToDraw + colOffset;
  int nRowTotal = nRowToDraw + rowOffset;

  int end_x = min((nColTotal+1)*max_texture_size, (int)parentCubeWidth);
  int end_y = min((nRowTotal+1)*max_texture_size, (int)parentCubeHeight);

  GLfloat pModelViewMatrix[16];
  glGetFloatv(GL_MODELVIEW_MATRIX, pModelViewMatrix);

  GLfloat** cubePoints = (GLfloat**)malloc(8*sizeof(GLfloat*));;

  cubePoints[0] = create_vector(-widthStep + nColTotal*max_texture_size*voxelWidth,
                                heightStep - nRowTotal*max_texture_size*voxelHeight, -depthStep, 1.0f);

  cubePoints[1] = create_vector(-widthStep + nColTotal*max_texture_size*voxelWidth,
                                heightStep - nRowTotal*max_texture_size*voxelHeight,  depthStep, 1.0f);

  cubePoints[2] = create_vector(-widthStep + end_x*voxelWidth,
                                heightStep - nRowTotal*max_texture_size*voxelHeight,  depthStep, 1.0f);

  cubePoints[3] = create_vector(-widthStep + end_x*voxelWidth,
                                heightStep - nRowTotal*max_texture_size*voxelHeight, -depthStep, 1.0f);

  cubePoints[4] = create_vector(-widthStep + nColTotal*max_texture_size*voxelWidth,
                                heightStep - end_y*voxelHeight,
                                -depthStep, 1.0f);

  cubePoints[5] = create_vector(-widthStep + nColTotal*max_texture_size*voxelWidth,
                                heightStep - end_y*voxelHeight,  depthStep, 1.0f);

  cubePoints[6] = create_vector(-widthStep + end_x*voxelWidth,
                                heightStep - end_y*voxelHeight,  depthStep, 1.0f);

  cubePoints[7] = create_vector(-widthStep + end_x*voxelWidth,
                                heightStep - end_y*voxelHeight, -depthStep, 1.0f);

  // We will get the coordinates of the vertex of the cube in the modelview coordinates
  glLoadIdentity();
  GLfloat* cubePoints_c[8];
  glColor3f(0,0,0);
  for(int i=0; i < 8; i++)
    cubePoints_c[i] = matrix_vector_product(pModelViewMatrix, cubePoints[i]);

  //Draws the points numbers and the coordinates of the textures
  if(1){
    for(int i=0; i < 8; i++)
      {
        glColor3f(0.0,1.0,0.0);
        glPushMatrix();
        glTranslatef(cubePoints_c[i][0], cubePoints_c[i][1], cubePoints_c[i][2]);
        render_string("%i",i);
        glPopMatrix();
      }
    glPushMatrix();
    glTranslatef(cubePoints_c[0][0], cubePoints_c[0][1], cubePoints_c[0][2]);
    glRotatef(rotx, 1.0,0,0);
    glRotatef(roty, 0,1.0,0);
    //Draw The Z axis
    glColor3f(0.0, 0.0, 1.0);
    glutSolidCone(1.0, 10, 20, 20);
    glPushMatrix();
    glTranslatef(0,0,10);
    render_string("T");
    glPopMatrix();
    //Draw the x axis
    glColor3f(1.0, 0.0, 0.0);
    glRotatef(90, 0.0, 1.0, 0.0);
    glutSolidCone(1.0, 10, 20, 20);
    glPushMatrix();
    glTranslatef(0,0,10);
    render_string("R");
    glPopMatrix();
    //Draw the y axis
    glColor3f(0.0, 1.0, 0.0);
    glRotatef(90, 1.0, 0.0, 0.0);
    glutSolidCone(1.0, 10, 20, 20);
    glPushMatrix();
    glTranslatef(0,0,10);
    render_string("S");
    glPopMatrix();
    glPopMatrix();
    glColor3f(1.0, 1.0, 1.0);
  }

  //Find the closest and furthest vertex of the square
  float closest_distance = 1e9;
  float furthest_distance= 0;
  int closest_point_idx = 0;
  int furthest_point_idx = 0;
  for(int i = 0; i < 8; i++)
    {
      float dist = cubePoints_c[i][0]*cubePoints_c[i][0] + cubePoints_c[i][1]*cubePoints_c[i][1] + cubePoints_c[i][2]*cubePoints_c[i][2];
      if(dist < closest_distance)
        {
          closest_distance = dist;
          closest_point_idx = i;
        }
      if(dist > furthest_distance)
        {
          furthest_distance = dist;
          furthest_point_idx = i;
        }
    }

  //Draws a sphere in the furthest and closest point of the cube
  if(0){
    glPushMatrix();
    glTranslatef(cubePoints_c[closest_point_idx][0], cubePoints_c[closest_point_idx][1], cubePoints_c[closest_point_idx][2]);
    glColor3f(0.0,1.0,0.0);
    glutWireSphere(5,10,10);
    glPopMatrix();

    glPushMatrix();
    glTranslatef(cubePoints_c[furthest_point_idx][0], cubePoints_c[furthest_point_idx][1], cubePoints_c[furthest_point_idx][2]);
    glColor3f(0.0,0.0,1.0);
    glutWireSphere(5,10,10);
    glPopMatrix();
  }

//   printf("%f\n", cubePoints_c[furthest_point_idx][2] - cubePoints_c[closest_point_idx][2]);
  //Draws the cube
  for(float depth = 0/nPlanes; depth <= 1.0; depth+=1.0/nPlanes)
    {
      float z_plane = (cubePoints_c[furthest_point_idx][2]*(1-depth) + depth*cubePoints_c[closest_point_idx][2]);
      //Find the lines that intersect with the plane. For that we will define the lines and find the intersection of the line with the point
      GLfloat lambda_lines[12];
      if( ((cubePoints_c[1][2] > z_plane) && (cubePoints_c[0][2] > z_plane)) ||
          ((cubePoints_c[1][2] < z_plane) && (cubePoints_c[0][2] < z_plane)) )
            lambda_lines[0] = -1;
            else
            lambda_lines[0 ] = (z_plane - cubePoints_c[1][2]) / (cubePoints_c[0][2] - cubePoints_c[1][2]); //0-1

      if( ((cubePoints_c[3][2] > z_plane) && (cubePoints_c[0][2] > z_plane)) ||
          ((cubePoints_c[3][2] < z_plane) && (cubePoints_c[0][2] < z_plane)) )
            lambda_lines[1] = -1;
            else
            lambda_lines[1 ] = (z_plane - cubePoints_c[3][2]) / (cubePoints_c[0][2] - cubePoints_c[3][2]); //0-3

      if( ((cubePoints_c[4][2] > z_plane) && (cubePoints_c[0][2] > z_plane)) ||
          ((cubePoints_c[4][2] < z_plane) && (cubePoints_c[0][2] < z_plane)) )
            lambda_lines[2] = -1;
            else
            lambda_lines[2 ] = (z_plane - cubePoints_c[4][2]) / (cubePoints_c[0][2] - cubePoints_c[4][2]); //0-4

      if( ((cubePoints_c[7][2] > z_plane) && (cubePoints_c[4][2] > z_plane)) ||
          ((cubePoints_c[7][2] < z_plane) && (cubePoints_c[4][2] < z_plane)))
            lambda_lines[3] = -1;
            else
            lambda_lines[3 ] = (z_plane - cubePoints_c[7][2]) / (cubePoints_c[4][2] - cubePoints_c[7][2]); //4-7

      if( ((cubePoints_c[5][2] > z_plane) && (cubePoints_c[4][2] > z_plane)) ||
          ((cubePoints_c[5][2] < z_plane) && (cubePoints_c[4][2] < z_plane)))
            lambda_lines[4] = -1;
            else
            lambda_lines[4 ] = (z_plane - cubePoints_c[5][2]) / (cubePoints_c[4][2] - cubePoints_c[5][2]); //4-5

      if( ((cubePoints_c[1][2] > z_plane) && (cubePoints_c[2][2] > z_plane)) ||
          ((cubePoints_c[1][2] < z_plane) && (cubePoints_c[2][2] < z_plane)))
            lambda_lines[5] = -1;
            else
            lambda_lines[5 ] = (z_plane - cubePoints_c[2][2]) / (cubePoints_c[1][2] - cubePoints_c[2][2]); //1-2

      if( ((cubePoints_c[1][2] > z_plane) && (cubePoints_c[5][2] > z_plane)) ||
          ((cubePoints_c[1][2] < z_plane) && (cubePoints_c[5][2] < z_plane)))
            lambda_lines[6] = -1;
            else
            lambda_lines[6 ] = (z_plane - cubePoints_c[5][2]) / (cubePoints_c[1][2] - cubePoints_c[5][2]); //1-5

      if( ((cubePoints_c[6][2] > z_plane) && (cubePoints_c[5][2] > z_plane)) ||
          ((cubePoints_c[6][2] < z_plane) && (cubePoints_c[5][2] < z_plane)))
            lambda_lines[7] = -1;
            else
            lambda_lines[7 ] = (z_plane - cubePoints_c[6][2]) / (cubePoints_c[5][2] - cubePoints_c[6][2]); //5-6

      if( ((cubePoints_c[2][2] > z_plane) && (cubePoints_c[3][2] > z_plane)) ||
          ((cubePoints_c[2][2] < z_plane) && (cubePoints_c[3][2] < z_plane)))
            lambda_lines[8] = -1;
            else
            lambda_lines[8 ] = (z_plane - cubePoints_c[2][2]) / (cubePoints_c[3][2] - cubePoints_c[2][2]); //3-2

      if( ((cubePoints_c[7][2] > z_plane) && (cubePoints_c[3][2] > z_plane)) ||
          ((cubePoints_c[7][2] < z_plane) && (cubePoints_c[3][2] < z_plane)))
            lambda_lines[9] = -1;
            else
            lambda_lines[9 ] = (z_plane - cubePoints_c[7][2]) / (cubePoints_c[3][2] - cubePoints_c[7][2]); //3-7

      if( ((cubePoints_c[6][2] > z_plane) && (cubePoints_c[7][2] > z_plane)) ||
          ((cubePoints_c[6][2] < z_plane) && (cubePoints_c[7][2] < z_plane)))
            lambda_lines[10] = -1;
            else
            lambda_lines[10] = (z_plane - cubePoints_c[6][2]) / (cubePoints_c[7][2] - cubePoints_c[6][2]); //7-6

      if( ((cubePoints_c[2][2] > z_plane) && (cubePoints_c[6][2] > z_plane)) ||
          ((cubePoints_c[2][2] < z_plane) && (cubePoints_c[6][2] < z_plane)))
            lambda_lines[11] = -1;
            else
            lambda_lines[11] = (z_plane - cubePoints_c[2][2]) / (cubePoints_c[6][2] - cubePoints_c[2][2]); //6-2

      // We will store the point and texture coordinates of the points that we will draw afterwards
      //There is at maximum five intersections -> therefore we will define an array of five points
      GLfloat intersectionPoints[5][6];
      int intersectionPointsIdx = 0;
      for(int i = 0; i < 12; i++)
        {
          if( (lambda_lines[i] > 0) && (lambda_lines[i] < 1))
            {
              float x_point = 0;
              float y_point = 0;
              float z_point = 0;
              float r_point = 0;
              float s_point = 0;
              float t_point = 0;
              switch(i)
                {
                case 0: //0-1
                  x_point = cubePoints_c[0][0]*lambda_lines[i] + cubePoints_c[1][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[0][1]*lambda_lines[i] + cubePoints_c[1][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[0][2]*lambda_lines[i] + cubePoints_c[1][2]*(1-lambda_lines[i]);
                  r_point = 0;
                  s_point = 0;
                  t_point = (1-lambda_lines[i]);
                  break;
                case 1: //0-3
                  x_point = cubePoints_c[0][0]*lambda_lines[i] + cubePoints_c[3][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[0][1]*lambda_lines[i] + cubePoints_c[3][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[0][2]*lambda_lines[i] + cubePoints_c[3][2]*(1-lambda_lines[i]);
                  r_point = 1-lambda_lines[i];
                  s_point = 0;
                  t_point = 0;
                  break;
                case 2: //0-4
                  x_point = cubePoints_c[0][0]*lambda_lines[i] + cubePoints_c[4][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[0][1]*lambda_lines[i] + cubePoints_c[4][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[0][2]*lambda_lines[i] + cubePoints_c[4][2]*(1-lambda_lines[i]);
                  r_point = 0;
                  s_point = 1-lambda_lines[i];
                  t_point = 0;
                  break;
                case 3: //4-7
                  x_point = cubePoints_c[4][0]*lambda_lines[i] + cubePoints_c[7][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[4][1]*lambda_lines[i] + cubePoints_c[7][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[4][2]*lambda_lines[i] + cubePoints_c[7][2]*(1-lambda_lines[i]);
                  r_point = 1-lambda_lines[i];
                  s_point = 1;
                  t_point = 0;
                  break;
                case 4: //4-5
                  x_point = cubePoints_c[4][0]*lambda_lines[i] + cubePoints_c[5][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[4][1]*lambda_lines[i] + cubePoints_c[5][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[4][2]*lambda_lines[i] + cubePoints_c[5][2]*(1-lambda_lines[i]);
                  r_point = 0;
                  s_point = 1;
                  t_point = 1-lambda_lines[i];
                  break;
                case 5: //1-2
                  x_point = cubePoints_c[1][0]*lambda_lines[i] + cubePoints_c[2][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[1][1]*lambda_lines[i] + cubePoints_c[2][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[1][2]*lambda_lines[i] + cubePoints_c[2][2]*(1-lambda_lines[i]);
                  r_point = 1-lambda_lines[i];
                  s_point = 0;
                  t_point = 1;
                  break;
                case 6: //1-5
                  x_point = cubePoints_c[1][0]*lambda_lines[i] + cubePoints_c[5][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[1][1]*lambda_lines[i] + cubePoints_c[5][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[1][2]*lambda_lines[i] + cubePoints_c[5][2]*(1-lambda_lines[i]);
                  r_point = 0;
                  s_point = 1-lambda_lines[i];
                  t_point = 1;
                  break;
                case 7: //5-6
                  x_point = cubePoints_c[5][0]*lambda_lines[i] + cubePoints_c[6][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[5][1]*lambda_lines[i] + cubePoints_c[6][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[5][2]*lambda_lines[i] + cubePoints_c[6][2]*(1-lambda_lines[i]);
                  r_point = 1-lambda_lines[i];
                  s_point = 1;
                  t_point = 1;
                  break;
                case 8: //3-2
                  x_point = cubePoints_c[3][0]*lambda_lines[i] + cubePoints_c[2][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[3][1]*lambda_lines[i] + cubePoints_c[2][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[3][2]*lambda_lines[i] + cubePoints_c[2][2]*(1-lambda_lines[i]);
                  r_point = 1;
                  s_point = 0;
                  t_point = 1-lambda_lines[i];
                  break;
                case 9: //3-7
                  x_point = cubePoints_c[3][0]*lambda_lines[i] + cubePoints_c[7][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[3][1]*lambda_lines[i] + cubePoints_c[7][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[3][2]*lambda_lines[i] + cubePoints_c[7][2]*(1-lambda_lines[i]);
                  r_point = 1;
                  s_point = 1-lambda_lines[i];
                  t_point = 0;
                  break;
                case 10: //7-6
                  x_point = cubePoints_c[7][0]*lambda_lines[i] + cubePoints_c[6][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[7][1]*lambda_lines[i] + cubePoints_c[6][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[7][2]*lambda_lines[i] + cubePoints_c[6][2]*(1-lambda_lines[i]);
                  r_point = 1;
                  s_point = 1;
                  t_point = 1-lambda_lines[i];
                  break;
                case 11: //6-2
                  x_point = cubePoints_c[6][0]*lambda_lines[i] + cubePoints_c[2][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[6][1]*lambda_lines[i] + cubePoints_c[2][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[6][2]*lambda_lines[i] + cubePoints_c[2][2]*(1-lambda_lines[i]);
                  r_point = 1;
                  s_point = lambda_lines[i];
                  t_point = 1;
                  break;
                }
              intersectionPoints[intersectionPointsIdx][0] = x_point;
              intersectionPoints[intersectionPointsIdx][1] = y_point;
              intersectionPoints[intersectionPointsIdx][2] = z_point;
              intersectionPoints[intersectionPointsIdx][3] = r_point;
              intersectionPoints[intersectionPointsIdx][4] = s_point;
              intersectionPoints[intersectionPointsIdx][5] = t_point;
              intersectionPointsIdx++;

              //Draws spheres in the intersection points
              if(0){
                glPushMatrix();
                glTranslatef(x_point, y_point, z_point);
                glutWireSphere(5,10,10);
                glPopMatrix();
              }
            }
        }

      //Find the average of the position
      GLfloat x_average = 0;
      GLfloat y_average = 0;
      for(int i = 0; i < intersectionPointsIdx; i++)
        {
          x_average += intersectionPoints[i][0];
          y_average += intersectionPoints[i][1];
        }
      x_average = x_average / intersectionPointsIdx;
      y_average = y_average / intersectionPointsIdx;

      //Rank the points according to their angle (to display them in order)
      GLfloat points_angles[intersectionPointsIdx];
      for(int i = 0; i < intersectionPointsIdx; i++)
        {
          points_angles[i] = atan2(intersectionPoints[i][1]-y_average, intersectionPoints[i][0]-x_average);
          if(points_angles[i] < 0)
            points_angles[i] = points_angles[i] + 2*3.14159;
        }
      int indexes[intersectionPointsIdx];
      for(int i = 0; i < intersectionPointsIdx; i++)
        {
          GLfloat min_angle = 1e3;
          int min_index = 15;
          for(int j = 0; j < intersectionPointsIdx; j++)
            {
              if(points_angles[j] < min_angle)
                {
                  min_angle = points_angles[j];
                  min_index = j;
                }
            }
          indexes[i] = min_index;
          points_angles[min_index] = 1e3;
        }

      if(min_max==0)
        glColor3f(1.0,1.0,1.0);
      if(min_max==1)
        glColor3f(0.0,0.0,1.0);
      if(min_max==2)
        glColor3f(0.0,1.0,0.0);
      if(min_max==3)
        glColor3f(1.0,1.0,1.0);


      glEnable(GL_BLEND);
      if(min_max == 0)
        glBlendEquation(GL_MIN);
      else
        glBlendEquation(GL_MAX);

      glEnable(GL_TEXTURE_3D);
      glBindTexture(GL_TEXTURE_3D, wholeTexture);

      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

      //All the previous was preparation, here with draw the poligon
      glBegin(GL_POLYGON);
      for(int i = 0; i < intersectionPointsIdx; i++)
        {
          glTexCoord3f(intersectionPoints[indexes[i]][3],intersectionPoints[indexes[i]][4],intersectionPoints[indexes[i]][5]);
          glVertex3f(intersectionPoints[indexes[i]][0],intersectionPoints[indexes[i]][1],intersectionPoints[indexes[i]][2]);
        }
      glEnd();

      glDisable(GL_TEXTURE_3D);
      glDisable(GL_BLEND);

      //Draws an sphere on all the intersection points
      if(false)
        {
          glColor3f(0.0,1.0,1.0);
          for(int i = 0; i < intersectionPointsIdx; i++)
            {
              glPushMatrix();
              glTranslatef(intersectionPoints[indexes[i]][0],intersectionPoints[indexes[i]][1],intersectionPoints[indexes[i]][2]);
              glutSolidSphere(1,5,5);
              glPopMatrix();
            }
        }

      //Draws the texture coordinates of the intersection points
      if(false)
        {
          glColor3f(0.0,0.0,0.0);
          for(int i = 0; i < intersectionPointsIdx; i++)
            {
              glPushMatrix();
              glTranslatef(intersectionPoints[indexes[i]][0],intersectionPoints[indexes[i]][1],intersectionPoints[indexes[i]][2]);
              render_string("(%.2f %.2f %.2f)", intersectionPoints[indexes[i]][3],intersectionPoints[indexes[i]][4],intersectionPoints[indexes[i]][5]);
              glPopMatrix();
            }
          glColor3f(1.0,1.0,1.0);
         }
    } //depth loop

  //Put back the modelView matrix
  glMultMatrixf(pModelViewMatrix);
}

template <class T, class U>
void Cube<T,U>::draw_layer_tile_XY(float nLayerToDraw, int color)
{
//   if((nLayerToDraw < 0)||(nLayerToDraw > cubeDepth))
//     {
//       printf("Cube::draw_layer: invalid nLayerToDraw %i\n", nLayerToDraw);
//       return;
//     }

//   draw_orientation_grid();
  glEnable(GL_TEXTURE_3D);
  glBindTexture(GL_TEXTURE_3D, wholeTexture);
  GLint max_texture_size = 0;
  glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_texture_size);

  //In object coordinates, we will draw the cube
  GLfloat widthStep = float(parentCubeWidth)*voxelWidth/2;
  GLfloat heightStep = float(parentCubeHeight)*voxelHeight/2;
  GLfloat depthStep = float(parentCubeDepth)*voxelDepth/2;

  GLfloat increment_height = min(float(cubeHeight - nRowToDraw*max_texture_size), float(max_texture_size));
  increment_height = increment_height / max_texture_size;
  GLfloat increment_width = min(float(cubeWidth - nColToDraw*max_texture_size), float(max_texture_size));
  increment_width = increment_width / max_texture_size;

  if(color == 0)
    glColor3f(1.0,1.0,1.0);
  else
    glColor3f(0.0,0.0,1.0);
  glBegin(GL_QUADS);
    glTexCoord3f(0,0,nLayerToDraw/(cubeDepth-1));
    glVertex3f(-widthStep + nColToDraw*max_texture_size*voxelWidth,
               heightStep - nRowToDraw*max_texture_size*voxelHeight,
               -depthStep + nLayerToDraw*voxelDepth);
    glTexCoord3f(1,0,nLayerToDraw/(cubeDepth-1));
    glVertex3f(-widthStep + (nColToDraw+increment_width)*max_texture_size*voxelWidth,
               heightStep - nRowToDraw*max_texture_size*voxelHeight,
               -depthStep + nLayerToDraw*voxelDepth);
    glTexCoord3f(1,1,nLayerToDraw/(cubeDepth-1));
    glVertex3f(-widthStep + (nColToDraw+increment_width)*max_texture_size*voxelWidth,
               heightStep - (nRowToDraw+increment_height)*max_texture_size*voxelHeight,
               -depthStep + nLayerToDraw*voxelDepth);
    glTexCoord3f(0,1,nLayerToDraw/(cubeDepth-1));
    glVertex3f(-widthStep + nColToDraw*max_texture_size*voxelWidth,
               heightStep - (nRowToDraw+increment_height)*max_texture_size*voxelHeight,
               -depthStep + nLayerToDraw*voxelDepth);
  glEnd();
  glColor3f(0.0,0.0,1.0);
  glBegin(GL_LINE_LOOP);
    glVertex3f(-widthStep + nColToDraw*max_texture_size*voxelWidth,
               heightStep - nRowToDraw*max_texture_size*voxelHeight,
               -depthStep + nLayerToDraw*voxelDepth);
    glVertex3f(-widthStep + (nColToDraw+increment_width)*max_texture_size*voxelWidth,
               heightStep - nRowToDraw*max_texture_size*voxelHeight,
               -depthStep + nLayerToDraw*voxelDepth);
    glVertex3f(-widthStep + (nColToDraw+increment_width)*max_texture_size*voxelWidth,
               heightStep - (nRowToDraw+increment_height)*max_texture_size*voxelHeight,
               -depthStep + nLayerToDraw*voxelDepth);
    glVertex3f(-widthStep + nColToDraw*max_texture_size*voxelWidth,
               heightStep - (nRowToDraw+increment_height)*max_texture_size*voxelHeight,
               -depthStep + nLayerToDraw*voxelDepth);
  glEnd();
  glDisable(GL_TEXTURE_3D);
  //Draws the coordinates

}

template <class T, class U>
void Cube<T,U>::draw_layer_tile_XZ(float nLayerToDraw)
{
//  if((nLayerToDraw < 0)||(nLayerToDraw > cubeDepth))
//     {
//       printf("Cube::draw_layer: invalid nLayerToDraw %i\n", nLayerToDraw);
//       return;
//     }

//   draw_orientation_grid();

  glEnable(GL_TEXTURE_3D);
  glBindTexture(GL_TEXTURE_3D, wholeTexture);
  GLint max_texture_size = 0;
  glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_texture_size);

  //In object coordinates, we will draw the cube
  GLfloat widthStep = float(parentCubeWidth)*voxelWidth/2;
  GLfloat heightStep = float(parentCubeHeight)*voxelHeight/2;
  GLfloat depthStep = float(parentCubeDepth)*voxelDepth/2;

  GLfloat depth_texture = nLayerToDraw/max_texture_size;

  glBegin(GL_QUADS);
  glTexCoord3f(0, depth_texture, 0);
  glVertex3f(-widthStep + nColToDraw*max_texture_size*voxelWidth,
             heightStep - nRowToDraw*max_texture_size*voxelHeight - depth_texture*max_texture_size*voxelHeight,
             -depthStep);
  glTexCoord3f(0, depth_texture, 1);
  glVertex3f(-widthStep + nColToDraw*max_texture_size*voxelWidth,
             heightStep - nRowToDraw*max_texture_size*voxelHeight - depth_texture*max_texture_size*voxelHeight,
             depthStep);
  glTexCoord3f(1, depth_texture, 1);
  glVertex3f(-widthStep + (nColToDraw+1)*max_texture_size*voxelWidth,
             heightStep - nRowToDraw*max_texture_size*voxelHeight - depth_texture*max_texture_size*voxelHeight,
             depthStep);
  glTexCoord3f(1, depth_texture, 0);
  glVertex3f(-widthStep + (nColToDraw+1)*max_texture_size*voxelWidth,
             heightStep - nRowToDraw*max_texture_size*voxelHeight - depth_texture*max_texture_size*voxelHeight,
             -depthStep);
  glEnd();
  glColor3f(0.0,1.0,1.0);
  glBegin(GL_LINE_LOOP);
  glVertex3f(-widthStep + nColToDraw*max_texture_size*voxelWidth,
             heightStep - nRowToDraw*max_texture_size*voxelHeight - depth_texture*max_texture_size*voxelHeight,
             -depthStep);
  glVertex3f(-widthStep + nColToDraw*max_texture_size*voxelWidth,
             heightStep - nRowToDraw*max_texture_size*voxelHeight - depth_texture*max_texture_size*voxelHeight,
             depthStep);
  glVertex3f(-widthStep + (nColToDraw+1)*max_texture_size*voxelWidth,
             heightStep - nRowToDraw*max_texture_size*voxelHeight - depth_texture*max_texture_size*voxelHeight,
             depthStep);
  glVertex3f(-widthStep + (nColToDraw+1)*max_texture_size*voxelWidth,
             heightStep - nRowToDraw*max_texture_size*voxelHeight - depth_texture*max_texture_size*voxelHeight,
             -depthStep);
  glEnd();

  glDisable(GL_TEXTURE_3D);
}

template <class T, class U>
void Cube<T,U>::draw_layer_tile_YZ(float nLayerToDraw)
{
//  if((nLayerToDraw < 0)||(nLayerToDraw > cubeDepth))
//     {
//       printf("Cube::draw_layer: invalid nLayerToDraw %i\n", nLayerToDraw);
//       return;
//     }

//   draw_orientation_grid();

  glEnable(GL_TEXTURE_3D);
  glBindTexture(GL_TEXTURE_3D, wholeTexture);
  GLint max_texture_size = 0;
  glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_texture_size);

  //In object coordinates, we will draw the cube
  GLfloat widthStep = float(parentCubeWidth)*voxelWidth/2;
  GLfloat heightStep = float(parentCubeHeight)*voxelHeight/2;
  GLfloat depthStep = float(parentCubeDepth)*voxelDepth/2;

  GLfloat depth_texture = nLayerToDraw/max_texture_size;

  glBegin(GL_QUADS);
  glTexCoord3f(depth_texture, 0, 0);
  glVertex3f(-widthStep + (nColToDraw + depth_texture)*max_texture_size*voxelWidth,
             heightStep - nRowToDraw*max_texture_size*voxelHeight,
             -depthStep);
  glTexCoord3f(depth_texture, 1, 0);
  glVertex3f(-widthStep + (nColToDraw + depth_texture)*max_texture_size*voxelWidth,
             heightStep - (nRowToDraw+1)*max_texture_size*voxelHeight,
             -depthStep);
  glTexCoord3f(depth_texture, 1, 1);
  glVertex3f(-widthStep + (nColToDraw + depth_texture)*max_texture_size*voxelWidth,
             heightStep - (nRowToDraw+1)*max_texture_size*voxelHeight,
             depthStep);
  glTexCoord3f(depth_texture, 0, 1);
  glVertex3f(-widthStep + (nColToDraw + depth_texture)*max_texture_size*voxelWidth,
             heightStep - nRowToDraw*max_texture_size*voxelHeight,
             depthStep);
  glEnd();
  glColor3f(1.0,0.0,1.0);
  glBegin(GL_LINE_LOOP);
  glVertex3f(-widthStep + (nColToDraw + depth_texture)*max_texture_size*voxelWidth,
             heightStep - nRowToDraw*max_texture_size*voxelHeight,
             -depthStep);
  glVertex3f(-widthStep + (nColToDraw + depth_texture)*max_texture_size*voxelWidth,
             heightStep - (nRowToDraw+1)*max_texture_size*voxelHeight,
             -depthStep);
  glVertex3f(-widthStep + (nColToDraw + depth_texture)*max_texture_size*voxelWidth,
             heightStep - (nRowToDraw+1)*max_texture_size*voxelHeight,
             depthStep);
  glVertex3f(-widthStep + (nColToDraw + depth_texture)*max_texture_size*voxelWidth,
             heightStep - nRowToDraw*max_texture_size*voxelHeight,
             depthStep);
  glEnd();


  glDisable(GL_TEXTURE_3D);
}


template <class T, class U>
void Cube<T,U>::render_string(const char* format, ...)
{
 va_list args;
 char    buffer[512];
 va_start(args,format);
 vsnprintf(buffer,sizeof(buffer)-1,format,args);
 va_end(args);
 void *font = GLUT_BITMAP_8_BY_13;
 glRasterPos2f(-1,-1);
 for (const char *c=buffer; *c != '\0'; c++) {
   glutBitmapCharacter(font, *c);
 }
}

template <class T, class U>
void Cube<T,U>::draw_orientation_grid(bool include_split)
{
  GLint max_texture_size = 0;
  glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_texture_size);

  //In object coordinates, we will draw the cube
  GLfloat widthStep = float(cubeWidth)*voxelWidth/2;
  GLfloat heightStep = float(cubeHeight)*voxelHeight/2;
  GLfloat depthStep = float(cubeDepth)*voxelDepth/2;

  if(include_split) //draws the OpenGL coordinates
    {
      glColor3f(1.0, 1.0, 1.0);
      glutSolidSphere(1.0, 20, 20);
      glPushMatrix();
      //Draw The Z axis
      glColor3f(0.0, 0.0, 1.0);
      glRotatef(180, 1.0, 0, 0);
      glutSolidCone(1.0, 10, 20, 20);
      glPushMatrix();
      glTranslatef(0,0,10);
      render_string("Z");
      glPopMatrix();
      //Draw the x axis
      glColor3f(1.0, 0.0, 0.0);
      glRotatef(90, 0.0, 1.0, 0.0);
      glutSolidCone(1.0, 10, 20, 20);
      glPushMatrix();
      glTranslatef(0,0,10);
      render_string("X");
      glPopMatrix();
      //Draw the y axis
      glColor3f(0.0, 1.0, 0.0);
      glRotatef(90, 1.0, 0.0, 0.0);
      glutSolidCone(1.0, 10, 20, 20);
      glPopMatrix();
      glPushMatrix();
      glTranslatef(0,10,0);
      render_string("Y");
      glPopMatrix();
      glColor3f(0.0,0.0,0.0);
      //Draws the coordinates
      glPushMatrix();
      glTranslatef(-widthStep,heightStep+5,-depthStep);
      for(float x = -widthStep; x <= widthStep+1; x+=max_texture_size*voxelWidth)
        {
          render_string("%.2f", x);
          glTranslatef(max_texture_size*voxelWidth,0,0);
        }
      glPopMatrix();
      glPushMatrix();
      glTranslatef(-widthStep-20,heightStep,-depthStep);
      for(float y = heightStep; y >= -heightStep; y-=max_texture_size*voxelHeight)
        {
          render_string("%.2f", y);
          glTranslatef(0,-max_texture_size*voxelHeight,0);
        }
      glPopMatrix();
      glColor3f(1.0, 1.0, 1.0);
    }

  glMatrixMode(GL_MODELVIEW);
//   glColor3f(0.0,0.0,0.0);
  glColor3f(0.0,0.0,0.0);
  glBegin(GL_LINE_STRIP);
  glVertex3f(-widthStep,  heightStep, -depthStep); //0
  glVertex3f(-widthStep,  heightStep,  depthStep); //1
  glVertex3f( widthStep,  heightStep,  depthStep); //2
  glVertex3f( widthStep, -heightStep,  depthStep); //6
  glVertex3f( widthStep, -heightStep, -depthStep); //7
  glVertex3f(-widthStep, -heightStep, -depthStep); //4
  glVertex3f(-widthStep,  heightStep, -depthStep); //0
  glVertex3f(-widthStep, -heightStep, -depthStep); //4
  glVertex3f(-widthStep, -heightStep, +depthStep); //5
  glVertex3f(-widthStep, +heightStep, +depthStep); //1
  glVertex3f(-widthStep, -heightStep, +depthStep); //5
  glVertex3f( widthStep, -heightStep,  depthStep); //6
  glVertex3f( widthStep,  heightStep,  depthStep); //2
  glVertex3f( widthStep,  heightStep, -depthStep); //3
  glVertex3f( widthStep, -heightStep, -depthStep); //7
  glVertex3f( widthStep,  heightStep, -depthStep); //3
  glVertex3f(-widthStep,  heightStep, -depthStep); //0
  glEnd();

  //Draws a grid for orientation purposes
  if(include_split)
    {
      glColor3f(1.0,0.0,0.0);
      for(float i = -widthStep; i <= widthStep; i+= voxelWidth*512)
        {
          glBegin(GL_LINES);
          glVertex3f(i, heightStep, -depthStep);
          glVertex3f(i, -heightStep, -depthStep);
          glEnd();
        }
      for(float i = heightStep; i >= -heightStep; i-= voxelHeight*512)
        {
          glBegin(GL_LINES);
          glVertex3f(-widthStep, i, -depthStep);
          glVertex3f(widthStep, i, -depthStep);
          glEnd();
        }
    }
  glColor3f(1.0,1.0,1.0);

}


//############ CUBE BOOSTING ####################

template <class T, class U>
void Cube<T,U>::apply_boosting_to_brick(string name_classifier, int row, int col, Cube* output)
{
  //FIXME!!
  GLint max_texture_size = 512;

  vector< float* > global_classifier = load_boosting_haar_classifier(name_classifier);

  printf("apply_boosting_to_brick:: applying %s to the brick %i %i\n", name_classifier.c_str(), row, col);
  printf("[");
  fflush(stdout);

  //Erases the places where we will have no information
  for(int z = 0; z < 3; z++)
    for(int y = 0; y < max_texture_size; y++)
      for(int x = 0; x < max_texture_size; x++)
        output->voxels[z][y][x] = 0;

  for(int z = cubeDepth-3; z < cubeDepth; z++)
    for(int y = 0; y < max_texture_size; y++)
      for(int x = 0; x < max_texture_size; x++)
        output->voxels[z][y][x] = 0;


  int z;
  #pragma omp parallel for private(x,y)
  for(z = 3; z < cubeDepth-3; z++){
//   for(z = 3; z < 4; z++){
    for(int y = max_texture_size*row; y < (row+1)*max_texture_size; y ++){
      for(int x = max_texture_size*col; x < (col+1)*max_texture_size; x++){
        output->voxels[z][y][x] = classify_voxel(global_classifier, x, y, z);
      }
    }
    if(z%5==0) {
      printf("#");
      fflush(stdout);
    }
  }
  printf("]\n");
}

template <class T, class U>
GLuint Cube<T,U>::classify_voxel(vector< float* >& global_classifier, int x, int y, int z)
{
//   printf("Here\n");

  float resp = 0;
  ulong response_w = 0;
  float weak_classification_var = 0;
  ulong first_volume = 0;
  ulong second_volume = 0;

  int i = 0;
  for(i = 0; i < global_classifier.size(); i++)
    {

      first_volume =
        integral_volume_at(x-16+(int)global_classifier[i][3],y-16+(int)global_classifier[i][4],z-3+(int)global_classifier[i][5]) - //point 8
        integral_volume_at(x-16+(int)global_classifier[i][3],y-16+(int)global_classifier[i][4],z-3+(int)global_classifier[i][2]) - //point 3
        integral_volume_at(x-16+(int)global_classifier[i][3],y-16+(int)global_classifier[i][1],z-3+(int)global_classifier[i][5]) - //point 7
        integral_volume_at(x-16+(int)global_classifier[i][0],y-16+(int)global_classifier[i][4],z-3+(int)global_classifier[i][5]) - //point 5
        integral_volume_at(x-16+(int)global_classifier[i][0],y-16+(int)global_classifier[i][1],z-3+(int)global_classifier[i][2]) + //point 1
        integral_volume_at(x-16+(int)global_classifier[i][0],y-16+(int)global_classifier[i][1],z-3+(int)global_classifier[i][5]) + //point 6
        integral_volume_at(x-16+(int)global_classifier[i][3],y-16+(int)global_classifier[i][1],z-3+(int)global_classifier[i][2]) + //point 2
        integral_volume_at(x-16+(int)global_classifier[i][0],y-16+(int)global_classifier[i][4],z-3+(int)global_classifier[i][2]);  //point 4


      second_volume =
        integral_volume_at(x-16+(int)global_classifier[i][9],y-16+(int)global_classifier[i][10],z-3+(int)global_classifier[i][11]) - //point 8
        integral_volume_at(x-16+(int)global_classifier[i][9],y-16+(int)global_classifier[i][10],z-3+(int)global_classifier[i][8 ]) - //point 3
        integral_volume_at(x-16+(int)global_classifier[i][9],y-16+(int)global_classifier[i][7 ],z-3+(int)global_classifier[i][11]) - //point 7
        integral_volume_at(x-16+(int)global_classifier[i][6],y-16+(int)global_classifier[i][10],z-3+(int)global_classifier[i][11]) - //point 5
        integral_volume_at(x-16+(int)global_classifier[i][6],y-16+(int)global_classifier[i][7 ],z-3+(int)global_classifier[i][8 ]) + //point 1
        integral_volume_at(x-16+(int)global_classifier[i][6],y-16+(int)global_classifier[i][7 ],z-3+(int)global_classifier[i][11]) + //point 6
        integral_volume_at(x-16+(int)global_classifier[i][9],y-16+(int)global_classifier[i][7 ],z-3+(int)global_classifier[i][8 ]) + //point 2
        integral_volume_at(x-16+(int)global_classifier[i][6],y-16+(int)global_classifier[i][10],z-3+(int)global_classifier[i][8 ]);  //point 4


      response_w = first_volume - second_volume;

      if (first_volume > second_volume)
        {
          weak_classification_var = 1;
        }
      else
        {
          weak_classification_var = -1;
        }

      resp = resp  +global_classifier[i][13]*weak_classification_var;   //  bal_classifier[i][13]*weak_classification_var;
    }

  if(resp > 0){
    return 255;
  }
  else
    return 0;
}

template <class T, class U>
vector< float*> Cube<T,U>::load_boosting_haar_classifier(string classifiers_file)
{
  vector< float* > global_classifier;

  std::ifstream in(classifiers_file.c_str());
  if(!in.good())
    {
      printf("The file %s can not be opened\n",classifiers_file.c_str());
      exit(0);
    }

  while(!in.eof())
    {
      float* parameters = (float*)malloc(14*sizeof(float));
      for(int i = 0; i < 14; i++){
        in >> parameters[i];
      }
      global_classifier.push_back(parameters);
    }
  in.close();

  global_classifier.pop_back();

//   printf("The global_classifier has %i weak learners\n", global_classifier.size());
//   for(int i2 = 0; i2 < global_classifier.size(); i2+=200) {
//     printf("%i ",i2);
//     for(int j = 0; j < 14; j++)
//       printf("%f ", global_classifier[i2][j]);
//     printf("\n");
//   }

  return global_classifier;
}

//####### CUBE MATH ###################################################################

template <class T, class U>
GLfloat* Cube<T,U>::invert_matrix(GLfloat* a)
{
  GLfloat* b = (GLfloat*) malloc(16*sizeof(GLfloat));

  double t14 = a[0]*a[5];
  double t15 = a[10]*a[15];
  double t17 = a[14]*a[11];
  double t19 = a[0]*a[6];
  double t20 = a[9]*a[15];
  double t22 = a[13]*a[11];
  double t24 = a[0]*a[7];
  double t25 = a[9]*a[14];
  double t27 = a[13]*a[10];
  double t29 = a[1]*a[4];
  double t32 = a[1]*a[6];
  double t33 = a[8]*a[15];
  double t35 = a[12]*a[11];
  double t37 = a[1]*a[7];
  double t38 = a[8]*a[14];
  double t40 = a[12]*a[10];
  double t42 = t14*t15-t14*t17-t19*t20+t19*t22+t24*t25-t24*t27-t29*t15+t29*t17+t32
    *t33-t32*t35-t37*t38+t37*t40;
  double t43 = a[2]*a[4];
  double t46 = a[2]*a[5];
  double t49 = a[2]*a[7];
  double t50 = a[8]*a[13];
  double t52 = a[12]*a[9];
  double t54 = a[3]*a[4];
  double t57 = a[3]*a[5];
  double t60 = a[3]*a[6];
  double t63 = t43*t20-t43*t22-t46*t33+t46*t35+t49*t50-t49*t52-t54*t25+t54*t27+t57
    *t38-t57*t40-t60*t50+t60*t52;
  double t65 = 1/(t42+t63);
  double t71 = a[8]*a[6];
  double t73 = a[12]*a[6];
  double t75 = a[8]*a[7];
  double t77 = a[12]*a[7];
  double t81 = a[4]*a[9];
  double t83 = a[4]*a[13];
  double t85 = a[8]*a[5];
  double t87 = a[12]*a[5];
  double t101 = a[1]*a[10];
  double t103 = a[1]*a[14];
  double t105 = a[2]*a[9];
  double t107 = a[2]*a[13];
  double t109 = a[3]*a[9];
  double t111 = a[3]*a[13];
  double t115 = a[0]*a[10];
  double t117 = a[0]*a[14];
  double t119 = a[2]*a[8];
  double t121 = a[2]*a[12];
  double t123 = a[3]*a[8];
  double t125 = a[3]*a[12];
  double t129 = a[0]*a[9];
  double t131 = a[0]*a[13];
  double t133 = a[1]*a[8];
  double t135 = a[1]*a[12];

  //b will be returned in the opengl ordening
  b[0] = (a[5]*a[10]*a[15]-a[5]*a[14]*a[11]-a[6]*a[9]*a[15]+a[6]*a[13]*a[11]+a[7]*a[9]*a[14]-a[7]*a[13]*
          a[10])*t65;

  b[4] = -(a[4]*a[10]*a[15]-a[4]*a[14]*a[11]-t71*a[15]+t73*a[11]+t75*a[14]-t77*a[10])*t65;
  b[8] = (t81*a[15]-t83*a[11]-t85*a[15]+t87*a[11]+t75*a[13]-t77*a[9])*t65;
  b[12] = -(t81*a[14]-t83*a[10]-t85*a[14]+t87*a[10]+t71*a[13]-t73*a[9])*t65;
  b[1] = -(t101*a[15]-t103*a[11]-t105*a[15]+t107*a[11]+t109*a[14]-t111*a[10])*t65;
  b[5] = (t115*a[15]-t117*a[11]-t119*a[15]+t121*a[11]+t123*a[14]-t125*a[10])*t65;
  b[9] = -(t129*a[15]-t131*a[11]-t133*a[15]+t135*a[11]+t123*a[13]-t125*a[9])*t65;
  b[13] = (t129*a[14]-t131*a[10]-t133*a[14]+t135*a[10]+t119*a[13]-t121*a[9])*t65;
  b[2] = (t32*a[15]-t103*a[7]-t46*a[15]+t107*a[7]+t57*a[14]-t111*a[6])*t65;
  b[6] = -(t19*a[15]-t117*a[7]-t43*a[15]+t121*a[7]+t54*a[14]-t125*a[6])*t65;
  b[10] = (t14*a[15]-t131*a[7]-t29*a[15]+t135*a[7]+t54*a[13]-t125*a[5])*t65;
  b[14] = -(t14*a[14]-t131*a[6]-t29*a[14]+t135*a[6]+t43*a[13]-t121*a[5])*t65;
  b[3] = -(t32*a[11]-t101*a[7]-t46*a[11]+t105*a[7]+t57*a[10]-t109*a[6])*t65;
  b[7] = (t19*a[11]-t115*a[7]-t43*a[11]+t119*a[7]+t54*a[10]-t123*a[6])*t65;
  b[11] = -(t14*a[11]-t129*a[7]-t29*a[11]+t133*a[7]+t54*a[9]-t123*a[5])*t65;
  b[15] = (t14*a[10]-t129*a[6]-t29*a[10]+t133*a[6]+t43*a[9]-t119*a[5])*t65;

  return b;
}

template <class T, class U>
GLfloat* Cube<T,U>::matrix_vector_product(GLfloat* m, GLfloat* v)
{
  GLfloat* b = (GLfloat*)malloc(4*sizeof(GLfloat));
  b[3] =  m[3]*v[0] + m[7]*v[1] + m[11]*v[2] + m[15]*v[3];
  b[0] = (m[0]*v[0] + m[4]*v[1] + m[8 ]*v[2] + m[12]*v[3])/b[3];
  b[1] = (m[1]*v[0] + m[5]*v[1] + m[9 ]*v[2] + m[13]*v[3])/b[3];
  b[2] = (m[2]*v[0] + m[6]*v[1] + m[10]*v[2] + m[14]*v[3])/b[3];
  b[3] = 1;
  return b;
}

template <class T, class U>
GLfloat* Cube<T,U>::get_matrix_angles(GLfloat* m)
{
  GLfloat* m2 = (GLfloat*)malloc(16*sizeof(GLfloat));
  memcpy(m2, m, 16*sizeof(GLfloat));
  m2[12]=0;
  m2[13]=0;
  m2[14]=0;

//   for(int i = 0; i < 16; i++)
//     printf("%f ", m2[i]);
//   printf("\n");

  GLfloat* b = (GLfloat*)malloc(4*sizeof(GLfloat));
  GLfloat* x_r = matrix_vector_product(m2, create_vector(1,0,0,1));
  float angle_x = atan2(-x_r[2],x_r[0]);
  float angle_y = atan2(-x_r[1],x_r[2]);
  float angle_z = atan2(x_r[1],x_r[0]);

  b[1] =  angle_x*180.0/3.14159;
  b[2] =  angle_y*180.0/3.14159;
  b[0] =  angle_z*180.0/3.14159;
  b[3] =  0;

  free(m2);

  return b;
}

template <class T, class U>
GLfloat* Cube<T,U>::create_vector(GLfloat x, GLfloat y, GLfloat z, GLfloat w)
{
  GLfloat* b = (GLfloat*)malloc(4*sizeof(GLfloat));
  b[0]=x; b[1] = y; b[2] = z; b[3] = w;
  return b;
}

template <class T, class U>
int Cube<T,U>::find_value_in_ordered_vector(vector< T >& vector_ord, T value)
{
  int begin = 0;
  int end = vector_ord.size() -1;
  int middle = floor((begin + end)/2);

  while( (end-begin)>1 )
    {
      middle = floor((begin + end)/2);
//       printf("%i %i %i %f %f\n", begin, end, middle, vector_ord[middle], value);
      if( vector_ord[middle] < value)
        end = middle;
      if( vector_ord[middle] > value)
        begin = middle;
      if( fabsf(vector_ord[middle] - value) < 1e-8 )
        break;
    }

  if( fabsf(vector_ord[begin] - value) < 1e-8)
    {
//       printf("Error: %f ", vector_ord[begin] - value);
      return begin;
    }
  if( fabsf(vector_ord[end] - value) < 1e-8)
    {
//       printf("Error: %f ", vector_ord[end] - value);
      return end;
    }
//   printf("Error: %f ", vector_ord[middle] - value);
  return middle;
}

template <class T, class U>
vector< T > Cube<T,U>::sort_values()
{
  vector<T> toSort;

  //We do the +- 3 to avoid layers without boosting result
  for(int z = 0 + 3; z < cubeDepth-3; z++)
    for(int y = 0; y < cubeHeight; y++)
      for(int x = 0; x < cubeWidth; x++)
        toSort.push_back(this->at2(x,y,z));

  sort(toSort.begin(), toSort.end());
  reverse(toSort.begin(), toSort.end());

  return toSort;
}

template <class T, class U>
vector< vector< int > > Cube<T,U>::decimate(float threshold, int window_xy, int window_z,  string filename, bool save_boosting_response)
{

  vector< vector < int > > toReturn;
  int cubeCardinality = cubeWidth*cubeHeight*cubeDepth;
  bool* visitedPoints = (bool*)malloc(cubeCardinality*sizeof(bool));
  for(int i = 0; i < cubeCardinality; i++)
    visitedPoints[i] = false;

  //Creates a map from the values of the cube to its coordinates
  multimap< T, int > valueToCoords;
  printf("Cube<T,U>::decimate Creating the map[\n");

  int min_layer = 8;
  int max_layer = 89;
  for(int z = 0; z < min_layer; z++)
    for(int y = 0; y < cubeHeight; y++)
      for(int x = 0; x < cubeWidth; x++)
        visitedPoints[x + y*cubeWidth + z*cubeWidth*cubeHeight] = true;

  for(int z = max_layer; z < cubeDepth; z++)
    for(int y = 0; y < cubeHeight; y++)
      for(int x = 0; x < cubeWidth; x++)
        visitedPoints[x + y*cubeWidth + z*cubeWidth*cubeHeight] = true;

  T min_value = 255;
  T max_value = 0;
  for(int z = min_layer; z < max_layer; z++)
    for(int y = 0; y < cubeHeight; y++)
      for(int x = 0; x < cubeWidth; x++){
        if(at2(x,y,z) > max_value)
          max_value = at2(x,y,z);
        if(at2(x,y,z) < min_value)
          min_value = at2(x,y,z);
      }

  if(sizeof(T) == 1)
    printf("Cube with max_value = %u and min_value = %u\n", max_value, min_value);
  else
    printf("Cube with max_value = %f and min_value = %f\n", max_value, min_value);

  double step_size = (max_value - min_value) / 10;
  double current_threshold = max_value - step_size;

  int position;

  while((current_threshold > min_value) && (current_threshold > threshold - step_size)){
    if( fabs(threshold - current_threshold) < step_size)
      current_threshold = threshold;

    valueToCoords.erase(valueToCoords.begin(), valueToCoords.end() );

    for(int z = min_layer; z < max_layer; z++){
      for(int y = window_xy*2; y < cubeHeight-window_xy*2; y++){
        for(int x = window_xy*2; x < cubeWidth-window_xy*2; x++)
          {
            position = x + y*cubeWidth + z*cubeWidth*cubeHeight;
            if( (this->at2(x,y,z) > current_threshold) &&
                (visitedPoints[position] == false))
              {
                valueToCoords.insert(pair<T, int >(this->at2(x,y,z), position));
              }
          }
      }
      printf("iteration %02i and %07i points\r", z, valueToCoords.size()); fflush(stdout);
    }

    typename multimap< T, int >::iterator iter = valueToCoords.begin();
    T min_value_it = (*iter).first;
    if(sizeof(T)==1)
      printf("\nCube<T,U>:: threshold: %u min_value = %u[", current_threshold, min_value_it);
    else
      printf("\nCube<T,U>:: threshold: %f min_value = %f[", current_threshold, min_value_it);
    fflush(stdout);

    typename multimap< T, int >::reverse_iterator riter = valueToCoords.rbegin();
    int pos;
    int counter = 0;
    int print_step = valueToCoords.size()/50;

    int x_p = 0;
    int y_p = 0;
    int z_p = 0;
    for( riter = valueToCoords.rbegin(); riter != valueToCoords.rend(); riter++)
      {
        counter++;
//         if(counter%print_step == 0)
//           printf("#");fflush(stdout);
        pos = (*riter).second;
        if(visitedPoints[pos] == true)
          continue;
        z_p = pos / (cubeWidth*cubeHeight);
        y_p = (pos - z_p*cubeWidth*cubeHeight)/cubeWidth;
        x_p = pos - z_p*cubeWidth*cubeHeight - y_p*cubeWidth;
        //       printf("%i %i %i %i \n", pos, x_p, y_p, z_p);
        for(int z = max(z_p-window_z,min_layer); z < min(z_p+window_z, (int)max_layer); z++)
          for(int y = max(y_p-window_xy,0); y < min(y_p+window_xy, (int)cubeHeight); y++)
            for(int x = max(x_p-window_xy,0); x < min(x_p+window_xy, (int)cubeWidth); x++)
              visitedPoints[x + y*cubeWidth + z*cubeWidth*cubeHeight] = true;
        vector< int > coords(3);
        coords[0] = x_p;
        coords[1] = y_p;
        coords[2] = z_p;
        toReturn.push_back(coords);
      }
    printf("] %i \n", toReturn.size());
    current_threshold = current_threshold - step_size;
  }

  if(filename!="")
    {
      printf("Cube<T,U>::decimating : saving the points in %s\n", filename.c_str());
      std::ofstream out(filename.c_str());
      for(int i = 0; i < toReturn.size(); i++){
        out << toReturn[i][0] << " " << toReturn[i][1] << " " << toReturn[i][2] ;
        if(save_boosting_response && (sizeof(T)==1) ){
          out << " " << (int)at2(toReturn[i][0], toReturn[i][1], toReturn[i][2]);
        }
        if(save_boosting_response && (sizeof(T)==4) ){
          out << " " << at2(toReturn[i][0], toReturn[i][1], toReturn[i][2]);
        }
        out << std::endl;
      }
      out.close();
    }
  return toReturn;

}

template <class T, class U>
vector< vector< int > > Cube<T,U>::decimate_log(float threshold, int window_xy, int window_z,  string filename, bool save_boosting_response)
{

  vector< vector < int > > toReturn;
  int cubeCardinality = cubeWidth*cubeHeight*cubeDepth;
  bool* visitedPoints = (bool*)malloc(cubeCardinality*sizeof(bool));
  for(int i = 0; i < cubeCardinality; i++)
    visitedPoints[i] = false;

  //Creates a map from the values of the cube to its coordinates
  multimap< T, int > valueToCoords;
  printf("Cube<T,U>::decimate Creating the map[\n");

  int min_layer = 10;
  int max_layer = 86;
  for(int z = 0; z < min_layer; z++)
    for(int y = 0; y < cubeHeight; y++)
      for(int x = 0; x < cubeWidth; x++)
        visitedPoints[x + y*cubeWidth + z*cubeWidth*cubeHeight] = true;

  for(int z = max_layer; z < cubeDepth; z++)
    for(int y = 0; y < cubeHeight; y++)
      for(int x = 0; x < cubeWidth; x++)
        visitedPoints[x + y*cubeWidth + z*cubeWidth*cubeHeight] = true;

  T min_value = 255;
  T max_value = 0;
  for(int z = min_layer; z < max_layer; z++)
    for(int y = 0; y < cubeHeight; y++)
      for(int x = 0; x < cubeWidth; x++){
        if(at2(x,y,z) > max_value)
          max_value = at2(x,y,z);
        if(at2(x,y,z) < min_value)
          min_value = at2(x,y,z);
      }

  if(sizeof(T) == 1)
    printf("Cube with max_value = %u and min_value = %u\n", max_value, min_value);
  else
    printf("Cube with max_value = %f and min_value = %f\n", max_value, min_value);

  T step_size = 0.1;
  T current_threshold = max_value/2;

  int position;

  while( (current_threshold > min_value) && (current_threshold > 1e-8) ){

    valueToCoords.erase(valueToCoords.begin(), valueToCoords.end() );

    for(int z = min_layer; z < max_layer; z++){
      for(int y = 20; y < cubeHeight-20; y++){
        for(int x = 20; x < cubeWidth-20; x++)
          {
            position = x + y*cubeWidth + z*cubeWidth*cubeHeight;
            if( (this->at2(x,y,z) > current_threshold) &&
                (visitedPoints[position] == false))
              {
                valueToCoords.insert(pair<T, int >(this->at2(x,y,z), position));
              }
          }
      }
      printf("iteration %02i and %07i points\r", z, valueToCoords.size()); fflush(stdout);
    }

    typename multimap< T, int >::iterator iter = valueToCoords.begin();
    T min_value_it = (*iter).first;
    if(sizeof(T)==1)
      printf("\nCube<T,U>:: threshold: %u min_value = %u[", current_threshold, min_value_it);
    else
      printf("\nCube<T,U>:: threshold: %f min_value = %f[", current_threshold, min_value_it);
    fflush(stdout);

    typename multimap< T, int >::reverse_iterator riter = valueToCoords.rbegin();
    int pos;
    int counter = 0;
    int print_step = valueToCoords.size()/50;

    int x_p = 0;
    int y_p = 0;
    int z_p = 0;
    for( riter = valueToCoords.rbegin(); riter != valueToCoords.rend(); riter++)
      {
        counter++;
//         if(counter%print_step == 0)
//           printf("#");fflush(stdout);
        pos = (*riter).second;
        if(visitedPoints[pos] == true)
          continue;
        z_p = pos / (cubeWidth*cubeHeight);
        y_p = (pos - z_p*cubeWidth*cubeHeight)/cubeWidth;
        x_p = pos - z_p*cubeWidth*cubeHeight - y_p*cubeWidth;
        //       printf("%i %i %i %i \n", pos, x_p, y_p, z_p);
        for(int z = max(z_p-window_z,min_layer); z < min(z_p+window_z, (int)max_layer); z++)
          for(int y = max(y_p-window_xy,0); y < min(y_p+window_xy, (int)cubeHeight); y++)
            for(int x = max(x_p-window_xy,0); x < min(x_p+window_xy, (int)cubeWidth); x++)
              visitedPoints[x + y*cubeWidth + z*cubeWidth*cubeHeight] = true;
        vector< int > coords(3);
        coords[0] = x_p;
        coords[1] = y_p;
        coords[2] = z_p;
        toReturn.push_back(coords);
      }
    printf("] %i \n", toReturn.size());
    current_threshold = current_threshold*step_size;
  }

  if(filename!="")
    {
      printf("Cube<T,U>::decimating : saving the points in %s\n", filename.c_str());
      std::ofstream out(filename.c_str());
      for(int i = 0; i < toReturn.size(); i++){
        out << toReturn[i][0] << " " << toReturn[i][1] << " " << toReturn[i][2] ;
        if(save_boosting_response && (sizeof(T)==1) ){
          out << " " << (int)at2(toReturn[i][0], toReturn[i][1], toReturn[i][2]);
        }
        if(save_boosting_response && (sizeof(T)==4) ){
          out << " " << at2(toReturn[i][0], toReturn[i][1], toReturn[i][2]);
        }
        out << std::endl;
      }
      out.close();
    }
  return toReturn;

}



// template <class T, class U>
// vector< vector< int > > Cube<T,U>::decimate(float threshold, int window_xy, int window_z,  string filename)
// {

//   vector< vector < int > > toReturn;
//   int cubeCardinality = cubeWidth*cubeHeight*cubeDepth;
//   bool* visitedPoints = (bool*)malloc(cubeCardinality*sizeof(bool));
//   for(int i = 0; i < cubeCardinality; i++)
//     visitedPoints[i] = true;

//   //Creates a map from the values of the cube to its coordinates
//   multimap< float, int > valueToCoords;
//   printf("Cube<T,U>::decimate Creating the map[\n");

//   int min_layer = 8;
//   int max_layer = 90;
//   int position;
//   for(int z = min_layer; z < max_layer; z++){
//     for(int y = window_xy; y < cubeHeight-window_xy; y++){
//       for(int x = window_xy; x < cubeWidth-window_xy; x++)
//         {
//           if(this->at2(x,y,z) > threshold)
//             {
//               position = x + y*cubeWidth + z*cubeWidth*cubeHeight;
//               visitedPoints[position] = false; //Put them as candidates for decimation
//               valueToCoords.insert(pair<float, int >(this->at2(x,y,z), position));
//             }
//         }
//     }
//     printf("iteration %02i and %07i points\r", z, valueToCoords.size()); fflush(stdout);
//   }

//   multimap<float, int >::iterator iter = valueToCoords.begin();
//   T min_value = (*iter).first;
//   int p2 = (*iter).second;
//   int zp = p2/(cubeWidth*cubeHeight);
//   int yp = (p2 - zp*cubeWidth*cubeHeight)/cubeWidth;
//   int xp = p2 - zp*cubeWidth*cubeHeight - yp*cubeWidth;
//   printf("\nCube<T,U>::the minimum value is %f in %i %i %i %i [", min_value, p2,xp,yp,zp);
//   fflush(stdout);

//   multimap<float, int >::reverse_iterator riter = valueToCoords.rbegin();
//   int pos;
//   int counter = 0;
//   int print_step = valueToCoords.size()/50;

//   int x_p = 0;
//   int y_p = 0;
//   int z_p = 0;
//   for( riter = valueToCoords.rbegin(); riter != valueToCoords.rend(); riter++)
//     {
//       counter++;
//       if(counter%print_step == 0)
//         printf("#");fflush(stdout);
//       pos = (*riter).second;
//       if(visitedPoints[pos] == true)
//         continue;
//       z_p = pos / (cubeWidth*cubeHeight);
//       y_p = (pos - z_p*cubeWidth*cubeHeight)/cubeWidth;
//       x_p = pos - z_p*cubeWidth*cubeHeight - y_p*cubeWidth;
// //       printf("%i %i %i %i \n", pos, x_p, y_p, z_p);
//       for(int z = max(z_p-window_z,min_layer); z < min(z_p+window_z, (int)max_layer); z++)
//         for(int y = max(y_p-window_xy,0); y < min(y_p+window_xy, (int)cubeHeight); y++)
//           for(int x = max(x_p-window_xy,0); x < min(x_p+window_xy, (int)cubeWidth); x++)
//             visitedPoints[x + y*cubeWidth + z*cubeWidth*cubeHeight] = true;
//       vector< int > coords(3);
//       coords[0] = x_p;
//       coords[1] = y_p;
//       coords[2] = z_p;
//       toReturn.push_back(coords);
//     }
//   printf("] %i \n", toReturn.size());

//   if(filename!="")
//     {
//       printf("Cube<T,U>::decimating : saving the points in %s\n", filename.c_str());
//       std::ofstream out(filename.c_str());
//       for(int i = 0; i < toReturn.size(); i++)
//         out << toReturn[i][0] << " " << toReturn[i][1] << " " << toReturn[i][2] << std::endl;
//       out.close();
//     }
//   return toReturn;
// }


// template <class T, class U>
// vector< vector< int > > Cube<T,U>::decimate(float threshold, int window_xy, int window_z,  string filename)
// {

//   vector< vector < int > > toReturn;
//   bool*** visitedPoints = (bool***)malloc(cubeDepth*sizeof(bool**));
//   for(int z = 0; z < cubeDepth; z++){
//     visitedPoints[z] = (bool**)malloc(cubeHeight*sizeof(bool*));
//     for(int y = 0; y < cubeHeight; y++)
//       {
//         visitedPoints[z][y] = (bool*)malloc(cubeWidth*sizeof(bool));
//         for(int x = 0; x < cubeWidth; x++)
//           visitedPoints[z][y][x] = true;
//       }
//   }

//   //Creates a map from the values of the cube to its coordinates
//   multimap< float, vector< int > > valueToCoords;
//   printf("Cube<T,U>::decimate Creating the map[\n");

//   //We do the +- window_z to avoid layers without boosting result
//   //for(int z = window_z; z < cubeDepth-window_z; z++){
//   //Lets forget the first 40 layers
// //   for(int z = 40; z < 90; z++){
//   int min_layer = 8;
//   int max_layer = 90;
//   for(int z = min_layer; z < max_layer; z++){
//     for(int y = window_xy; y < cubeHeight-window_xy; y++){
//       for(int x = window_xy; x < cubeWidth-window_xy; x++)
//         {
//           if(this->at2(x,y,z) > threshold)
//             {
//               visitedPoints[z][y][x] = false; //Put them as candidates for decimation
//               vector<int> coords(3);
//               coords[0] = x;
//               coords[1] = y;
//               coords[2] = z;
//               valueToCoords.insert(pair<float, vector<int> >(this->at2(x,y,z), coords));
//             }
//         }
//     }
//     printf("iteration %02i and %07i points\r", z, valueToCoords.size()); fflush(stdout);
//   }

//   multimap<float,vector< int > >::iterator iter = valueToCoords.begin();
//   T min_value = (*iter).first;
//   printf("\nCube<T,U>::the minimum value is %f [", min_value);
//   fflush(stdout);

//   multimap<float,vector< int > >::reverse_iterator riter = valueToCoords.rbegin();
//   vector<int> pos;
//   int counter = 0;
//   int print_step = valueToCoords.size()/50;

//   for( riter = valueToCoords.rbegin(); riter != valueToCoords.rend(); riter++)
//     {
//       counter++;
//       if(counter%print_step == 0)
//         printf("#");fflush(stdout);
//       pos = (*riter).second;
// //       printf("%i %i %i %f\n", pos[0], pos[1], pos[2], (*riter).first);
//       if(visitedPoints[pos[2]][pos[1]][pos[0]] == true)
//         continue;
// //       for(int z = max(pos[2]-window_z,window_z); z < min(pos[2]+window_z, (int)cubeDepth); z++)
//       for(int z = max(pos[2]-window_z,min_layer); z < min(pos[2]+window_z, (int)max_layer); z++)
//         for(int y = max(pos[1]-window_xy,0); y < min(pos[1]+window_xy, (int)cubeHeight); y++)
//           for(int x = max(pos[0]-window_xy,0); x < min(pos[0]+window_xy, (int)cubeWidth); x++)
//             visitedPoints[z][y][x] = true;
//       toReturn.push_back(pos);
//     }
//   printf("] %i \n", toReturn.size());

//   if(filename!="")
//     {
//       printf("Cube<T,U>::decimating : saving the points in %s\n", filename.c_str());
//       std::ofstream out(filename.c_str());
//       for(int i = 0; i < toReturn.size(); i++)
//         out << toReturn[i][0] << " " << toReturn[i][1] << " " << toReturn[i][2] << std::endl;
//       out.close();
//     }
//   return toReturn;
// }

// #if 0
// template <class T, class U>
// vector< vector< int > > Cube<T,U>::decimate(float threshold, int window_xy, int window_z,  string filename)
// {
//   vector< vector < int > > toReturn;

//   //Creates a map from the values of the cube to its coordinates
//   multimap< float, vector< int > > valueToCoords;
//   printf("Cube<T,U>::decimate Creating the map[");

//   //We do the +- window_z to avoid layers without boosting result
// //   for(int z = window_z; z < cubeDepth-window_z; z++){
//   //Lets forget the first 40 layers
//   for(int z = 40; z < cubeDepth-4; z++){
//     for(int y = window_xy; y < cubeHeight-window_xy; y++){
//       for(int x = window_xy; x < cubeWidth-window_xy; x++)
//         {
//           if(this->at2(x,y,z) > threshold)
//             {
//               vector<int> coords(3);
//               coords[0] = x;
//               coords[1] = y;
//               coords[2] = z;
//               valueToCoords.insert(pair<float, vector<int> >(this->at2(x,y,z), coords));
//             }
//         }
//     }
//     printf("iteration %i and %i points\n", z, valueToCoords.size()); fflush(stdout);
//   }

//   multimap<float,vector< int > >::iterator iter = valueToCoords.begin();
//   T min_value = (*iter).first;
//   printf("\nCube<T,U>::the minimum value is %f\n", min_value);

// //   printf("Cube<T,U>::Checking that the keys exist %f\n", min_value);
// //   for(int z = 0 + 3; z < 50; z++)
// //     for(int y = 0; y < 50; y++)
// //       for(int x = 0; x < 50; x++)
// //         {
// //           multimap<float,vector< int > >::iterator iter2 = valueToCoords.find(this->at2(x,y,z));
// //           if(iter2 == valueToCoords.end())
// //             {
// //               printf("Check failed!\npoint: %i %i %i %f\n", x, y, z,
// //                      this->at2(x,y,z));
// //             }
// //         }

//   printf("Cube<T,U>::decimating[");
//   int counter = 0;
//   int totalPoints = valueToCoords.size();
//   int print_step  = totalPoints/10;

//   while(valueToCoords.size()!=0)
//     {
//       int old_size = valueToCoords.size();

//       //Gets the highest valued point
//       iter = valueToCoords.end();
//       iter --;

//       float max_value = (*iter).first;
//       vector<int> pos = (*iter).second;
//       toReturn.push_back(pos);

//       //Cleans in output the surroundings and erase the points of the map
//       for(int z = max(pos[2]-window_z,window_z); z < min(pos[2]+window_z, (int)cubeDepth); z++)
//         for(int y = max(pos[1]-window_xy,0); y < min(pos[1]+window_xy, (int)cubeHeight); y++)
//           for(int x = max(pos[0]-window_xy,0); x < min(pos[0]+window_xy, (int)cubeWidth); x++)
//             {
//               //output->put2(x,y,z,min_value);
//               int nOccurrences = valueToCoords.count(this->at2(x,y,z));
//               if(nOccurrences == 1){
//                 multimap<float,vector< int > >::iterator iter2 = valueToCoords.find(this->at2(x,y,z));
//                 valueToCoords.erase(iter2);
// //                 printf("here\n");
//                 continue;
//               } else
//                 if(nOccurrences == 0){
// //                   printf("Decimating error, point with key %f not found\n", this->at2(x,y,z));
//                   continue;
//                 } else
//                   if(nOccurrences > 1){
//                     pair<multimap< float,vector<int> >::iterator, multimap< float,vector<int> >::iterator> ii;
//                     ii = valueToCoords.equal_range(this->at2(x,y,z));
//                     multimap< float,vector<int> >::iterator i;
//                     for( i = ii.first; i != ii.second; i++){
//                       if( (i->second[0] == x) && (i->second[1] == y) && (i->second[3]==z) )
//                         valueToCoords.erase(i);
//                     }
//                   }
// //               if(iter2 != valueToCoords.end())
// //                 valueToCoords.erase(iter2);
//             }
//       //output->put2(pos[0],pos[1],pos[2], max_value);
//       counter++;
//       if (counter%print_step == 0)
//         {
//           printf("#");
//           fflush(stdout);
//         }
//       if(valueToCoords.size() == old_size)
//         break;
//     }
//   printf("]\n");

//   if(filename!="")
//     {
//       printf("Cube<T,U>::decimating : saving the points in %s\n", filename.c_str());
//       std::ofstream out(filename.c_str());
//       for(int i = 0; i < toReturn.size(); i++)
//         out << toReturn[i][0] << " " << toReturn[i][1] << " " << toReturn[i][2] << std::endl;
//       out.close();
//     }
//   return toReturn;
// }
// #endif

// This is stolen from libvision and modified
#define MIN_RATIO .01
#define MAX_WIDTH  20
#define VERB  0
template< class T, class U >
int Cube<T,U>::gaussian_mask(float sigma, vector< float >& Mask0, vector< float >& Mask1)
{
  float val,sum,Aux0[MAX_WIDTH],Aux1[MAX_WIDTH],coeff=-2.0/(sigma*sigma);
  int i,j,k,n;

  for(i=0;i<MAX_WIDTH;i++) Aux0[i]=0;

  val=sum=Aux0[0]=1.0;
  Aux1[0]=0.0;

  for(n=1;n<MAX_WIDTH;n++){
    val  = n/sigma;
    Aux0[n] = val = exp (- val*val);
    Aux1[n] = coeff * val * n;
    sum += (2.0*val);
    if(val<MIN_RATIO)
      break;
  }

  //Normalization of Mask1 to integrate each side to 1
//   float integral_mask_1 = 0;
//   for(int i = 0; i <= n; i++)
//     integral_mask_1 +=Aux1[i];
//   for(int i = 0; i <= n; i++)
//     Aux1[i] = Aux1[i]/integral_mask_1;


  if(MAX_WIDTH==n){
    fprintf(stderr,"GaussianMask: %f too large, truncating mask.\n",sigma);
    n--;
  }


    Mask0.resize(1+2*n);
    Mask0[n]=1.0/sum;
    Mask1.resize(1+2*n);
    Mask1[n]=0.0;

#if 0
  fprintf(stderr,"--> %f\n",sum);
  for(i=0;i<=n;i++) fprintf(stderr,"%f ",Aux1[i]);
#endif


  for(i=n+1,j=n-1,k=1;k<=n;i++,j--,k++){
      Mask0[i]=Mask0[j]=Aux0[k]/sum;
      Mask1[i]=Aux1[k]/sum;
      Mask1[j]=-(Mask1[i]);
  }
return(n);

}

template <class T, class U>
void Cube<T,U>::convolve_horizontally(vector< float >& mask, Cube< float,double >* output)
{
  int mask_side = mask.size()/2;
  float result = 0;
  int q = 0;
  int x = 0;

  printf("Cube<T,U>::convolve_horizontally [");
  for(int z = 0; z < cubeDepth; z++){
    for(int y = 0; y < cubeHeight; y++)
      {
        //Beginning of the line
        for(x = 0; x <= mask_side; x++)
          {
            result = 0;
            for(q = -x; q <= mask_side;q++){
              result +=this->at2(x+q,y,z)*mask[mask_side+q];
            }
            output->put2(x,y,z,result);
          }
        //Middle of the line
        for(x = mask_side+1; x <= cubeWidth-mask_side-1; x++)
          {
            result = 0;
            for(q = -mask_side; q <=mask_side; q++)
              result += this->at2(x+q,y,z)*mask[mask_side + q];
            output->put2(x,y,z,result);
          }
        //End of the line
        for(x = cubeWidth - mask_side; x < cubeWidth; x++)
          {
            result = 0;
            for(q = -mask_side; q <= cubeWidth -x -1; q++)
              result += this->at2(x+q,y,z)*mask[mask_side + q];
            output->put2(x,y,z,result);
          }
      }
    printf("#");fflush(stdout);
  }
  printf("]\n");

}

template <class T, class U>
void Cube<T,U>::convolve_vertically(vector< float >& mask, Cube<float,double>* output)
{
  int mask_side = mask.size()/2;
  float result = 0;
  int q = 0;
  int y = 0;

  printf("Cube<T,U>::convolve_vertically [");
  for(int z = 0; z < cubeDepth; z++){
    for(int x = 0; x < cubeWidth; x++)
      {
        //Beginning of the line
        for(y = 0; y <= mask_side; y++)
          {
            result = 0;
            for(q = -y; q <= mask_side;q++){
              result +=this->at2(x,y+q,z)*mask[mask_side+q];
            }
            output->put2(x,y,z,result);
          }
        //Middle of the line
        for(y = mask_side+1; y <= cubeHeight-mask_side-1; y++)
          {
            result = 0;
            for(q = -mask_side; q <=mask_side; q++)
              result += this->at2(x,y+q,z)*mask[mask_side + q];
            output->put2(x,y,z,result);
          }
        //End of the line
        for(y = cubeHeight - mask_side; y < cubeHeight; y++)
          {
            result = 0;
            for(q = -mask_side; q <= cubeHeight -y -1; q++)
              result += this->at2(x,y+q,z)*mask[mask_side + q];
            output->put2(x,y,z,result);
          }
      }
    printf("#");fflush(stdout);
  }
  printf("]\n");
}

template <class T, class U>
void Cube<T,U>::convolve_depth(vector< float >& mask, Cube<float,double>* output)
{
  int mask_side = mask.size()/2;
  float result = 0;
  int q = 0;
  int z = 0;

  printf("Cube<T,U>::convolve_depth [");
  for(int y = 0; y < cubeHeight; y++){
    for(int x = 0; x < cubeWidth; x++)
      {
        //Beginning of the line
        for(z = 0; z <= mask_side; z++)
          {
            result = 0;
            for(q = -z; q <= mask_side;q++){
              result +=this->at2(x,y,z+q)*mask[mask_side+q];
            }
            output->put2(x,y,z,result);
          }
        //Middle of the line
        for(z = mask_side+1; z <= cubeDepth-mask_side-1; z++)
          {
            result = 0;
            for(q = -mask_side; q <=mask_side; q++)
              result += this->at2(x,y,z+q)*mask[mask_side + q];
            output->put2(x,y,z,result);
          }
        //End of the line
        for(z = cubeDepth - mask_side; z < cubeDepth; z++)
          {
            result = 0;
            for(q = -mask_side; q <= cubeDepth -z -1; q++)
              result += this->at2(x,y,z+q)*mask[mask_side + q];
            output->put2(x,y,z,result);
          }
      }
    printf("#");fflush(stdout);
  }
  printf("]\n");
}

template <class T, class U>
void Cube<T,U>::gradient_x(float sigma, Cube<float,double>* output, Cube< float,double>* tmp)
{
  vector< float > mask0;
  vector< float > mask1;
  gaussian_mask(sigma, mask0, mask1);

  this->convolve_depth(mask0, output);
  output->convolve_vertically(mask0, tmp);
  tmp->convolve_horizontally(mask1, output);
}

template <class T, class U>
void Cube<T,U>::gradient_y(float sigma, Cube<float,double>* output, Cube<float,double>* tmp)
{
  vector< float > mask0;
  vector< float > mask1;
  gaussian_mask(sigma, mask0, mask1);

  this->convolve_depth(mask0, output);
  output->convolve_horizontally(mask0, tmp);
  tmp->convolve_vertically(mask1, output);
}

template <class T, class U>
void Cube<T,U>::gradient_z(float sigma, Cube<float,double>* output, Cube<float,double>* tmp)
{
  vector< float > mask0;
  vector< float > mask1;
  gaussian_mask(sigma, mask0, mask1);

  this->convolve_vertically(mask0, output);
  output->convolve_horizontally(mask0, tmp);
  tmp->convolve_depth(mask1, output);
}


template <class T, class U>
void Cube<T,U>::calculate_eigen_values(string directory_name)
{
  //Load the input of the hessian
  string volume_nfo = directory_name + "/volume_subsampled.nfo";
  Cube<float,double>* gxx = new Cube<float, double>(volume_nfo, directory_name+"/gxx.vl");
  Cube<float,double>* gxy = new Cube<float,double>(volume_nfo, directory_name+"/gxy.vl");
  Cube<float,double>* gxz = new Cube<float,double>(volume_nfo, directory_name+"/gxz.vl");
  Cube<float,double>* gyy = new Cube<float,double>(volume_nfo, directory_name+"/gyy.vl");
  Cube<float,double>* gyz = new Cube<float,double>(volume_nfo, directory_name+"/gyz.vl");
  Cube<float,double>* gzz = new Cube<float,double>(volume_nfo, directory_name+"/gzz.vl");

  // Places where the eigenvectors will be stored. |eign1| < |eign2| < |eign3|
  Cube<float,double>* eign1 = new Cube<float,double>(volume_nfo, directory_name+"/lambda1.vl");
  Cube<float,double>* eign2 = new Cube<float,double>(volume_nfo, directory_name+"/lambda2.vl");
  Cube<float,double>* eign3 = new Cube<float,double>(volume_nfo, directory_name+"/lambda3.vl");

  gsl_vector *eign = gsl_vector_alloc (3);

  gsl_eigen_symm_workspace* w =  gsl_eigen_symm_alloc (3);
  double data[9];

  float l1 = 0;
  float l2 = 0;
  float l3 = 0;

  float l1_tmp = 0;
  float l2_tmp = 0;
  float l3_tmp = 0;

  printf("Cube<T,U>::calculate_eigen_values[ ");
  for(int z = 0; z < cubeDepth; z++){
    for(int y = 0; y < cubeHeight; y++){
      for(int x = 0; x < cubeWidth; x++){

        data[0] = gxx->at2(x,y,z);
        data[1] = gxy->at2(x,y,z);
        data[2] = gxz->at2(x,y,z);
        data[3] = data[1];
        data[4] = gyy->at2(x,y,z);
        data[5] = gyz->at2(x,y,z);
        data[6] = data[2];
        data[7] = data[5];
        data[8] = gzz->at2(x,y,z);

        gsl_matrix_view M
          = gsl_matrix_view_array (data, 3, 3);

        gsl_eigen_symm (&M.matrix, eign, w);

        l1 = gsl_vector_get (eign, 0);
        l2 = gsl_vector_get (eign, 1);
        l3 = gsl_vector_get (eign, 2);
        l1_tmp = fabs(l1);
        l2_tmp = fabs(l2);
        l3_tmp = fabs(l3);

        //Orders and stores the eigenvalues
        if( (l1_tmp <= l2_tmp) && (l2_tmp <= l3_tmp)){
          eign1->put2(x,y,z,l1);
          eign2->put2(x,y,z,l2);
          eign3->put2(x,y,z,l3);
        }
        else
        if( (l1_tmp <= l3_tmp) && (l3_tmp <= l2_tmp) )
        {
          eign1->put2(x,y,z,l1);
          eign2->put2(x,y,z,l3);
          eign3->put2(x,y,z,l2);
        }
        else
        if( (l2_tmp <= l1_tmp) && (l1_tmp <= l3_tmp) )
        {
          eign1->put2(x,y,z,l2);
          eign2->put2(x,y,z,l1);
          eign3->put2(x,y,z,l3);
        }
        else
        if( (l2_tmp <= l3_tmp) && (l3_tmp <= l1_tmp) )
        {
          eign1->put2(x,y,z,l2);
          eign2->put2(x,y,z,l3);
          eign3->put2(x,y,z,l1);
        }
        else
        if( (l3_tmp <= l1_tmp) && (l1_tmp <= l2_tmp) )
        {
          eign1->put2(x,y,z,l3);
          eign2->put2(x,y,z,l1);
          eign3->put2(x,y,z,l2);
        }
        else
        if( (l3_tmp <= l2_tmp) && (l2_tmp <= l1_tmp) )
        {
          eign1->put2(x,y,z,l3);
          eign2->put2(x,y,z,l2);
          eign3->put2(x,y,z,l1);
        }
      }
    }
    printf("#"); fflush(stdout);
  }
  printf("]\n");
  gsl_vector_free (eign);

  delete gxx;
  delete gxy;
  delete gxz;
  delete gyy;
  delete gyz;
  delete gzz;
  delete eign1;
  delete eign2;
  delete eign3;

//   gsl_eigen_symmv_free(w);

//   //Creates the output
//   Cube<float>* l1 = new Cube<float>(volume_nfo, directory_name+"/l1.vl");
}

template <class T, class U>
void Cube<T,U>::calculate_eigen_values(float sigma, string directory_name)
{

  char gxx_b[512];
  char gxy_b[512];
  char gxz_b[512];
  char gyy_b[512];
  char gyz_b[512];
  char gzz_b[512];
  char lbd_1[512];
  char lbd_2[512];
  char lbd_3[512];


  sprintf(gxx_b, "/g_xx_%02.02f.vl", sigma);
  sprintf(gxy_b, "/g_xy_%02.02f.vl", sigma);
  sprintf(gxz_b, "/g_xz_%02.02f.vl", sigma);
  sprintf(gyz_b, "/g_yy_%02.02f.vl", sigma);
  sprintf(gyz_b, "/g_yz_%02.02f.vl", sigma);
  sprintf(gzz_b, "/g_zz_%02.02f.vl", sigma);
  sprintf(lbd_1, "/lambda1_%02.02f.vl", sigma);
  sprintf(lbd_2, "/lambda2_%02.02f.vl", sigma);
  sprintf(lbd_3, "/lambda3_%02.02f.vl", sigma);


  //Load the input of the hessian
  string volume_nfo = directory_name + "/volume_subsampled.nfo";
  Cube<float,double>* gxx = new Cube<float, double>(volume_nfo, directory_name+ gxx_b);
  Cube<float,double>* gxy = new Cube<float,double>(volume_nfo, directory_name+  gxy_b);
  Cube<float,double>* gxz = new Cube<float,double>(volume_nfo, directory_name+  gxz_b);
  Cube<float,double>* gyy = new Cube<float,double>(volume_nfo, directory_name+  gyy_b);
  Cube<float,double>* gyz = new Cube<float,double>(volume_nfo, directory_name+  gyz_b);
  Cube<float,double>* gzz = new Cube<float,double>(volume_nfo, directory_name+  gzz_b);

  // Places where the eigenvalues will be stored. |eign1| < |eign2| < |eign3|
  Cube<float,double>* eign1 = new Cube<float,double>(volume_nfo, directory_name+ lbd_1);
  Cube<float,double>* eign2 = new Cube<float,double>(volume_nfo, directory_name+ lbd_2);
  Cube<float,double>* eign3 = new Cube<float,double>(volume_nfo, directory_name+ lbd_3);

  gsl_vector *eign = gsl_vector_alloc (3);

  gsl_eigen_symm_workspace* w =  gsl_eigen_symm_alloc (3);
  double data[9];

  float l1 = 0;
  float l2 = 0;
  float l3 = 0;

  float l1_tmp = 0;
  float l2_tmp = 0;
  float l3_tmp = 0;

  printf("Cube<T,U>::calculate_eigen_values[ ");
  for(int z = 0; z < cubeDepth; z++){
    for(int y = 0; y < cubeHeight; y++){
      for(int x = 0; x < cubeWidth; x++){

        data[0] = gxx->at2(x,y,z);
        data[1] = gxy->at2(x,y,z);
        data[2] = gxz->at2(x,y,z);
        data[3] = data[1];
        data[4] = gyy->at2(x,y,z);
        data[5] = gyz->at2(x,y,z);
        data[6] = data[2];
        data[7] = data[5];
        data[8] = gzz->at2(x,y,z);

        gsl_matrix_view M
          = gsl_matrix_view_array (data, 3, 3);

        gsl_eigen_symm (&M.matrix, eign, w);

        l1 = gsl_vector_get (eign, 0);
        l2 = gsl_vector_get (eign, 1);
        l3 = gsl_vector_get (eign, 2);

        eign1->put2(x,y,z,l1);
        eign2->put2(x,y,z,l2);
        eign3->put2(x,y,z,l3);
      }
    }
    printf("#"); fflush(stdout);
  }
  printf("]\n");
  gsl_vector_free (eign);

  delete gxx;
  delete gxy;
  delete gxz;
  delete gyy;
  delete gyz;
  delete gzz;
  delete eign1;
  delete eign2;
  delete eign3;

//   gsl_eigen_symmv_free(w);

//   //Creates the output
//   Cube<float>* l1 = new Cube<float>(volume_nfo, directory_name+"/l1.vl");
}


template <class T, class U>
void Cube<T,U>::calculate_eigen_vector_lower_eigenvalue(string directory_name)
{
  //Load the input of the hessian
  string volume_nfo = directory_name + "/volume.nfo";
  Cube<float,double>* gxx = new Cube<float, double>(volume_nfo, directory_name+"/gxx.vl");
  Cube<float,double>* gxy = new Cube<float,double>(volume_nfo, directory_name+"/gxy.vl");
  Cube<float,double>* gxz = new Cube<float,double>(volume_nfo, directory_name+"/gxz.vl");
  Cube<float,double>* gyy = new Cube<float,double>(volume_nfo, directory_name+"/gyy.vl");
  Cube<float,double>* gyz = new Cube<float,double>(volume_nfo, directory_name+"/gyz.vl");
  Cube<float,double>* gzz = new Cube<float,double>(volume_nfo, directory_name+"/gzz.vl");

  // Places where the eigenvectors will be stored. |eign1| < |eign2| < |eign3|
  Cube<float,double>* eigv_x = new Cube<float,double>(volume_nfo, directory_name+"/eigv_x.vl");
  Cube<float,double>* eigv_y = new Cube<float,double>(volume_nfo, directory_name+"/eigv_y.vl");
  Cube<float,double>* eigv_z = new Cube<float,double>(volume_nfo, directory_name+"/eigv_z.vl");

  gsl_vector *eign = gsl_vector_alloc (3);
  gsl_matrix *evec = gsl_matrix_alloc (3, 3);


  gsl_eigen_symmv_workspace* w =  gsl_eigen_symmv_alloc (3);
  double data[9];

  float l1 = 0;
  float l2 = 0;
  float l3 = 0;

  float l1_tmp = 0;
  float l2_tmp = 0;
  float l3_tmp = 0;

  printf("Cube<T,U>::calculate_eigen_vector_lower_eigenvalue[ ");
  fflush(stdout);
  for(int z = 0; z < cubeDepth; z++){
    for(int y = 0; y < cubeHeight; y++){
      for(int x = 0; x < cubeWidth; x++){

        data[0] = gxx->at2(x,y,z);
        data[1] = gxy->at2(x,y,z);
        data[2] = gxz->at2(x,y,z);
        data[3] = data[1];
        data[4] = gyy->at2(x,y,z);
        data[5] = gyz->at2(x,y,z);
        data[6] = data[2];
        data[7] = data[5];
        data[8] = gzz->at2(x,y,z);

        gsl_matrix_view M
          = gsl_matrix_view_array (data, 3, 3);

        gsl_eigen_symmv (&M.matrix, eign, evec, w);

        l1 = gsl_vector_get (eign, 0);
        l2 = gsl_vector_get (eign, 1);
        l3 = gsl_vector_get (eign, 2);
        l1_tmp = fabs(l1);
        l2_tmp = fabs(l2);
        l3_tmp = fabs(l3);

        //Orders and stores the eigenvalues
        if( (l1_tmp <= l2_tmp) && (l2_tmp <= l3_tmp) ||
            (l1_tmp <= l3_tmp) && (l3_tmp <= l2_tmp) ){
          eigv_x->put2(x,y,z,gsl_matrix_get(&M.matrix, 0,0));
          eigv_y->put2(x,y,z,gsl_matrix_get(&M.matrix, 1,0));
          eigv_z->put2(x,y,z,gsl_matrix_get(&M.matrix, 2,0));
        }
        else
        if( (l2_tmp <= l1_tmp) && (l1_tmp <= l3_tmp) ||
            (l2_tmp <= l3_tmp) && (l3_tmp <= l1_tmp) )
        {
          eigv_x->put2(x,y,z,gsl_matrix_get(&M.matrix, 0,1));
          eigv_y->put2(x,y,z,gsl_matrix_get(&M.matrix, 1,1));
          eigv_z->put2(x,y,z,gsl_matrix_get(&M.matrix, 2,1));
        }
        else
        if( (l3_tmp <= l1_tmp) && (l1_tmp <= l2_tmp) ||
            (l3_tmp <= l2_tmp) && (l2_tmp <= l1_tmp) )
        {
          eigv_x->put2(x,y,z,gsl_matrix_get(&M.matrix, 0,2));
          eigv_y->put2(x,y,z,gsl_matrix_get(&M.matrix, 1,2));
          eigv_z->put2(x,y,z,gsl_matrix_get(&M.matrix, 2,2));
        }
      }
    }
    printf("#"); fflush(stdout);
  }
  printf("]\n");
  gsl_vector_free (eign);

  delete gxx;
  delete gxy;
  delete gxz;
  delete gyy;
  delete gyz;
  delete gzz;
  delete eigv_x;
  delete eigv_y;
  delete eigv_z;
}


//Implements the f-measure as described in "multiscale vessel enhancement filtering", by Alejandro F. Frangi
//The third eigenvalue will be scaled by 4 to account for the ellipsis of the filament, it is not a tube in the 
// subsampled images, but an ellipsoid
template <class T, class U>
void Cube<T,U>::calculate_f_measure(string directory_name)
{
  string volume_nfo = directory_name + "/volume_subsampled.nfo";
  Cube<float,double>* eign1 = new Cube<float,double>(volume_nfo, directory_name+"/lambda1.vl");
  Cube<float,double>* eign2 = new Cube<float,double>(volume_nfo, directory_name+"/lambda2.vl");
  Cube<float,double>* eign3 = new Cube<float,double>(volume_nfo, directory_name+"/lambda3.vl");

  Cube<float,double>* f_measure = new Cube<float,double>(volume_nfo, directory_name + "/f_measure.vl");

  printf("Cube<T,U>::calculate_F_measure getting the max of the norm of the hessians [");

  float max_s = 0;
  float s;

  for(int z = 0; z < cubeDepth; z++){
    for(int y = 0; y < cubeHeight; y++){
      for(int x = 0; x < cubeWidth; x++){
        s = eign1->at2(x,y,z)*eign1->at2(x,y,z) +
            eign2->at2(x,y,z)*eign2->at2(x,y,z) +
            eign3->at2(x,y,z)*eign3->at2(x,y,z);
        f_measure->put2(x,y,z,s);
        if (s > max_s) max_s = s;
//         printf("%f %f %f -> %f\n", eign1->at2(x,y,z), eign2->at2(x,y,z), eign3->at2(x,y,z),s);
      }
    }
    printf("#");fflush(stdout);
  }
  printf("]\n   - the max is %f\n", max_s);

  float l1 = 0;
  float l2 = 0;
  float l3 = 0;
  printf("Cube<T,U>::calculate_F_measure: [");
  for(int z = 0; z < cubeDepth; z++){
    for(int y = 0; y < cubeHeight; y++){
      for(int x = 0; x < cubeWidth; x++){
        l1 = eign1->at(x,y,z);
        l2 = eign2->at(x,y,z);
        l3 = eign3->at(x,y,z);

//         if( (l2 < 0) || (l3 < 0)){
//             f_measure->put2(x,y,z,0);
//             continue;
//         }
        s = ( 1 - exp( -l2*l2/(l3*l3*0.5) ) )*
            exp(-l1*l1/(fabs(l2*l3)*0.5) )*
            (1 - exp(-2*f_measure->at(x,y,z)/(max_s)));
        f_measure->put2(x,y,z,s);
      }
    }
    printf("#");fflush(stdout);
  }
  printf("]\n");
}

template <class T, class U>
void Cube<T,U>::norm_cube(Cube<float,double>* c1, Cube<float,double>* c2, Cube<float,double>* output)
{
  printf("Cube<T,U>::norm_cube[");
  for(int z = 0; z < output->cubeDepth; z++){
    for(int y = 0; y < output->cubeHeight; y++){
      for(int x = 0; x < output->cubeWidth; x++){
        output->put2(x,y,z,
                    sqrt(c1->at(x,y,z)*c1->at(x,y,z) + c2->at(x,y,z)*c2->at(x,y,z)));
      }
    }
    printf("#");fflush(stdout);
  }
  printf("]\n");
}

template <class T, class U>
void Cube<T,U>::norm_cube(string volume_nfo, string volume_1, string volume_2, string volume_3, string volume_output)
{
  Cube<float,double>* output = new Cube<float,double>(volume_nfo, volume_output);
  Cube<float,double>* v1 = new Cube<float,double>(volume_nfo, volume_1);
  Cube<float,double>* v2 = new Cube<float,double>(volume_nfo, volume_2);
  Cube<float,double>* v3 = new Cube<float,double>(volume_nfo, volume_3);

  printf("Cube<T,U>::norm_cube[");
  for(int z = 0; z < output->cubeDepth; z++){
    for(int y = 0; y < output->cubeHeight; y++){
      for(int x = 0; x < output->cubeWidth; x++){
        output->put2(x,y,z,
                    sqrt(v1->at2(x,y,z)*v1->at2(x,y,z) +
                         v2->at2(x,y,z)*v2->at2(x,y,z) +
                         v3->at2(x,y,z)*v3->at2(x,y,z) )
                     );
      }
    }
    printf("#");fflush(stdout);
  }
  printf("]\n");
  delete v1;
  delete v2;
  delete v3;
  delete output;
}

template <class T, class U>
void Cube<T,U>::get_ROC_curve
(string volume_nfo,
 string volume_positive,
 string volume_negative,
 string output_file,
 int nPoints
)
{
//   Cube<float,double>* positive = new Cube<float,double>(volume_nfo, volume_positive);
//   Cube<float,double>* negative = new Cube<float,double>(volume_nfo, volume_negative);

  Cube<uchar,ulong>* positive = new Cube<uchar,ulong>(volume_nfo, volume_positive);
  Cube<uchar,ulong>* negative = new Cube<uchar,ulong>(volume_nfo, volume_negative);


  float min = 1e6;
  float max = -1e6;
  int nPositivePoints = 0;
  int nNegativePoints = 0;

  printf("Cube<T,U>::getROC_curve: "); fflush(stdout);
  for(int z = 0; z < positive->cubeDepth; z++){
    for(int y = 0; y < positive->cubeHeight; y++){
      for(int x = 0; x < positive->cubeWidth; x++)
        {
          if(positive->at2(x,y,z)!=0){
            if(positive->at2(x,y,z)<min)
              min = positive->at2(x,y,z);
            if(positive->at2(x,y,z)>max)
              max = positive->at2(x,y,z);
            nPositivePoints++;
          }
          if(negative->at2(x,y,z)!=0){
            if(negative->at2(x,y,z)<min)
              min = negative->at2(x,y,z);
            if(negative->at2(x,y,z)>max)
              max = negative->at2(x,y,z);
            nNegativePoints++;
          }
        }
    }
  }
  printf("max:%f min:%f nPos:%i nNeg:%i [", max, min, nPositivePoints, nNegativePoints);
  fflush(stdout);
  std::ofstream out(output_file.c_str());

  float tp = 0;
  float fp = 0;

  for(float threshold = min; threshold < max; threshold += (max-min)/nPoints)
    {
      tp = 0;
      fp = 0;
      for(int z = 0; z < positive->cubeDepth; z++){
        for(int y = 0; y < positive->cubeHeight; y++){
          for(int x = 0; x < positive->cubeWidth; x++){
            if( (positive->at2(x,y,z) != 0) &&
                (positive->at2(x,y,z) > threshold) )
              tp++;
            if( (negative->at2(x,y,z) != 0) &&
                (negative->at2(x,y,z) > threshold) )
              fp++;
          }
        }
      }
      out << fp/nNegativePoints << " " << tp/nPositivePoints << std::endl;
      printf("#"); fflush(stdout);
    }
  printf("]\n");

  out.close();
  delete positive;
  delete negative;
}


template <class T, class U>
void Cube<T,U>::draw_whole(float rotx, float roty, float nPlanes, int min_max)
{

//   GLboolean resident[1];
//   GLboolean pepe = glAreTexturesResident(1, &wholeTextureDepth, resident);
//   if(resident[0] == GL_TRUE)
//     printf("Texture resident\n");
//   else
//     printf("Texture NOT resident\n");

  GLint max_texture_size = 0;
  glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_texture_size);

  draw_orientation_grid(false);

  //In object coordinates, we will draw the cube
  GLfloat widthStep = float(parentCubeWidth)*voxelWidth/2;
  GLfloat heightStep = float(parentCubeHeight)*voxelHeight/2;
  GLfloat depthStep = float(parentCubeDepth)*voxelDepth/2;

  int nColTotal = nColToDraw + colOffset;
  int nRowTotal = nRowToDraw + rowOffset;

  int end_x = min((nColTotal+1)*max_texture_size, (int)parentCubeWidth);
  int end_y = min((nRowTotal+1)*max_texture_size, (int)parentCubeHeight);

  GLfloat pModelViewMatrix[16];
  glGetFloatv(GL_MODELVIEW_MATRIX, pModelViewMatrix);

  GLfloat** cubePoints = (GLfloat**)malloc(8*sizeof(GLfloat*));;

  cubePoints[0] = create_vector(-widthStep,
                                heightStep, -depthStep, 1.0f);

  cubePoints[1] = create_vector(-widthStep,
                                heightStep,  depthStep, 1.0f);

  cubePoints[2] = create_vector(widthStep,
                                heightStep,  depthStep, 1.0f);

  cubePoints[3] = create_vector(widthStep,
                                heightStep, -depthStep, 1.0f);

  cubePoints[4] = create_vector(-widthStep,
                                -heightStep,
                                -depthStep, 1.0f);

  cubePoints[5] = create_vector(-widthStep,
                                -heightStep,  depthStep, 1.0f);

  cubePoints[6] = create_vector(widthStep,
                                -heightStep,  depthStep, 1.0f);

  cubePoints[7] = create_vector(widthStep,
                                -heightStep, -depthStep, 1.0f);

  // We will get the coordinates of the vertex of the cube in the modelview coordinates
  glLoadIdentity();
  GLfloat* cubePoints_c[8];
  glColor3f(0,0,0);
  for(int i=0; i < 8; i++)
    cubePoints_c[i] = matrix_vector_product(pModelViewMatrix, cubePoints[i]);

  //Draws the points numbers and the coordinates of the textures
  if(0){
    for(int i=0; i < 8; i++)
      {
        glColor3f(0.0,1.0,0.0);
        glPushMatrix();
        glTranslatef(cubePoints_c[i][0], cubePoints_c[i][1], cubePoints_c[i][2]);
        render_string("%i",i);
        glPopMatrix();
      }
    glPushMatrix();
    glTranslatef(cubePoints_c[0][0], cubePoints_c[0][1], cubePoints_c[0][2]);
    glRotatef(rotx, 1.0,0,0);
    glRotatef(roty, 0,1.0,0);
    //Draw The Z axis
    glColor3f(0.0, 0.0, 1.0);
    glutSolidCone(1.0, 10, 20, 20);
    glPushMatrix();
    glTranslatef(0,0,10);
    render_string("T");
    glPopMatrix();
    //Draw the x axis
    glColor3f(1.0, 0.0, 0.0);
    glRotatef(90, 0.0, 1.0, 0.0);
    glutSolidCone(1.0, 10, 20, 20);
    glPushMatrix();
    glTranslatef(0,0,10);
    render_string("R");
    glPopMatrix();
    //Draw the y axis
    glColor3f(0.0, 1.0, 0.0);
    glRotatef(90, 1.0, 0.0, 0.0);
    glutSolidCone(1.0, 10, 20, 20);
    glPushMatrix();
    glTranslatef(0,0,10);
    render_string("S");
    glPopMatrix();
    glPopMatrix();
    glColor3f(1.0, 1.0, 1.0);
  }

  //Find the closest and furthest vertex of the square
  float closest_distance = 1e9;
  float furthest_distance= 0;
  int closest_point_idx = 0;
  int furthest_point_idx = 0;
  for(int i = 0; i < 8; i++)
    {
      float dist = cubePoints_c[i][0]*cubePoints_c[i][0] + cubePoints_c[i][1]*cubePoints_c[i][1] + cubePoints_c[i][2]*cubePoints_c[i][2];
      if(dist < closest_distance)
        {
          closest_distance = dist;
          closest_point_idx = i;
        }
      if(dist > furthest_distance)
        {
          furthest_distance = dist;
          furthest_point_idx = i;
        }
    }

  //Draws a sphere in the furthest and closest point of the cube
  if(0){
    glPushMatrix();
    glTranslatef(cubePoints_c[closest_point_idx][0], cubePoints_c[closest_point_idx][1], cubePoints_c[closest_point_idx][2]);
    glColor3f(0.0,1.0,0.0);
    glutWireSphere(5,10,10);
    glPopMatrix();

    glPushMatrix();
    glTranslatef(cubePoints_c[furthest_point_idx][0], cubePoints_c[furthest_point_idx][1], cubePoints_c[furthest_point_idx][2]);
    glColor3f(0.0,0.0,1.0);
    glutWireSphere(5,10,10);
    glPopMatrix();
  }

//   printf("%f\n", cubePoints_c[furthest_point_idx][2] - cubePoints_c[closest_point_idx][2]);
  //Draws the cube
  for(float depth = 0/nPlanes; depth <= 1.0; depth+=1.0/nPlanes)
    {
      float z_plane = (cubePoints_c[furthest_point_idx][2]*(1-depth) + depth*cubePoints_c[closest_point_idx][2]);
      //Find the lines that intersect with the plane. For that we will define the lines and find the intersection of the line with the point
      GLfloat lambda_lines[12];
      if( ((cubePoints_c[1][2] > z_plane) && (cubePoints_c[0][2] > z_plane)) ||
          ((cubePoints_c[1][2] < z_plane) && (cubePoints_c[0][2] < z_plane)) )
            lambda_lines[0] = -1;
            else
            lambda_lines[0 ] = (z_plane - cubePoints_c[1][2]) / (cubePoints_c[0][2] - cubePoints_c[1][2]); //0-1

      if( ((cubePoints_c[3][2] > z_plane) && (cubePoints_c[0][2] > z_plane)) ||
          ((cubePoints_c[3][2] < z_plane) && (cubePoints_c[0][2] < z_plane)) )
            lambda_lines[1] = -1;
            else
            lambda_lines[1 ] = (z_plane - cubePoints_c[3][2]) / (cubePoints_c[0][2] - cubePoints_c[3][2]); //0-3

      if( ((cubePoints_c[4][2] > z_plane) && (cubePoints_c[0][2] > z_plane)) ||
          ((cubePoints_c[4][2] < z_plane) && (cubePoints_c[0][2] < z_plane)) )
            lambda_lines[2] = -1;
            else
            lambda_lines[2 ] = (z_plane - cubePoints_c[4][2]) / (cubePoints_c[0][2] - cubePoints_c[4][2]); //0-4

      if( ((cubePoints_c[7][2] > z_plane) && (cubePoints_c[4][2] > z_plane)) ||
          ((cubePoints_c[7][2] < z_plane) && (cubePoints_c[4][2] < z_plane)))
            lambda_lines[3] = -1;
            else
            lambda_lines[3 ] = (z_plane - cubePoints_c[7][2]) / (cubePoints_c[4][2] - cubePoints_c[7][2]); //4-7

      if( ((cubePoints_c[5][2] > z_plane) && (cubePoints_c[4][2] > z_plane)) ||
          ((cubePoints_c[5][2] < z_plane) && (cubePoints_c[4][2] < z_plane)))
            lambda_lines[4] = -1;
            else
            lambda_lines[4 ] = (z_plane - cubePoints_c[5][2]) / (cubePoints_c[4][2] - cubePoints_c[5][2]); //4-5

      if( ((cubePoints_c[1][2] > z_plane) && (cubePoints_c[2][2] > z_plane)) ||
          ((cubePoints_c[1][2] < z_plane) && (cubePoints_c[2][2] < z_plane)))
            lambda_lines[5] = -1;
            else
            lambda_lines[5 ] = (z_plane - cubePoints_c[2][2]) / (cubePoints_c[1][2] - cubePoints_c[2][2]); //1-2

      if( ((cubePoints_c[1][2] > z_plane) && (cubePoints_c[5][2] > z_plane)) ||
          ((cubePoints_c[1][2] < z_plane) && (cubePoints_c[5][2] < z_plane)))
            lambda_lines[6] = -1;
            else
            lambda_lines[6 ] = (z_plane - cubePoints_c[5][2]) / (cubePoints_c[1][2] - cubePoints_c[5][2]); //1-5

      if( ((cubePoints_c[6][2] > z_plane) && (cubePoints_c[5][2] > z_plane)) ||
          ((cubePoints_c[6][2] < z_plane) && (cubePoints_c[5][2] < z_plane)))
            lambda_lines[7] = -1;
            else
            lambda_lines[7 ] = (z_plane - cubePoints_c[6][2]) / (cubePoints_c[5][2] - cubePoints_c[6][2]); //5-6

      if( ((cubePoints_c[2][2] > z_plane) && (cubePoints_c[3][2] > z_plane)) ||
          ((cubePoints_c[2][2] < z_plane) && (cubePoints_c[3][2] < z_plane)))
            lambda_lines[8] = -1;
            else
            lambda_lines[8 ] = (z_plane - cubePoints_c[2][2]) / (cubePoints_c[3][2] - cubePoints_c[2][2]); //3-2

      if( ((cubePoints_c[7][2] > z_plane) && (cubePoints_c[3][2] > z_plane)) ||
          ((cubePoints_c[7][2] < z_plane) && (cubePoints_c[3][2] < z_plane)))
            lambda_lines[9] = -1;
            else
            lambda_lines[9 ] = (z_plane - cubePoints_c[7][2]) / (cubePoints_c[3][2] - cubePoints_c[7][2]); //3-7

      if( ((cubePoints_c[6][2] > z_plane) && (cubePoints_c[7][2] > z_plane)) ||
          ((cubePoints_c[6][2] < z_plane) && (cubePoints_c[7][2] < z_plane)))
            lambda_lines[10] = -1;
            else
            lambda_lines[10] = (z_plane - cubePoints_c[6][2]) / (cubePoints_c[7][2] - cubePoints_c[6][2]); //7-6

      if( ((cubePoints_c[2][2] > z_plane) && (cubePoints_c[6][2] > z_plane)) ||
          ((cubePoints_c[2][2] < z_plane) && (cubePoints_c[6][2] < z_plane)))
            lambda_lines[11] = -1;
            else
            lambda_lines[11] = (z_plane - cubePoints_c[2][2]) / (cubePoints_c[6][2] - cubePoints_c[2][2]); //6-2

      // We will store the point and texture coordinates of the points that we will draw afterwards
      //There is at maximum five intersections -> therefore we will define an array of five points
      GLfloat intersectionPoints[5][6];
      int intersectionPointsIdx = 0;
      for(int i = 0; i < 12; i++)
        {
          if( (lambda_lines[i] > 0) && (lambda_lines[i] < 1))
            {
              float x_point = 0;
              float y_point = 0;
              float z_point = 0;
              float r_point = 0;
              float s_point = 0;
              float t_point = 0;
              switch(i)
                {
                case 0: //0-1
                  x_point = cubePoints_c[0][0]*lambda_lines[i] + cubePoints_c[1][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[0][1]*lambda_lines[i] + cubePoints_c[1][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[0][2]*lambda_lines[i] + cubePoints_c[1][2]*(1-lambda_lines[i]);
                  r_point = 0;
                  s_point = 0;
                  t_point = (1-lambda_lines[i]);
                  break;
                case 1: //0-3
                  x_point = cubePoints_c[0][0]*lambda_lines[i] + cubePoints_c[3][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[0][1]*lambda_lines[i] + cubePoints_c[3][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[0][2]*lambda_lines[i] + cubePoints_c[3][2]*(1-lambda_lines[i]);
                  r_point = 1-lambda_lines[i];
                  s_point = 0;
                  t_point = 0;
                  break;
                case 2: //0-4
                  x_point = cubePoints_c[0][0]*lambda_lines[i] + cubePoints_c[4][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[0][1]*lambda_lines[i] + cubePoints_c[4][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[0][2]*lambda_lines[i] + cubePoints_c[4][2]*(1-lambda_lines[i]);
                  r_point = 0;
                  s_point = 1-lambda_lines[i];
                  t_point = 0;
                  break;
                case 3: //4-7
                  x_point = cubePoints_c[4][0]*lambda_lines[i] + cubePoints_c[7][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[4][1]*lambda_lines[i] + cubePoints_c[7][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[4][2]*lambda_lines[i] + cubePoints_c[7][2]*(1-lambda_lines[i]);
                  r_point = 1-lambda_lines[i];
                  s_point = 1;
                  t_point = 0;
                  break;
                case 4: //4-5
                  x_point = cubePoints_c[4][0]*lambda_lines[i] + cubePoints_c[5][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[4][1]*lambda_lines[i] + cubePoints_c[5][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[4][2]*lambda_lines[i] + cubePoints_c[5][2]*(1-lambda_lines[i]);
                  r_point = 0;
                  s_point = 1;
                  t_point = 1-lambda_lines[i];
                  break;
                case 5: //1-2
                  x_point = cubePoints_c[1][0]*lambda_lines[i] + cubePoints_c[2][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[1][1]*lambda_lines[i] + cubePoints_c[2][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[1][2]*lambda_lines[i] + cubePoints_c[2][2]*(1-lambda_lines[i]);
                  r_point = 1-lambda_lines[i];
                  s_point = 0;
                  t_point = 1;
                  break;
                case 6: //1-5
                  x_point = cubePoints_c[1][0]*lambda_lines[i] + cubePoints_c[5][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[1][1]*lambda_lines[i] + cubePoints_c[5][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[1][2]*lambda_lines[i] + cubePoints_c[5][2]*(1-lambda_lines[i]);
                  r_point = 0;
                  s_point = 1-lambda_lines[i];
                  t_point = 1;
                  break;
                case 7: //5-6
                  x_point = cubePoints_c[5][0]*lambda_lines[i] + cubePoints_c[6][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[5][1]*lambda_lines[i] + cubePoints_c[6][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[5][2]*lambda_lines[i] + cubePoints_c[6][2]*(1-lambda_lines[i]);
                  r_point = 1-lambda_lines[i];
                  s_point = 1;
                  t_point = 1;
                  break;
                case 8: //3-2
                  x_point = cubePoints_c[3][0]*lambda_lines[i] + cubePoints_c[2][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[3][1]*lambda_lines[i] + cubePoints_c[2][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[3][2]*lambda_lines[i] + cubePoints_c[2][2]*(1-lambda_lines[i]);
                  r_point = 1;
                  s_point = 0;
                  t_point = 1-lambda_lines[i];
                  break;
                case 9: //3-7
                  x_point = cubePoints_c[3][0]*lambda_lines[i] + cubePoints_c[7][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[3][1]*lambda_lines[i] + cubePoints_c[7][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[3][2]*lambda_lines[i] + cubePoints_c[7][2]*(1-lambda_lines[i]);
                  r_point = 1;
                  s_point = 1-lambda_lines[i];
                  t_point = 0;
                  break;
                case 10: //7-6
                  x_point = cubePoints_c[7][0]*lambda_lines[i] + cubePoints_c[6][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[7][1]*lambda_lines[i] + cubePoints_c[6][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[7][2]*lambda_lines[i] + cubePoints_c[6][2]*(1-lambda_lines[i]);
                  r_point = 1;
                  s_point = 1;
                  t_point = 1-lambda_lines[i];
                  break;
                case 11: //6-2
                  x_point = cubePoints_c[6][0]*lambda_lines[i] + cubePoints_c[2][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[6][1]*lambda_lines[i] + cubePoints_c[2][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[6][2]*lambda_lines[i] + cubePoints_c[2][2]*(1-lambda_lines[i]);
                  r_point = 1;
                  s_point = lambda_lines[i];
                  t_point = 1;
                  break;
                }
              intersectionPoints[intersectionPointsIdx][0] = x_point;
              intersectionPoints[intersectionPointsIdx][1] = y_point;
              intersectionPoints[intersectionPointsIdx][2] = z_point;
              intersectionPoints[intersectionPointsIdx][3] = r_point;
              intersectionPoints[intersectionPointsIdx][4] = s_point;
              intersectionPoints[intersectionPointsIdx][5] = t_point;
              intersectionPointsIdx++;

              //Draws spheres in the intersection points
              if(0){
                glPushMatrix();
                glTranslatef(x_point, y_point, z_point);
                glutWireSphere(5,10,10);
                glPopMatrix();
              }
            }
        }

      //Find the average of the position
      GLfloat x_average = 0;
      GLfloat y_average = 0;
      for(int i = 0; i < intersectionPointsIdx; i++)
        {
          x_average += intersectionPoints[i][0];
          y_average += intersectionPoints[i][1];
        }
      x_average = x_average / intersectionPointsIdx;
      y_average = y_average / intersectionPointsIdx;

      //Rank the points according to their angle (to display them in order)
      GLfloat points_angles[intersectionPointsIdx];
      for(int i = 0; i < intersectionPointsIdx; i++)
        {
          points_angles[i] = atan2(intersectionPoints[i][1]-y_average, intersectionPoints[i][0]-x_average);
          if(points_angles[i] < 0)
            points_angles[i] = points_angles[i] + 2*3.14159;
        }
      int indexes[intersectionPointsIdx];
      for(int i = 0; i < intersectionPointsIdx; i++)
        {
          GLfloat min_angle = 1e3;
          int min_index = 15;
          for(int j = 0; j < intersectionPointsIdx; j++)
            {
              if(points_angles[j] < min_angle)
                {
                  min_angle = points_angles[j];
                  min_index = j;
                }
            }
          indexes[i] = min_index;
          points_angles[min_index] = 1e3;
        }

      if(min_max==0)
        glColor3f(1.0,1.0,1.0);
      if(min_max==1)
        glColor3f(0.0,0.0,1.0);
      if(min_max==2)
        glColor3f(0.0,1.0,0.0);
      if(min_max==3)
        glColor3f(1.0,1.0,1.0);


      glEnable(GL_BLEND);
      if(min_max == 0)
        glBlendEquation(GL_MIN);
//         glBlendEquation(GL_FUNC_REVERSE_SUBTRACT);
//         glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
      else
        glBlendEquation(GL_MAX);

      glEnable(GL_TEXTURE_3D);
      glBindTexture(GL_TEXTURE_3D, wholeTexture);

      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

      //All the previous was preparation, here with draw the poligon
      glBegin(GL_POLYGON);
//       for(int i = 0; i < intersectionPointsIdx; i++)
      for(int i = intersectionPointsIdx-1; i >=0; i--)
        {
          glTexCoord3f(intersectionPoints[indexes[i]][3],intersectionPoints[indexes[i]][4],intersectionPoints[indexes[i]][5]);
          glVertex3f(intersectionPoints[indexes[i]][0],intersectionPoints[indexes[i]][1],intersectionPoints[indexes[i]][2]);
        }
      glEnd();

      glDisable(GL_TEXTURE_3D);
      glDisable(GL_BLEND);

      //Draws an sphere on all the intersection points
      if(false)
        {
          glColor3f(0.0,1.0,1.0);
          for(int i = 0; i < intersectionPointsIdx; i++)
            {
              glPushMatrix();
              glTranslatef(intersectionPoints[indexes[i]][0],intersectionPoints[indexes[i]][1],intersectionPoints[indexes[i]][2]);
              glutSolidSphere(1,5,5);
              glPopMatrix();
            }
        }

      //Draws the texture coordinates of the intersection points
      if(false)
        {
          glColor3f(0.0,0.0,0.0);
          for(int i = 0; i < intersectionPointsIdx; i++)
            {
              glPushMatrix();
              glTranslatef(intersectionPoints[indexes[i]][0],intersectionPoints[indexes[i]][1],intersectionPoints[indexes[i]][2]);
              render_string("(%.2f %.2f %.2f)", intersectionPoints[indexes[i]][3],intersectionPoints[indexes[i]][4],intersectionPoints[indexes[i]][5]);
              glPopMatrix();
            }
          glColor3f(1.0,1.0,1.0);
         }
    } //depth loop

  //Put back the modelView matrix
  glMultMatrixf(pModelViewMatrix);
}

template <class T, class U>
void Cube<T,U>::draw(int x0, int y0, int z0, int x1, int y1, int z1, float rotx, float roty, float nPlanes, int min_max, float threshold)
{

//   draw_orientation_grid(false);

  if( (x1<x0) || (y1<y0) || (z1<z0))
    return;


  //Checks if I need to reload the 3D texture
  if(x1 > cubeWidth) x1 = cubeWidth;
  if(y1 > cubeHeight) y1 = cubeHeight;
  if(z1 > cubeDepth) z1 = cubeDepth;
//   x1 = min(512,x1-x0);
//   y1 = min(512,y1-y0);
//   z1 = min(512,z1-z0);
  int width = min(512, abs(x1-x0));
  int height = min(512, abs(y1-y0));
  int depth = min((int)cubeDepth, abs(z1-z0));

  if( (x0!=x0_old) || (x1!=x1_old) ||
      (y0!=y0_old) || (y1!=y1_old) ||
      (z0!=z0_old) || (z1!=z1_old) ||
      (threshold_old!=threshold))
    {
      x0_old = x0;
      y0_old = y0;
      z0_old = z0;
      x1_old = x1;
      y1_old = y1;
      z1_old = z1;
      threshold_old = threshold;

      //And now we reload the new texture
      glEnable(GL_TEXTURE_3D);
      glBindTexture(GL_TEXTURE_3D, wholeTexture);
      glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
      glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);
      glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP);
      glTexParameterf(GL_TEXTURE_3D,  GL_TEXTURE_PRIORITY, 1.0);
      GLfloat border_color[4];
      for(int i = 0; i < 4; i++)
        border_color[i] = 1.0;
      glTexParameterfv(GL_TEXTURE_3D, GL_TEXTURE_BORDER_COLOR, border_color);

      printf("P0=[%i %i %i] P1=[%i %i %i] D=[%i %i %i]\n", x0, y0, z0, x1, y1, z1, width, height, depth);
      if(sizeof(T)==1){
        uchar* texels =(uchar*)( malloc(width*height*depth*sizeof(uchar)));
        float x_step = max(1.0,fabs((float)x1-(float)x0)/512);
        float y_step = max(1.0,fabs((float)y1-(float)y0)/512);

        //       x_step = 1.0;
        //       y_step = 1.0;

        float x = 0;
        float y = 0;
        float z = 0;

        for(int z_t = 0; z_t < depth; z_t++)
          {
            for(int y_t = 0; y_t < height; y_t++)
              {
                for(int x_t = 0; x_t < width; x_t++)
                  {
                    x = x0 + x_step*x_t;
                    y = y0 + y_step*y_t;
                    z = z0 + z_t;
                    if(threshold == -1e6)
                      texels[z_t*width*height + y_t*width + x_t] = at2((int)x,(int)y,(int)z);
                    else{
                      if(at2((int)x,(int)y,(int)z) < threshold)
                        texels[z_t*width*height + y_t*width + x_t] = at2((int)x,(int)y,(int)z);
                      else
                        texels[z_t*width*height + y_t*width + x_t] = 255;
                    }
                  }
              }
            printf("#");
            fflush(stdout);
          }
        printf("]\n");
        glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE8, width, height, depth, 0, GL_LUMINANCE,
                     GL_UNSIGNED_BYTE, texels);
        free(texels);
      }
      if(sizeof(T)==4){
        float* texels =(float*)( malloc(width*height*depth*sizeof(float)));
        float x_step = max(1.0,fabs((float)x1-(float)x0)/512);
        float y_step = max(1.0,fabs((float)y1-(float)y0)/512);
        float x = 0;
        float y = 0;
        float z = 0;

        //Calculates the max and the min of the texture to be loaded
        float text_min = 1e6;
        float text_max = -1e6;
        for(int z_t = 0; z_t < depth; z_t++){
            for(int y_t = 0; y_t < height; y_t++){
                for(int x_t = 0; x_t < width; x_t++){
                    x = x0 + x_step*x_t;
                    y = y0 + y_step*y_t;
                    z = z0 + z_t;
                    if(at2((int)x,(int)y,(int)z) > text_max)
                      text_max = at2((int)x,(int)y,(int)z);
                    if(at2((int)x,(int)y,(int)z) < text_min)
                      text_min = at2((int)x,(int)y,(int)z);
                  }
              }
          }

        for(int z_t = 0; z_t < depth; z_t++){
            for(int y_t = 0; y_t < height; y_t++){
                for(int x_t = 0; x_t < width; x_t++){
                    x = x0 + x_step*x_t;
                    y = y0 + y_step*y_t;
                    z = z0 + z_t;
                    if(threshold == -1e6)
                      texels[z_t*width*height + y_t*width + x_t] = (at2((int)x,(int)y,(int)z)-text_min)/(text_max-text_min);
                    else{
                      if(at2((int)x,(int)y,(int)z) > threshold)
                        texels[z_t*width*height + y_t*width + x_t] = (at2((int)x,(int)y,(int)z)-threshold)/(text_max-threshold);
                      else
                        texels[z_t*width*height + y_t*width + x_t] = 0.0;
                    }
                  }
              }
            printf("#");
            fflush(stdout);
          }
        printf("]\n");
        glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE, width, height, depth, 0, GL_LUMINANCE,
                     GL_FLOAT, texels);
        free(texels);
      }


    }

  GLint max_texture_size = 0;
  glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_texture_size);

  //In object coordinates, we will draw the cube
  GLfloat widthStep = float(parentCubeWidth)*voxelWidth/2;
  GLfloat heightStep = float(parentCubeHeight)*voxelHeight/2;
  GLfloat depthStep = float(parentCubeDepth)*voxelDepth/2;

  //Draws a grid arround the edges
  glLineWidth(1.0);
  if(sizeof(T)==1)
    glColor3f(0.0,0.0,0.0);
  else
    glColor3f(1.0,1.0,1.0);
  glBegin(GL_LINE_STRIP);
  glVertex3f(-widthStep + x0*voxelWidth,  heightStep - y0*voxelHeight, -depthStep + z0*voxelDepth); //0
  glVertex3f(-widthStep + x0*voxelWidth,  heightStep - y0*voxelHeight, -depthStep + z1*voxelDepth); //1
  glVertex3f(-widthStep + x1*voxelWidth,  heightStep - y0*voxelHeight, -depthStep + z1*voxelDepth); //2
  glVertex3f(-widthStep + x1*voxelWidth, heightStep - y1*voxelHeight, -depthStep + z1*voxelDepth); //6
  glVertex3f(-widthStep + x1*voxelWidth, heightStep - y1*voxelHeight, -depthStep + z0*voxelDepth); //7
  glVertex3f(-widthStep + x0*voxelWidth, heightStep - y1*voxelHeight, -depthStep + z0*voxelDepth); //4
  glVertex3f(-widthStep + x0*voxelWidth,  heightStep - y0*voxelHeight, -depthStep + z0*voxelDepth); //0
  glVertex3f(-widthStep + x0*voxelWidth, heightStep - y1*voxelHeight, -depthStep + z0*voxelDepth); //4
  glVertex3f(-widthStep + x0*voxelWidth, heightStep - y1*voxelHeight, -depthStep + z1*voxelDepth); //5
  glVertex3f(-widthStep + x0*voxelWidth, +heightStep - y0*voxelHeight, -depthStep + z1*voxelDepth); //1
  glVertex3f(-widthStep + x0*voxelWidth, heightStep - y1*voxelHeight, -depthStep + z1*voxelDepth); //5
  glVertex3f(-widthStep + x1*voxelWidth, heightStep - y1*voxelHeight, -depthStep + z1*voxelDepth); //6
  glVertex3f(-widthStep + x1*voxelWidth,  heightStep - y0*voxelHeight, -depthStep + z1*voxelDepth); //2
  glVertex3f(-widthStep + x1*voxelWidth,  heightStep - y0*voxelHeight, -depthStep + z0*voxelDepth); //3
  glVertex3f(-widthStep + x1*voxelWidth, heightStep - y1*voxelHeight, -depthStep + z0*voxelDepth); //7
  glVertex3f(-widthStep + x1*voxelWidth,  heightStep - y0*voxelHeight, -depthStep + z0*voxelDepth); //3
  glVertex3f(-widthStep + x0*voxelWidth,  heightStep - y0*voxelHeight, -depthStep + z0*voxelDepth); //0
  glEnd();

  glLineWidth(2.0);
  glBegin(GL_LINE_STRIP);
  glVertex3f(-widthStep + x0*voxelWidth,  heightStep - y0*voxelHeight, -depthStep + z0*voxelDepth); //0
  glVertex3f(-widthStep + x1*voxelWidth,  heightStep - y0*voxelHeight, -depthStep + z0*voxelDepth); //3
  glVertex3f(-widthStep + x1*voxelWidth, heightStep - y1*voxelHeight, -depthStep + z0*voxelDepth); //7
  glVertex3f(-widthStep + x0*voxelWidth, heightStep - y1*voxelHeight, -depthStep + z0*voxelDepth); //4
  glVertex3f(-widthStep + x0*voxelWidth,  heightStep - y0*voxelHeight, -depthStep + z0*voxelDepth); //0
  glEnd();

  int nColTotal = nColToDraw + colOffset;
  int nRowTotal = nRowToDraw + rowOffset;

  int end_x = min((nColTotal+1)*max_texture_size, (int)parentCubeWidth);
  int end_y = min((nRowTotal+1)*max_texture_size, (int)parentCubeHeight);

  GLfloat pModelViewMatrix[16];
  glGetFloatv(GL_MODELVIEW_MATRIX, pModelViewMatrix);

  GLfloat** cubePoints = (GLfloat**)malloc(8*sizeof(GLfloat*));;

  cubePoints[0] = create_vector(-widthStep + x0*voxelWidth,
                                heightStep - y0*voxelHeight,
                                -depthStep + z0*voxelDepth,
                                1.0f);

  cubePoints[1] = create_vector(-widthStep + x0*voxelWidth,
                                heightStep - y0*voxelHeight,
                                -depthStep + z1*voxelDepth,
                                1.0f);

  cubePoints[2] = create_vector(-widthStep + x1*voxelWidth,
                                heightStep - y0*voxelHeight,
                                -depthStep + z1*voxelDepth,
                                1.0f);

  cubePoints[3] = create_vector(-widthStep + x1*voxelWidth,
                                heightStep - y0*voxelHeight,
                                -depthStep + z0*voxelDepth,
                                1.0f);

  cubePoints[4] = create_vector(-widthStep + x0*voxelWidth,
                                heightStep - y1*voxelHeight,
                                -depthStep + z0*voxelDepth,
                                1.0f);

  cubePoints[5] = create_vector(-widthStep + x0*voxelWidth,
                                heightStep - y1*voxelHeight,
                                -depthStep + z1*voxelDepth,
                                1.0f);

  cubePoints[6] = create_vector(-widthStep + x1*voxelWidth,
                                heightStep - y1*voxelHeight,
                                -depthStep + z1*voxelDepth,
                                1.0f);

  cubePoints[7] = create_vector(-widthStep + x1*voxelWidth,
                                heightStep - y1*voxelHeight,
                                -depthStep + z0*voxelDepth,
                                1.0f);

  // We will get the coordinates of the vertex of the cube in the modelview coordinates
  glLoadIdentity();
  GLfloat* cubePoints_c[8];
  glColor3f(0,0,0);
  for(int i=0; i < 8; i++)
    cubePoints_c[i] = matrix_vector_product(pModelViewMatrix, cubePoints[i]);

  //Draws the points numbers and the coordinates of the textures
  if(1){
    for(int i=0; i < 8; i++)
      {
        glColor3f(0.0,1.0,0.0);
        glPushMatrix();
        glTranslatef(cubePoints_c[i][0], cubePoints_c[i][1], cubePoints_c[i][2]);
        render_string("%i",i);
        glPopMatrix();
      }
    glPushMatrix();
    glTranslatef(cubePoints_c[0][0], cubePoints_c[0][1], cubePoints_c[0][2]);
    glRotatef(rotx, 1.0,0,0);
    glRotatef(roty, 0,1.0,0);
    //Draw The Z axis
    glColor3f(0.0, 0.0, 1.0);
    glutSolidCone(1.0, 10, 20, 20);
    glPushMatrix();
    glTranslatef(0,0,10);
    render_string("T");
    glPopMatrix();
    //Draw the x axis
    glColor3f(1.0, 0.0, 0.0);
    glRotatef(90, 0.0, 1.0, 0.0);
    glutSolidCone(1.0, 10, 20, 20);
    glPushMatrix();
    glTranslatef(0,0,10);
    render_string("R");
    glPopMatrix();
    //Draw the y axis
    glColor3f(0.0, 1.0, 0.0);
    glRotatef(90, 1.0, 0.0, 0.0);
    glutSolidCone(1.0, 10, 20, 20);
    glPushMatrix();
    glTranslatef(0,0,10);
    render_string("S");
    glPopMatrix();
    glPopMatrix();
    glColor3f(1.0, 1.0, 1.0);
  }

  //Find the closest and furthest vertex of the square
  float closest_distance = 1e9;
  float furthest_distance= 0;
  int closest_point_idx = 0;
  int furthest_point_idx = 0;
  for(int i = 0; i < 8; i++)
    {
      float dist = cubePoints_c[i][0]*cubePoints_c[i][0] + cubePoints_c[i][1]*cubePoints_c[i][1] + cubePoints_c[i][2]*cubePoints_c[i][2];
      if(dist < closest_distance)
        {
          closest_distance = dist;
          closest_point_idx = i;
        }
      if(dist > furthest_distance)
        {
          furthest_distance = dist;
          furthest_point_idx = i;
        }
    }

  //Draws a sphere in the furthest and closest point of the cube
  if(0){
    glPushMatrix();
    glTranslatef(cubePoints_c[closest_point_idx][0], cubePoints_c[closest_point_idx][1], cubePoints_c[closest_point_idx][2]);
    glColor3f(0.0,1.0,0.0);
    glutWireSphere(5,10,10);
    glPopMatrix();

    glPushMatrix();
    glTranslatef(cubePoints_c[furthest_point_idx][0], cubePoints_c[furthest_point_idx][1], cubePoints_c[furthest_point_idx][2]);
    glColor3f(0.0,0.0,1.0);
    glutWireSphere(5,10,10);
    glPopMatrix();
  }

//   printf("%f\n", cubePoints_c[furthest_point_idx][2] - cubePoints_c[closest_point_idx][2]);
  //Draws the cube
  for(float depth = 0/nPlanes; depth <= 1.0; depth+=1.0/nPlanes)
    {
      float z_plane = (cubePoints_c[furthest_point_idx][2]*(1-depth) + depth*cubePoints_c[closest_point_idx][2]);
      //Find the lines that intersect with the plane. For that we will define the lines and find the intersection of the line with the point
      GLfloat lambda_lines[12];
      if( ((cubePoints_c[1][2] > z_plane) && (cubePoints_c[0][2] > z_plane)) ||
          ((cubePoints_c[1][2] < z_plane) && (cubePoints_c[0][2] < z_plane)) )
            lambda_lines[0] = -1;
            else
            lambda_lines[0 ] = (z_plane - cubePoints_c[1][2]) / (cubePoints_c[0][2] - cubePoints_c[1][2]); //0-1

      if( ((cubePoints_c[3][2] > z_plane) && (cubePoints_c[0][2] > z_plane)) ||
          ((cubePoints_c[3][2] < z_plane) && (cubePoints_c[0][2] < z_plane)) )
            lambda_lines[1] = -1;
            else
            lambda_lines[1 ] = (z_plane - cubePoints_c[3][2]) / (cubePoints_c[0][2] - cubePoints_c[3][2]); //0-3

      if( ((cubePoints_c[4][2] > z_plane) && (cubePoints_c[0][2] > z_plane)) ||
          ((cubePoints_c[4][2] < z_plane) && (cubePoints_c[0][2] < z_plane)) )
            lambda_lines[2] = -1;
            else
            lambda_lines[2 ] = (z_plane - cubePoints_c[4][2]) / (cubePoints_c[0][2] - cubePoints_c[4][2]); //0-4

      if( ((cubePoints_c[7][2] > z_plane) && (cubePoints_c[4][2] > z_plane)) ||
          ((cubePoints_c[7][2] < z_plane) && (cubePoints_c[4][2] < z_plane)))
            lambda_lines[3] = -1;
            else
            lambda_lines[3 ] = (z_plane - cubePoints_c[7][2]) / (cubePoints_c[4][2] - cubePoints_c[7][2]); //4-7

      if( ((cubePoints_c[5][2] > z_plane) && (cubePoints_c[4][2] > z_plane)) ||
          ((cubePoints_c[5][2] < z_plane) && (cubePoints_c[4][2] < z_plane)))
            lambda_lines[4] = -1;
            else
            lambda_lines[4 ] = (z_plane - cubePoints_c[5][2]) / (cubePoints_c[4][2] - cubePoints_c[5][2]); //4-5

      if( ((cubePoints_c[1][2] > z_plane) && (cubePoints_c[2][2] > z_plane)) ||
          ((cubePoints_c[1][2] < z_plane) && (cubePoints_c[2][2] < z_plane)))
            lambda_lines[5] = -1;
            else
            lambda_lines[5 ] = (z_plane - cubePoints_c[2][2]) / (cubePoints_c[1][2] - cubePoints_c[2][2]); //1-2

      if( ((cubePoints_c[1][2] > z_plane) && (cubePoints_c[5][2] > z_plane)) ||
          ((cubePoints_c[1][2] < z_plane) && (cubePoints_c[5][2] < z_plane)))
            lambda_lines[6] = -1;
            else
            lambda_lines[6 ] = (z_plane - cubePoints_c[5][2]) / (cubePoints_c[1][2] - cubePoints_c[5][2]); //1-5

      if( ((cubePoints_c[6][2] > z_plane) && (cubePoints_c[5][2] > z_plane)) ||
          ((cubePoints_c[6][2] < z_plane) && (cubePoints_c[5][2] < z_plane)))
            lambda_lines[7] = -1;
            else
            lambda_lines[7 ] = (z_plane - cubePoints_c[6][2]) / (cubePoints_c[5][2] - cubePoints_c[6][2]); //5-6

      if( ((cubePoints_c[2][2] > z_plane) && (cubePoints_c[3][2] > z_plane)) ||
          ((cubePoints_c[2][2] < z_plane) && (cubePoints_c[3][2] < z_plane)))
            lambda_lines[8] = -1;
            else
            lambda_lines[8 ] = (z_plane - cubePoints_c[2][2]) / (cubePoints_c[3][2] - cubePoints_c[2][2]); //3-2

      if( ((cubePoints_c[7][2] > z_plane) && (cubePoints_c[3][2] > z_plane)) ||
          ((cubePoints_c[7][2] < z_plane) && (cubePoints_c[3][2] < z_plane)))
            lambda_lines[9] = -1;
            else
            lambda_lines[9 ] = (z_plane - cubePoints_c[7][2]) / (cubePoints_c[3][2] - cubePoints_c[7][2]); //3-7

      if( ((cubePoints_c[6][2] > z_plane) && (cubePoints_c[7][2] > z_plane)) ||
          ((cubePoints_c[6][2] < z_plane) && (cubePoints_c[7][2] < z_plane)))
            lambda_lines[10] = -1;
            else
            lambda_lines[10] = (z_plane - cubePoints_c[6][2]) / (cubePoints_c[7][2] - cubePoints_c[6][2]); //7-6

      if( ((cubePoints_c[2][2] > z_plane) && (cubePoints_c[6][2] > z_plane)) ||
          ((cubePoints_c[2][2] < z_plane) && (cubePoints_c[6][2] < z_plane)))
            lambda_lines[11] = -1;
            else
            lambda_lines[11] = (z_plane - cubePoints_c[2][2]) / (cubePoints_c[6][2] - cubePoints_c[2][2]); //6-2

      // We will store the point and texture coordinates of the points that we will draw afterwards
      //There is at maximum five intersections -> therefore we will define an array of five points
      GLfloat intersectionPoints[5][6];
      int intersectionPointsIdx = 0;
      for(int i = 0; i < 12; i++)
        {
          if( (lambda_lines[i] > 0) && (lambda_lines[i] < 1))
            {
              float x_point = 0;
              float y_point = 0;
              float z_point = 0;
              float r_point = 0;
              float s_point = 0;
              float t_point = 0;
              switch(i)
                {
                case 0: //0-1
                  x_point = cubePoints_c[0][0]*lambda_lines[i] + cubePoints_c[1][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[0][1]*lambda_lines[i] + cubePoints_c[1][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[0][2]*lambda_lines[i] + cubePoints_c[1][2]*(1-lambda_lines[i]);
                  r_point = 0;
                  s_point = 0;
                  t_point = (1-lambda_lines[i]);
                  break;
                case 1: //0-3
                  x_point = cubePoints_c[0][0]*lambda_lines[i] + cubePoints_c[3][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[0][1]*lambda_lines[i] + cubePoints_c[3][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[0][2]*lambda_lines[i] + cubePoints_c[3][2]*(1-lambda_lines[i]);
                  r_point = 1-lambda_lines[i];
                  s_point = 0;
                  t_point = 0;
                  break;
                case 2: //0-4
                  x_point = cubePoints_c[0][0]*lambda_lines[i] + cubePoints_c[4][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[0][1]*lambda_lines[i] + cubePoints_c[4][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[0][2]*lambda_lines[i] + cubePoints_c[4][2]*(1-lambda_lines[i]);
                  r_point = 0;
                  s_point = 1-lambda_lines[i];
                  t_point = 0;
                  break;
                case 3: //4-7
                  x_point = cubePoints_c[4][0]*lambda_lines[i] + cubePoints_c[7][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[4][1]*lambda_lines[i] + cubePoints_c[7][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[4][2]*lambda_lines[i] + cubePoints_c[7][2]*(1-lambda_lines[i]);
                  r_point = 1-lambda_lines[i];
                  s_point = 1;
                  t_point = 0;
                  break;
                case 4: //4-5
                  x_point = cubePoints_c[4][0]*lambda_lines[i] + cubePoints_c[5][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[4][1]*lambda_lines[i] + cubePoints_c[5][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[4][2]*lambda_lines[i] + cubePoints_c[5][2]*(1-lambda_lines[i]);
                  r_point = 0;
                  s_point = 1;
                  t_point = 1-lambda_lines[i];
                  break;
                case 5: //1-2
                  x_point = cubePoints_c[1][0]*lambda_lines[i] + cubePoints_c[2][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[1][1]*lambda_lines[i] + cubePoints_c[2][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[1][2]*lambda_lines[i] + cubePoints_c[2][2]*(1-lambda_lines[i]);
                  r_point = 1-lambda_lines[i];
                  s_point = 0;
                  t_point = 1;
                  break;
                case 6: //1-5
                  x_point = cubePoints_c[1][0]*lambda_lines[i] + cubePoints_c[5][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[1][1]*lambda_lines[i] + cubePoints_c[5][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[1][2]*lambda_lines[i] + cubePoints_c[5][2]*(1-lambda_lines[i]);
                  r_point = 0;
                  s_point = 1-lambda_lines[i];
                  t_point = 1;
                  break;
                case 7: //5-6
                  x_point = cubePoints_c[5][0]*lambda_lines[i] + cubePoints_c[6][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[5][1]*lambda_lines[i] + cubePoints_c[6][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[5][2]*lambda_lines[i] + cubePoints_c[6][2]*(1-lambda_lines[i]);
                  r_point = 1-lambda_lines[i];
                  s_point = 1;
                  t_point = 1;
                  break;
                case 8: //3-2
                  x_point = cubePoints_c[3][0]*lambda_lines[i] + cubePoints_c[2][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[3][1]*lambda_lines[i] + cubePoints_c[2][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[3][2]*lambda_lines[i] + cubePoints_c[2][2]*(1-lambda_lines[i]);
                  r_point = 1;
                  s_point = 0;
                  t_point = 1-lambda_lines[i];
                  break;
                case 9: //3-7
                  x_point = cubePoints_c[3][0]*lambda_lines[i] + cubePoints_c[7][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[3][1]*lambda_lines[i] + cubePoints_c[7][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[3][2]*lambda_lines[i] + cubePoints_c[7][2]*(1-lambda_lines[i]);
                  r_point = 1;
                  s_point = 1-lambda_lines[i];
                  t_point = 0;
                  break;
                case 10: //7-6
                  x_point = cubePoints_c[7][0]*lambda_lines[i] + cubePoints_c[6][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[7][1]*lambda_lines[i] + cubePoints_c[6][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[7][2]*lambda_lines[i] + cubePoints_c[6][2]*(1-lambda_lines[i]);
                  r_point = 1;
                  s_point = 1;
                  t_point = 1-lambda_lines[i];
                  break;
                case 11: //6-2
                  x_point = cubePoints_c[6][0]*lambda_lines[i] + cubePoints_c[2][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[6][1]*lambda_lines[i] + cubePoints_c[2][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[6][2]*lambda_lines[i] + cubePoints_c[2][2]*(1-lambda_lines[i]);
                  r_point = 1;
                  s_point = lambda_lines[i];
                  t_point = 1;
                  break;
                }
              intersectionPoints[intersectionPointsIdx][0] = x_point;
              intersectionPoints[intersectionPointsIdx][1] = y_point;
              intersectionPoints[intersectionPointsIdx][2] = z_point;
              intersectionPoints[intersectionPointsIdx][3] = r_point;
              intersectionPoints[intersectionPointsIdx][4] = s_point;
              intersectionPoints[intersectionPointsIdx][5] = t_point;
              intersectionPointsIdx++;

              //Draws spheres in the intersection points
              if(0){
                glPushMatrix();
                glTranslatef(x_point, y_point, z_point);
                glutWireSphere(5,10,10);
                glPopMatrix();
              }
            }
        }

      //Find the average of the position
      GLfloat x_average = 0;
      GLfloat y_average = 0;
      for(int i = 0; i < intersectionPointsIdx; i++)
        {
          x_average += intersectionPoints[i][0];
          y_average += intersectionPoints[i][1];
        }
      x_average = x_average / intersectionPointsIdx;
      y_average = y_average / intersectionPointsIdx;

      //Rank the points according to their angle (to display them in order)
      GLfloat points_angles[intersectionPointsIdx];
      for(int i = 0; i < intersectionPointsIdx; i++)
        {
          points_angles[i] = atan2(intersectionPoints[i][1]-y_average, intersectionPoints[i][0]-x_average);
          if(points_angles[i] < 0)
            points_angles[i] = points_angles[i] + 2*3.14159;
        }
      int indexes[intersectionPointsIdx];
      for(int i = 0; i < intersectionPointsIdx; i++)
        {
          GLfloat min_angle = 1e3;
          int min_index = 15;
          for(int j = 0; j < intersectionPointsIdx; j++)
            {
              if(points_angles[j] < min_angle)
                {
                  min_angle = points_angles[j];
                  min_index = j;
                }
            }
          indexes[i] = min_index;
          points_angles[min_index] = 1e3;
        }

      if(min_max==0)
        glColor3f(1.0,1.0,1.0);
      if(min_max==1)
        glColor3f(0.0,0.0,1.0);
      if(min_max==2)
        glColor3f(0.0,1.0,0.0);
      if(min_max==3)
        glColor3f(1.0,1.0,1.0);


      glEnable(GL_BLEND);
      if(min_max == 0)
        glBlendEquation(GL_MIN);
//         glBlendEquation(GL_FUNC_REVERSE_SUBTRACT);
//         glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
      else
        glBlendEquation(GL_MAX);

      glEnable(GL_TEXTURE_3D);
      glBindTexture(GL_TEXTURE_3D, wholeTexture);

      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

      //All the previous was preparation, here with draw the poligon
      glBegin(GL_POLYGON);
//       for(int i = 0; i < intersectionPointsIdx; i++)
      for(int i = intersectionPointsIdx-1; i >=0; i--)
        {
          glTexCoord3f(intersectionPoints[indexes[i]][3],intersectionPoints[indexes[i]][4],intersectionPoints[indexes[i]][5]);
          glVertex3f(intersectionPoints[indexes[i]][0],intersectionPoints[indexes[i]][1],intersectionPoints[indexes[i]][2]);
        }
      glEnd();

      glDisable(GL_TEXTURE_3D);
      glDisable(GL_BLEND);

      //Draws an sphere on all the intersection points
      if(false)
        {
          glColor3f(0.0,1.0,1.0);
          for(int i = 0; i < intersectionPointsIdx; i++)
            {
              glPushMatrix();
              glTranslatef(intersectionPoints[indexes[i]][0],intersectionPoints[indexes[i]][1],intersectionPoints[indexes[i]][2]);
              glutSolidSphere(1,5,5);
              glPopMatrix();
            }
        }

      //Draws the texture coordinates of the intersection points
      if(false)
        {
          glColor3f(0.0,0.0,0.0);
          for(int i = 0; i < intersectionPointsIdx; i++)
            {
              glPushMatrix();
              glTranslatef(intersectionPoints[indexes[i]][0],intersectionPoints[indexes[i]][1],intersectionPoints[indexes[i]][2]);
              render_string("(%.2f %.2f %.2f)", intersectionPoints[indexes[i]][3],intersectionPoints[indexes[i]][4],intersectionPoints[indexes[i]][5]);
              glPopMatrix();
            }
          glColor3f(1.0,1.0,1.0);
         }
    } //depth loop

  //Put back the modelView matrix
  glMultMatrixf(pModelViewMatrix);


}




#endif
