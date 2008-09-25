
/** Non parametrized cube class.*/


#ifndef CUBE_P_H_
#define CUBE_P_H_

#include "Cube.h"

template<typename T, typename U> class Cube;

class Cube_P : public VisibleE
{
public:
  // Dimension of the voxels in micrometers
  float voxelWidth;
  float voxelHeight;
  float voxelDepth;

  // Dimensions of the cube in voxels. Defined as long to allow for big cubes
  ulong cubeWidth;
  ulong cubeHeight;
  ulong cubeDepth;

  // Offset of the cube in micrometers
  float x_offset;
  float y_offset;
  float z_offset;

  //More variables, to know the last time it was drawn between two points
  int x0_old;
  int y0_old;
  int z0_old;
  int x1_old;
  int y1_old;
  int z1_old;
  float threshold_old;

  //3D textures are constraint to have a size of 2^m 2^n 2^k. The part of the texture that has no cube is filled with dark stuff. It needs to be cut when drawing the cube. This is done with the r,s,t max.
  double r_max;
  double s_max;
  double t_max;

  int nColToDraw;
  int nRowToDraw;

  string filenameVoxelData;
  string directory; //directory where the cube is
  string filenameParameters; //name of the parameters file

  //Type of data that the cube is. Type = "uchar, float..."
  string type;

  Cube_P() : VisibleE(){}

  virtual void print_size()=0;

  virtual void micrometersToIndexes(vector<float>& micrometers, vector< int >& indexes)=0;

  virtual void indexesToMicrometers(vector< int >& indexes, vector< float >& micrometers)=0;

  virtual void load_texture_brick(int row, int col)=0;

  virtual void draw
  (float rotx, float roty, float nPlanes, int min_max, int microm_voxels)=0;

  virtual void draw_layer_tile_XY(float depth, int color=0)=0;

  virtual void draw_layer_tile_XZ(float depth, int color=0)=0;

  virtual void draw_layer_tile_YZ(float depth, int color=0)=0;

  virtual void min_max(float* min, float* max)=0;

  virtual void print_statistics(string name = "")=0;

  virtual void histogram(string name = "")=0;

  virtual void save_as_image_stack(string dirname)=0;

  virtual void draw()=0;

  virtual  vector< vector< int > > decimate
   (float threshold, int window_xy = 8, int window_z = 3, string filemane = "",
    bool save_boosting_response = false)=0;

  virtual vector< vector< int > > decimate_log
  (float threshold, int window_xy = 8, int window_z = 3, string filemane = "",
     bool save_boosting_response = false)=0;

  /** Produces a vector of the NMS in the layer indicated.*/
  virtual vector< vector< int > > decimate_layer
  (int nLayer, float threshold, int window_xy, string filename)=0;

  ~Cube_P(){}

};




#endif
