
/** The mother of all cubes - Cube.h has been modified to make it even more generic.*/

/** We will have some tuning of the parameters for the rendering here
    as defines. At some point they should be put as variables, so that
    they can be accessed from the program.
 */
#define D_MAX_TEXTURE_SIZE      512
// #define D_MAX_TEXTURE_SIZE      1024
// #define D_MAX_TEXTURE_SIZE      2048
// #define D_TEXTURE_INTERPOLATION GL_LINEAR
#define D_TEXTURE_INTERPOLATION GL_NEAREST

#ifndef CUBE_P_H_
#define CUBE_P_H_

#include "VisibleE.h"
#include "utils.h"

 class Cube_P : public VisibleE
{
public:
  // Dimension of the voxels in micrometers
  float voxelWidth;
  float voxelHeight;
  float voxelDepth;

  // Dimensions of the cube in voxels. Defined as long to allow for big cubes
  int cubeWidth;
  int cubeHeight;
  int cubeDepth;

  string directory; //directory where the cube is
  string filenameVoxelData;
  string filenameParameters; //name of the parameters file

  //Type of data that the cube is. Type = "uchar, float..."
  string type;

  bool dummy; //Used for something by Aurelien

  Cube_P() : VisibleE(),dummy(false)
  {
    wholeTexture = 0; v_r = 1.0; v_g = 1.0; v_b = 1.0;
    v_draw_projection = true; v_blend = true; v_enable_depth_test = true;
  }

  virtual void print_size()=0;

  // Moved to here from Cube
  //////////////////////////////////////////////////////////////////////
  // BASIC FUNCTIONS
  //////////////////////////////////////////////////////////////////////

  void micrometersToIndexes(vector<float>& micrometers, vector< int >& indexes);

  void micrometersToIndexes3(float mx, float my, float mz, int& x, int& y, int& z);

  void indexesToMicrometers(vector< int >& indexes, vector< float >& micrometers);

  void indexesToMicrometers3(int x, int y, int z, float& mx, float& my, float& mz);

  virtual float get(int x, int y, int z)=0;

  virtual void save_as_image_stack(string dirname="")=0;

  ~Cube_P(){}

  virtual string className(){
    return "Cube";
  }

  //////////////////////////////////////////////////////////////////////
  // RENDERING FUNCTIONS AND VARIABLES
  //////////////////////////////////////////////////////////////////////

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

  //3D textures are constraint to have a size of 2^m 2^n 2^k. The part of the
  //texture that has no cube is filled with dark stuff. It needs to be cut when
  //drawing the cube. This is done with the r,s,t max.
  double r_max;
  double s_max;
  double t_max;
  GLuint wholeTexture;

  int nColToDraw;
  int nRowToDraw;

  // OpenGL variables
  GLubyte*** alphas; // alpha values used to manage transparency
  GLubyte min_alpha;
  GLubyte max_alpha;

  // blend function
  typedef enum {
    MIN_MAX, ALPHA_TEST
  } eBlendFunction;
  eBlendFunction blendFunction;


  // Stolen from the original cube.
  virtual void load_texture_brick(int row, int col, float scale=1.0,
                                  float min = 0.0, float max = 0.0)=0;

  virtual void draw
  (float nPlanes, int min_max, int microm_voxels);

  virtual void draw();

  // To be implemented
  virtual void draw
  (int x0, int y0, int z0, int x1, int y1, int z1,
   float rotx, float roty, float nPlanes, int min_max,
   float threshold){}

  virtual void draw_layer_tile_XY(float depth, int color=0);

  virtual void draw_layer_tile_XZ(float depth, int color=0);

  virtual void draw_layer_tile_YZ(float depth, int color=0);


  /** Product between a matrix and a vector.*/
  GLfloat* matrix_vector_product(GLfloat* matrix, GLfloat* vector);

  /** Creates a vector with values x, y, z.*/
  GLfloat* create_vector(float x, float y, float z, float w);

  //////////////////////////////////////////////////////////////////////
  // STATISTICAL FUNCTIONS
  //////////////////////////////////////////////////////////////////////
  virtual void min_max(float& min, float& max);

  virtual void histogram(string name="");

  virtual void print_statistics(string name="");


  //////////////////////////////////////////////////////////////////////
  // IMAGE ANALYSIS RELATED FUNCTIONS
  //////////////////////////////////////////////////////////////////////
  virtual Cube_P* threshold(float thres, string outputName="output",
                    bool putHigherValuesTo = false, bool putLowerValuesTo = true,
                    float highValue = 1, float lowValue = 0)=0;

  virtual  vector< vector< int > > decimate
   (float threshold, int window_xy = 8, int window_z = 3, string filemane = "",
    bool save_boosting_response = false)=0;

  virtual vector< vector< int > > decimate_log
  (float threshold, int window_xy = 8, int window_z = 3, string filemane = "",
     bool save_boosting_response = false)=0;

  /** Produces a vector of the NMS in the layer indicated.*/
  virtual vector< vector< int > > decimate_layer
  (int nLayer, float threshold, int window_xy, string filename)=0;

  virtual void allocate_alphas(int ni, int nj, int nk)=0;

  virtual void delete_alphas(int ni, int nj, int nk)=0;
};




#endif
