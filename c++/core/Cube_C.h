#ifndef CUBE_C_H_
#define CUBE_C_H_

/** Cube_C is a cube with color. It is forced to be of uchar type.*/

#include <fstream>
#include "utils.h"
#include "Cube.h"
#include "Cube_P.h"
#include "tiffio.h"

class Cube_C : public Cube_P
{
public:
  /** Strings where the data is stored*/
  string filenameVoxelDataR;
  string filenameVoxelDataG;
  string filenameVoxelDataB;

  vector< Cube<uchar, ulong>* > data;

  Cube_C() {}

  Cube_C(string filename);

  ~Cube_C(){}

  /** Methods inherited from Cube_P.*/
  void print_size();

  void load_texture_brick(int row, int col, float scale=1.0, float min = 0.0, float max = 0.0);

  void loadFromTIFFImage(string image);

  void min_max(float* min, float* max);

  Cube_P* threshold(float thres, string outputName="output",
                    bool putHigherValuesTo = false, bool putLowerValuesTo = true,
                    float highValue = 1, float lowValue = 0);

  void print_statistics(string name="");

  void histogram(string name="");

  void save_as_image_stack(string dirname="");

  vector< vector< int > > decimate
  (float threshold, int window_xy = 8, int window_z = 3, string filemane = "",
   bool save_boosting_response = false);

  vector< vector< int > > decimate_log
  (float threshold, int window_xy = 8, int window_z = 3, string filemane = "",
   bool save_boosting_response = false);

  /** Produces a vector of the NMS in the layer indicated.*/
  vector< vector< int > > decimate_layer
  (int nLayer, float threshold, int window_xy, string filename);

  void allocate_alphas(int ni, int nj, int nk);

  void delete_alphas(int ni, int nj, int nk);

  float get(int x, int y, int z);

  string className(){
    return "Cube_C";
  }


  /** Taken from Cube_aux.*/
  GLfloat* matrix_vector_product(GLfloat* m, GLfloat* v)
  {
    GLfloat* b = (GLfloat*)malloc(4*sizeof(GLfloat));
    b[3] =  m[3]*v[0] + m[7]*v[1] + m[11]*v[2] + m[15]*v[3];
    b[0] = (m[0]*v[0] + m[4]*v[1] + m[8 ]*v[2] + m[12]*v[3])/b[3];
    b[1] = (m[1]*v[0] + m[5]*v[1] + m[9 ]*v[2] + m[13]*v[3])/b[3];
    b[2] = (m[2]*v[0] + m[6]*v[1] + m[10]*v[2] + m[14]*v[3])/b[3];
    b[3] = 1;
    if(sizeof(b)/sizeof(GLfloat) < 2){
      GLfloat* b = (GLfloat*)malloc(4*sizeof(GLfloat));
      b[0]=0; b[1] =0; b[2] = 0; b[3]=0;
    }
    return b;
  }

  GLfloat* create_vector(GLfloat x, GLfloat y, GLfloat z, GLfloat w){
    GLfloat* b = (GLfloat*)malloc(4*sizeof(GLfloat));
    b[0]=x; b[1] = y; b[2] = z; b[3] = w;
    return b;
  }

};


#endif
