#ifndef CUBE_T_H_
#define CUBE_T_H_

/** Cube_T represents a 4D cube (3D+T). In it's first implementation it will contain a vector     of cubes. Something simple and hacky (efficient in coding time. The file extension of
    this class will be "cbt" (cube time)*/
#include "Cube_P.h"
#include <fstream>
#include "utils.h"

class Cube_T : public Cube_P
{
public:
  vector< Cube_P* > cubes;

  // Hack to put kevin's gt in there
  vector< vector< double > > gtData;

  int timeStep;

  bool d_halo;

  bool d_allInOne;

  bool d_gt;

  Cube_T(string filename);

  void print_size();

  void load_texture_brick(int row, int col, float scale=1.0, float max = 0.0, float min = 0.0);

  void draw();

  void draw
  (float rotx, float roty, float nPlanes, int min_max, int microm_voxels);

  void draw
  (int x0, int y0, int z0, int x1, int y1, int z1,
   float rotx, float roty, float nPlanes, int min_max, float threshold);

  void drawgt();

  void draw_layer_tile_XY(float depth, int color=0);

  void draw_layer_tile_XZ(float depth, int color=0);

  void draw_layer_tile_YZ(float depth, int color=0);

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

  ~Cube_T(){}

  virtual string className(){
    return "Cube_T";
  }

  float getValueAsFloat(int x, int y, int z){return 0.0;}


};



#endif
