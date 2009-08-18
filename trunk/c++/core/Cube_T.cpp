#include "Cube_T.h"
#include "CubeFactory.h"

Cube_T::Cube_T(string filename){

  printf("Cube_T::loading from %s\n", filename.c_str());
  timeStep = 0;
  d_halo = false;
  d_gt = false;

  cubes.resize(0);
  string extension = getExtension(filename);
  if(extension != "cbt"){
    printf("Cube_T::error::the file %s does not end with cbt\n", filename.c_str());
    exit(0);
  }
  std::ifstream in(filename.c_str());
  if(!in.good())
    {
      printf("Cube_T::error::The file %s can not be opened\n",filename.c_str());
      exit(0);
    }
  string s;
  while(getline(in,s)){
    Cube_P* cb = CubeFactory::load(s);
    if(!cb){
      printf("Cube_T::error::The cubee %s can not be loaded\n",s.c_str());
      exit(0);
    }
    cubes.push_back(cb);
  }
  in.close();
  if(cubes.size()==0){
      printf("Cube_T::error::There is no cube loaded from %s\n",s.c_str());
      exit(0);
  }

  // Get all the information from cubes[0] (we assume all the cubes are the same and
  //   do not check for that (check needed if we want to make the code robust)
  this->voxelWidth  = cubes[0]->voxelWidth;
  this->voxelHeight = cubes[0]->voxelHeight;
  this->voxelDepth  = cubes[0]->voxelDepth;
  this->cubeWidth   = cubes[0]->cubeWidth;
  this->cubeHeight  = cubes[0]->cubeHeight;
  this->cubeDepth   = cubes[0]->cubeDepth;
  this->r_max       = cubes[0]->r_max;
  this->s_max       = cubes[0]->s_max;
  this->t_max       = cubes[0]->t_max;
  this->nColToDraw  = cubes[0]->nColToDraw;
  this->nRowToDraw  = cubes[0]->nRowToDraw;
  this->filenameVoxelData = "";
  this->directory          = getDirectoryFromPath(filename);
  this->filenameParameters = getNameFromPath(filename);
  this->type               = cubes[0]->type;

  string noExt = getNameFromPathWithoutExtension(filename);
  string gtName = getDirectoryFromPath(filename) +  noExt + ".gt";
  printf("The ground truth data should be in %s\n", gtName.c_str());
  if(fileExists(gtName)){
    gtData = loadMatrix(gtName);
  }
  printf("ans it's size is %i\n", gtData.size());

}


void Cube_T::load_texture_brick(int row, int col, float scale)
{
  for(int i = 0; i < cubes.size(); i++){
    cubes[i]->load_texture_brick(row, col, scale);
  }
}

void Cube_T::drawgt()
{
 //Draws the ground truth data
  if(d_gt){
    float mx1, my1, mz1, mx2, my2, mz2;
    for(int i = 0; i < gtData.size(); i++){
      if(gtData[i][6]-1 == (double)timeStep){
        indexesToMicrometers3(gtData[i][0],gtData[i][1],gtData[i][2],
                              mx1, my1, mz1);
        indexesToMicrometers3(gtData[i][0]+gtData[i][3],
                              gtData[i][1]+gtData[i][4],
                              gtData[i][2]+gtData[i][5],
                              mx2, my2, mz2);

        glColor3f(1.0,0.0,0.0);
        glPushMatrix();
        glTranslatef((mx1+mx2)/2, (my1+my2)/2, (mz1+mz2)/2);
        glutSolidSphere(1.0, 10, 10);
        glPopMatrix();
      }
    }
  }
}

void Cube_T::draw()
{
  // Draws the cube
  int nPlanes = 500;
  glDisable(GL_DEPTH_TEST);
  GLfloat pModelViewMatrix[16];
  glGetFloatv(GL_MODELVIEW_MATRIX, pModelViewMatrix);
  cubes[timeStep]->v_r = this->v_r;
  cubes[timeStep]->v_g = this->v_g;
  cubes[timeStep]->v_b = this->v_b;
  cubes[timeStep]->v_draw_projection = this->v_draw_projection;
  glPushMatrix();
  cubes[timeStep]->draw(0,0,nPlanes,this->v_draw_projection, 0);
  glPopMatrix();

  drawgt();

  // Draws the halo
  if(d_halo){
    if(timeStep - 1 >= 0){
      cubes[timeStep-1]->v_r = 0.6;
      cubes[timeStep-1]->v_g = 0;
      cubes[timeStep-1]->v_b = 0;
      cubes[timeStep-1]->v_draw_projection = this->v_draw_projection;
      glPushMatrix();
      glLoadMatrixf(pModelViewMatrix);
      cubes[timeStep-1]->draw(0,0,nPlanes,this->v_draw_projection, 0);
      glPopMatrix();
    }
    if(timeStep - 2 >= 0){
      cubes[timeStep-2]->v_r = 0;
      cubes[timeStep-2]->v_g = 0;
      cubes[timeStep-2]->v_b = 0.5;
      cubes[timeStep-2]->v_draw_projection = this->v_draw_projection;
      glPushMatrix();
      glLoadMatrixf(pModelViewMatrix);
      cubes[timeStep-2]->draw(0,0,nPlanes,this->v_draw_projection, 0);
      glPopMatrix();
    }
    if(timeStep - 3 >= 0){
      cubes[timeStep-3]->v_r = 0;
      cubes[timeStep-3]->v_g = 0.4;
      cubes[timeStep-3]->v_b = 0;
      cubes[timeStep-3]->v_draw_projection = this->v_draw_projection;
      glPushMatrix();
      glLoadMatrixf(pModelViewMatrix);
      cubes[timeStep-3]->draw(0,0,nPlanes,this->v_draw_projection, 0);
      glPopMatrix();
    }
  }


}

void Cube_T::draw
(float rotx, float roty, float nPlanes, int min_max, int microm_voxels)
{
  cubes[timeStep]->v_r = this->v_r;
  cubes[timeStep]->v_g = this->v_g;
  cubes[timeStep]->v_b = this->v_b;
  cubes[timeStep]->v_draw_projection = this->v_draw_projection;
  cubes[timeStep]->draw(rotx, roty, nPlanes, min_max, microm_voxels);
}


void Cube_T::draw
(int x0, int y0, int z0, int x1, int y1, int z1,
 float rotx, float roty, float nPlanes, int min_max, float threshold)
{
  cubes[timeStep]->v_r = this->v_r;
  cubes[timeStep]->v_g = this->v_g;
  cubes[timeStep]->v_b = this->v_b;
  cubes[timeStep]->v_draw_projection = this->v_draw_projection;
  cubes[timeStep]->draw(x0, y0, z0, x1, y1, z1,
                        rotx, roty, nPlanes, min_max, threshold);
}

void Cube_T::draw_layer_tile_XY(float depth, int color)
{
  cubes[timeStep]->v_r = this->v_r;
  cubes[timeStep]->v_g = this->v_g;
  cubes[timeStep]->v_b = this->v_b;
  cubes[timeStep]->v_draw_projection = this->v_draw_projection;
  cubes[timeStep]->draw_layer_tile_XY(depth, color);
  if(color==0)
    drawgt();
}

void Cube_T::draw_layer_tile_XZ(float depth, int color)
{
  cubes[timeStep]->v_r = this->v_r;
  cubes[timeStep]->v_g = this->v_g;
  cubes[timeStep]->v_b = this->v_b;
  cubes[timeStep]->v_draw_projection = this->v_draw_projection;
  cubes[timeStep]->draw_layer_tile_XZ(depth, color);
  if(color==0)
    drawgt();
}

void Cube_T::draw_layer_tile_YZ(float depth, int color)
{
  cubes[timeStep]->v_r = this->v_r;
  cubes[timeStep]->v_g = this->v_g;
  cubes[timeStep]->v_b = this->v_b;
  cubes[timeStep]->v_draw_projection = this->v_draw_projection;
  cubes[timeStep]->draw_layer_tile_YZ(depth, color);
  if(color==0)
    drawgt();
}

void Cube_T::min_max(float* min, float* max)
{
  cubes[timeStep]->min_max(min, max);
}

Cube_P* Cube_T::threshold(float thres, string outputName,
                          bool putHigherValuesTo, bool putLowerValuesTo,
                          float highValue, float lowValue)
{
  cubes[timeStep]->threshold(thres, outputName, putHigherValuesTo,
                             putLowerValuesTo, highValue, lowValue);
}

void Cube_T::print_statistics(string name)
{
  cubes[timeStep]->print_statistics(name);
}

void Cube_T::histogram(string name)
{
  cubes[timeStep]->histogram(name);
}

void Cube_T::save_as_image_stack(string dirname)
{
  cubes[timeStep]->save_as_image_stack(dirname);
}

vector< vector< int > > Cube_T::decimate
(float threshold, int window_xy, int window_z, string filemane,
 bool save_boosting_response)
{
  cubes[timeStep]->decimate(threshold, window_xy, window_z, filemane,
                            save_boosting_response);
}

vector< vector< int > > Cube_T::decimate_log
(float threshold, int window_xy, int window_z, string filemane,
 bool save_boosting_response)
{
  cubes[timeStep]->decimate_log(threshold, window_xy, window_z, filemane,
                                save_boosting_response);
}

  /** Produces a vector of the NMS in the layer indicated.*/
vector< vector< int > > Cube_T::decimate_layer
(int nLayer, float threshold, int window_xy, string filename)
{
  cubes[timeStep]->decimate_layer(nLayer, threshold, window_xy, filename);
}

void Cube_T::allocate_alphas(int ni, int nj, int nk)
{
  cubes[timeStep]->allocate_alphas(ni, nj, nk);
}

void Cube_T::delete_alphas(int ni, int nj, int nk)
{
  cubes[timeStep]->delete_alphas(ni, nj, nk);
}

void Cube_T::print_size()
{
  cubes[timeStep]->print_size();
}

