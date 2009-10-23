#include "Cube_P.h"


void Cube_P::micrometersToIndexes(vector< float >& micrometers, vector< int >& indexes)
{
  indexes.clear();
  indexes.push_back((int)(float(cubeWidth)/2 + micrometers[0]/voxelWidth));
  indexes.push_back((int)(float(cubeHeight)/2 - micrometers[1]/voxelHeight));
  indexes.push_back((int)(float(cubeDepth)/2 + micrometers[2]/voxelDepth));
}

/** Converts from 3d coordinates in micrometers to a position in indexes.*/

void Cube_P::micrometersToIndexes3(float mx, float my, float mz, int& x, int& y, int& z)
{
  x = (int)(float(cubeWidth)/2 + mx/voxelWidth);
  y = (int)(float(cubeHeight)/2 - my/voxelHeight);
  z = (int)(float(cubeDepth)/2 + mz/voxelDepth);
}


void Cube_P::indexesToMicrometers(vector< int >& indexes, vector< float >& micrometers)
{
  // micrometers[0] = (float)(-((int)cubeWidth)*voxelWidth/2   
                           // + indexes[0]*voxelWidth  + x_offset);
  // micrometers[1] = (float)( ((int)cubeHeight)*voxelHeight/2 
                            // - indexes[1]*voxelHeight - y_offset);
  // micrometers[2] = (float)(-((int)cubeDepth)*voxelDepth/2   
                           // + indexes[2]*voxelDepth  + z_offset);
  /*
  micrometers[0] = (float)(-((int)cubeWidth)*voxelWidth/2   
                           + indexes[0]*voxelWidth);
  micrometers[1] = (float)( ((int)cubeHeight)*voxelHeight/2 
                            - indexes[1]*voxelHeight);
  micrometers[2] = (float)(-((int)cubeDepth)*voxelDepth/2   
                           + indexes[2]*voxelDepth);
  */
  micrometers.clear();
  micrometers.push_back((float)(-((int)cubeWidth)*voxelWidth/2   
				 + indexes[0]*voxelWidth));
  micrometers.push_back((float)( ((int)cubeHeight)*voxelHeight/2 
				  - indexes[1]*voxelHeight));
  micrometers.push_back((float)(-((int)cubeDepth)*voxelDepth/2   
                           + indexes[2]*voxelDepth));
}


void Cube_P::indexesToMicrometers3(int x, int y, int z, float& mx, float& my, float& mz)
{
  mx = (float)(-((int)cubeWidth)*voxelWidth/2   
				 + x*voxelWidth);
  my = (float)( ((int)cubeHeight)*voxelHeight/2 
				  - y*voxelHeight);
  mz = (float)(-((int)cubeDepth)*voxelDepth/2   
               + (z+0.5)*voxelDepth);
}


