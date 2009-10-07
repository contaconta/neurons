#include "Cube_C.h"

Cube_C::Cube_C(string filenameParams)
{

  v_r = 1.0;
  v_g = 1.0;
  v_b = 1.0;

  printf("Loading cube_C\n");
  std::ifstream file(filenameParams.c_str());
  if(!file.good())
    printf("Cube_C::load_parameters: error loading the file %s\n", filenameParams.c_str());

  string name;
  string attribute;
  while(file.good())
    {
       file >> name;
       file >> attribute;
       if(!strcmp(name.c_str(), "filenameVoxelDataR"))
         filenameVoxelDataR = attribute;
       else if(!strcmp(name.c_str(), "filenameVoxelDataG"))
         filenameVoxelDataG = attribute;
       else if(!strcmp(name.c_str(), "filenameVoxelDataB"))
         filenameVoxelDataB = attribute;
       else if(!strcmp(name.c_str(), "type"))
         type = attribute;
       else
         printf("Cube_C::load_parameters: Attribute %s and value %s not known\n",
                name.c_str(), attribute.c_str());
     }
  if(type != "color"){
    printf("Cube_C called to load an nfo file that is not a Cube_C... exiting\n");
    exit(0);
  }
  if ( (filenameVoxelDataR == "") || (filenameVoxelDataG == "") ||
       (filenameVoxelDataB == "") )
    {
    printf("Cube_C one of the color channels is not defined... exiting\n");
    exit(0);
  }

  data.resize(0);
  data.push_back(new Cube<uchar, ulong>(filenameVoxelDataR));
  data.push_back(new Cube<uchar, ulong>(filenameVoxelDataG));
  data.push_back(new Cube<uchar, ulong>(filenameVoxelDataB));
  printf("Now all the cubes should be loaded -> %i\n", data.size());

  this->cubeHeight  = data[0]->cubeHeight;
  this->cubeDepth   = data[0]->cubeDepth;
  this->cubeWidth   = data[0]->cubeWidth;
  this->voxelHeight = data[0]->voxelHeight;
  this->voxelDepth  = data[0]->voxelDepth;
  this->voxelWidth  = data[0]->voxelWidth;
  // this->x_offset = data[0]->x_offset;
  // this->y_offset = data[0]->y_offset;
  // this->z_offset = data[0]->z_offset;
}


void Cube_C::load_texture_brick(int row, int col, float scale, float _min, float _max)
{

  printf("Cube_C, loading texture brick\n");
  nColToDraw = col;
  nRowToDraw = row;
  GLint max_texture_size = 0;
  glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_texture_size);
  max_texture_size = D_MAX_TEXTURE_SIZE;

  printf("Max texture size %i\n", max_texture_size);

  int limit_x = min((int)cubeWidth,
                    min((int)max_texture_size,
                        (int)cubeWidth - (nColToDraw*max_texture_size)));
  int limit_y = min((int)cubeHeight,
                    min((int)max_texture_size,
                        (int)cubeHeight - (nRowToDraw*max_texture_size)));
  int limit_z =  (int)min(max_texture_size, (int)cubeDepth);

  int texture_size_x = (int) pow(2.0, ceil(log((double)limit_x)/log(2.0)) );
  int texture_size_y = (int) pow(2.0, ceil(log((double)limit_y)/log(2.0)) );
  int texture_size_z = (int) pow(2.0, ceil(log((double)limit_z)/log(2.0)) );

  //Limit od the textures. They are object variables
  r_max = (double)limit_x/texture_size_x;
  s_max = (double)limit_y/texture_size_y;
  t_max = (double)limit_z/texture_size_z;

  printf("Load_texture_brick: texture size %i, limit_x = %i, limit_y = %i limit_z = %i\n               texture_size: x=%i y=%i z=%i, r_max=%f, s_max=%f, t_max=%f\n",
         max_texture_size, limit_x, limit_y, limit_z,
         texture_size_x, texture_size_y,texture_size_z,
         r_max,s_max,t_max);


  if( (limit_x<0) || (limit_y<0))
    {
      printf("Cube_C::load_texture_brick requested col %i row %i out of range"
             ", loading 0,0\n", nColToDraw, nRowToDraw);
      nColToDraw = 0;
      nRowToDraw = 0;
      limit_x = min(max_texture_size,(int)cubeWidth);
      limit_y = min(max_texture_size,(int)cubeHeight);
    }

  glEnable(GL_TEXTURE_3D);
  glBindTexture(GL_TEXTURE_3D, wholeTexture);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, D_TEXTURE_INTERPOLATION);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, D_TEXTURE_INTERPOLATION);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP);
  glTexParameterf(GL_TEXTURE_3D,  GL_TEXTURE_PRIORITY, 1.0);
  GLfloat border_color[4];
  for(int i = 0; i < 4; i++)
    border_color[i] = 1.0;
  glTexParameterfv(GL_TEXTURE_3D, GL_TEXTURE_BORDER_COLOR, border_color);

  int col_offset = (col)*max_texture_size;
  int row_offset = (row)*max_texture_size;

  int size_texture = max_texture_size;
  GLubyte voxel;

  int texture_size;
  texture_size = texture_size_x*texture_size_y*texture_size_z*3;
  GLubyte* texels =(GLubyte*)(malloc(texture_size*sizeof(GLubyte)));
  for(int t = 0; t < texture_size; t++)
    texels[t] = 0;

  int depth_y;
  int depth_z;
  for(int z = 0; z < limit_z; z++)
    {
      depth_z = z*texture_size_x*texture_size_y*3;
      for(int y = 0; y < limit_y; y++)
        {
          depth_y = y*texture_size_x*3;
          for(int x = 0; x < limit_x; x++)
            {
              voxel = (GLubyte)data[0]->at(col_offset+x,row_offset+y,z)*scale;
              texels[depth_z + depth_y + 3*x + 0] = voxel;
              voxel = (GLubyte)data[1]->at(col_offset+x,row_offset+y,z)*scale;
              texels[depth_z + depth_y + 3*x + 1] = voxel;
              voxel = (GLubyte)data[2]->at(col_offset+x,row_offset+y,z)*scale;
              texels[depth_z + depth_y + 3*x + 2] = voxel;
            }
        }
      printf("#");
      fflush(stdout);
    }
  printf("]\n");
  glTexImage3D(GL_TEXTURE_3D,
               0, // level
               GL_RGB,
               texture_size_x, texture_size_y, texture_size_z,
               0, //border
               GL_RGB,
               GL_UNSIGNED_BYTE,
               texels);
  free(texels);

  // glBindTexture(GL_TEXTURE_3D, wholeTexture);
}

void Cube_C::draw()
{
  draw(0,0,200,v_draw_projection,0);
}

void Cube_C::draw
(float rotx, float roty, float nPlanes, int min_max, int microm_voxels)
{
  //Parches one bug with the matrices
  int nMatrices = 0;

  GLint max_texture_size = 0;
  glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_texture_size);
  max_texture_size = D_MAX_TEXTURE_SIZE;

  // draw_orientation_grid(false, min_max);

  //In object coordinates, we will draw the cube
  GLfloat widthStep = float(cubeWidth)*voxelWidth/2;
  GLfloat heightStep = float(cubeHeight)*voxelHeight/2;
  GLfloat depthStep = float(cubeDepth)*voxelDepth/2;

  int nColTotal = nColToDraw;
  int nRowTotal = nRowToDraw;

  int end_x = min((nColTotal+1)*max_texture_size, (int)cubeWidth);
  int end_y = min((nRowTotal+1)*max_texture_size, (int)cubeHeight);

  GLfloat pModelViewMatrix[16];
  glGetFloatv(GL_MODELVIEW_MATRIX, pModelViewMatrix);

  GLfloat** cubePoints = (GLfloat**)malloc(8*sizeof(GLfloat*));;

  cubePoints[0] = create_vector
    (-widthStep + nColTotal*max_texture_size*voxelWidth,
     heightStep - nRowTotal*max_texture_size*voxelHeight, -depthStep, 1.0f);
  cubePoints[1] = create_vector
    (-widthStep + nColTotal*max_texture_size*voxelWidth,
     heightStep - nRowTotal*max_texture_size*voxelHeight,  depthStep, 1.0f);
  cubePoints[2] = create_vector
    (-widthStep + end_x*voxelWidth,
     heightStep - nRowTotal*max_texture_size*voxelHeight,  depthStep, 1.0f);
  cubePoints[3] = create_vector
    (-widthStep + end_x*voxelWidth,
     heightStep - nRowTotal*max_texture_size*voxelHeight, -depthStep, 1.0f);
  cubePoints[4] = create_vector
    (-widthStep + nColTotal*max_texture_size*voxelWidth,
     heightStep - end_y*voxelHeight,
     -depthStep, 1.0f);
  cubePoints[5] = create_vector
    (-widthStep + nColTotal*max_texture_size*voxelWidth,
     heightStep - end_y*voxelHeight,  depthStep, 1.0f);
  cubePoints[6] = create_vector
    (-widthStep + end_x*voxelWidth,
     heightStep - end_y*voxelHeight,  depthStep, 1.0f);
  cubePoints[7] = create_vector
    (-widthStep + end_x*voxelWidth,
     heightStep - end_y*voxelHeight, -depthStep, 1.0f);

  glLoadIdentity();
  GLfloat** cubePoints_c = (GLfloat**) malloc(8*sizeof(GLfloat*));
  for(int i=0; i < 8; i++)
    cubePoints_c[i] = matrix_vector_product(pModelViewMatrix, cubePoints[i]);

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
                  t_point = (1-lambda_lines[i])*t_max;
                  break;
                case 1: //0-3
                  x_point = cubePoints_c[0][0]*lambda_lines[i] + cubePoints_c[3][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[0][1]*lambda_lines[i] + cubePoints_c[3][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[0][2]*lambda_lines[i] + cubePoints_c[3][2]*(1-lambda_lines[i]);
                  r_point = (1-lambda_lines[i])*r_max;
                  s_point = 0;
                  t_point = 0;
                  break;
                case 2: //0-4
                  x_point = cubePoints_c[0][0]*lambda_lines[i] + cubePoints_c[4][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[0][1]*lambda_lines[i] + cubePoints_c[4][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[0][2]*lambda_lines[i] + cubePoints_c[4][2]*(1-lambda_lines[i]);
                  r_point = 0;
                  s_point = (1-lambda_lines[i])*s_max;
                  t_point = 0;
                  break;
                case 3: //4-7
                  x_point = cubePoints_c[4][0]*lambda_lines[i] + cubePoints_c[7][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[4][1]*lambda_lines[i] + cubePoints_c[7][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[4][2]*lambda_lines[i] + cubePoints_c[7][2]*(1-lambda_lines[i]);
                  r_point = (1-lambda_lines[i])*r_max;
                  s_point = s_max;
                  t_point = 0;
                  break;
                case 4: //4-5
                  x_point = cubePoints_c[4][0]*lambda_lines[i] + cubePoints_c[5][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[4][1]*lambda_lines[i] + cubePoints_c[5][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[4][2]*lambda_lines[i] + cubePoints_c[5][2]*(1-lambda_lines[i]);
                  r_point = 0;
                  s_point = s_max;
                  t_point = (1-lambda_lines[i])*t_max;
                  break;
                case 5: //1-2
                  x_point = cubePoints_c[1][0]*lambda_lines[i] + cubePoints_c[2][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[1][1]*lambda_lines[i] + cubePoints_c[2][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[1][2]*lambda_lines[i] + cubePoints_c[2][2]*(1-lambda_lines[i]);
                  r_point = (1-lambda_lines[i])*r_max;
                  s_point = 0;
                  t_point = t_max;
                  break;
                case 6: //1-5
                  x_point = cubePoints_c[1][0]*lambda_lines[i] + cubePoints_c[5][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[1][1]*lambda_lines[i] + cubePoints_c[5][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[1][2]*lambda_lines[i] + cubePoints_c[5][2]*(1-lambda_lines[i]);
                  r_point = 0;
                  s_point = (1-lambda_lines[i])*s_max;
                  t_point = t_max;
                  break;
                case 7: //5-6
                  x_point = cubePoints_c[5][0]*lambda_lines[i] + cubePoints_c[6][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[5][1]*lambda_lines[i] + cubePoints_c[6][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[5][2]*lambda_lines[i] + cubePoints_c[6][2]*(1-lambda_lines[i]);
                  r_point = (1-lambda_lines[i])*r_max;
                  s_point = s_max;
                  t_point = t_max;
                  break;
                case 8: //3-2
                  x_point = cubePoints_c[3][0]*lambda_lines[i] + cubePoints_c[2][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[3][1]*lambda_lines[i] + cubePoints_c[2][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[3][2]*lambda_lines[i] + cubePoints_c[2][2]*(1-lambda_lines[i]);
                  r_point = r_max;
                  s_point = 0;
                  t_point = (1-lambda_lines[i])*t_max;
                  break;
                case 9: //3-7
                  x_point = cubePoints_c[3][0]*lambda_lines[i] + cubePoints_c[7][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[3][1]*lambda_lines[i] + cubePoints_c[7][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[3][2]*lambda_lines[i] + cubePoints_c[7][2]*(1-lambda_lines[i]);
                  r_point = r_max;
                  s_point = (1-lambda_lines[i])*s_max;
                  t_point = 0;
                  break;
                case 10: //7-6
                  x_point = cubePoints_c[7][0]*lambda_lines[i] + cubePoints_c[6][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[7][1]*lambda_lines[i] + cubePoints_c[6][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[7][2]*lambda_lines[i] + cubePoints_c[6][2]*(1-lambda_lines[i]);
                  r_point = r_max;
                  s_point = s_max;
                  t_point = (1-lambda_lines[i])*t_max;
                  break;
                case 11: //6-2
                  x_point = cubePoints_c[6][0]*lambda_lines[i] + cubePoints_c[2][0]*(1-lambda_lines[i]);
                  y_point = cubePoints_c[6][1]*lambda_lines[i] + cubePoints_c[2][1]*(1-lambda_lines[i]);
                  z_point = cubePoints_c[6][2]*lambda_lines[i] + cubePoints_c[2][2]*(1-lambda_lines[i]);
                  r_point = r_max;
                  s_point = lambda_lines[i]*s_max;
                  t_point = t_max;
                  break;
                }
              intersectionPoints[intersectionPointsIdx][0] = x_point;
              intersectionPoints[intersectionPointsIdx][1] = y_point;
              intersectionPoints[intersectionPointsIdx][2] = z_point;
              intersectionPoints[intersectionPointsIdx][3] = r_point;
              intersectionPoints[intersectionPointsIdx][4] = s_point;
              intersectionPoints[intersectionPointsIdx][5] = t_point;
              intersectionPointsIdx++;

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
          points_angles[i] = atan2(intersectionPoints[i][1]-y_average,
                                   intersectionPoints[i][0]-x_average);
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

      glColor3f(v_r,v_g,v_b);

      if(blendFunction == MIN_MAX)
        {
          glEnable(GL_BLEND);
          if(min_max == 0)
            glBlendEquation(GL_MIN);
          else
            glBlendEquation(GL_MAX);
        }

      glEnable(GL_TEXTURE_3D);
      glBindTexture(GL_TEXTURE_3D, wholeTexture);

      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

      //All the previous was preparation, here with draw the polygon
      glBegin(GL_POLYGON);
      for(int i = 0; i < intersectionPointsIdx; i++)
        {
          glTexCoord3f(intersectionPoints[indexes[i]][3],
                       intersectionPoints[indexes[i]][4],
                       intersectionPoints[indexes[i]][5]);
          glVertex3f(intersectionPoints[indexes[i]][0],
                     intersectionPoints[indexes[i]][1],
                     intersectionPoints[indexes[i]][2]);
        }
      glEnd();

      glDisable(GL_TEXTURE_3D);
      glDisable(GL_BLEND);
    } //depth loop

  if(nMatrices!=0)
    printf("nMatrices = %i\n", nMatrices);

  //Put back the modelView matrix
  glLoadIdentity();
  glMultMatrixf(pModelViewMatrix);
}


void Cube_C::draw_layer_tile_XY(float nLayerToDraw, int color)
{
//   draw_orientation_grid();
  glEnable(GL_TEXTURE_3D);
  glBindTexture(GL_TEXTURE_3D, wholeTexture);
  GLint max_texture_size = 0;
  glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_texture_size);
  max_texture_size = D_MAX_TEXTURE_SIZE;
  int size_texture = max_texture_size;

  if(nLayerToDraw < 0)
    {
      printf("Cube::draw_layer: invalid nLayerToDraw %f\n", nLayerToDraw);
      nLayerToDraw = 0;
    }
  if(nLayerToDraw > cubeDepth -1)
    {
      printf("Cube::draw_layer: invalid nLayerToDraw %f\n", nLayerToDraw);
      nLayerToDraw = cubeDepth-1;
    }


  //In object coordinates, we will draw the cube
  GLfloat widthStep = float(cubeWidth)*voxelWidth/2;
  GLfloat heightStep = float(cubeHeight)*voxelHeight/2;
  GLfloat depthStep = float(cubeDepth)*voxelDepth/2;

  GLfloat increment_height =
    min(float(cubeHeight - nRowToDraw*size_texture), float(size_texture));
  increment_height = increment_height / size_texture;
  GLfloat increment_width =
    min(float(cubeWidth - nColToDraw*size_texture), float(size_texture));
  increment_width = increment_width / size_texture;


//   printf("draw_layer_tile_XY %f %f size texture %i %f %f\n", increment_height, increment_width, size_texture, float(cubeWidth)/size_texture, float(cubeHeight)/size_texture);

  if(color == 0){
//     glColor3f(1.0,1.0,1.0);
    glColor3f(v_r,v_g,v_b);
    glBegin(GL_QUADS);
    glTexCoord3f(0,0,t_max*nLayerToDraw/(cubeDepth-1));
    glVertex3f(-widthStep + nColToDraw*size_texture*voxelWidth,
               heightStep - nRowToDraw*size_texture*voxelHeight,
               -depthStep + nLayerToDraw*voxelDepth);

    glTexCoord3f(r_max,0,t_max*nLayerToDraw/(cubeDepth-1));
    glVertex3f(-widthStep + (nColToDraw+increment_width)*size_texture*voxelWidth,
               heightStep - nRowToDraw*size_texture*voxelHeight,
               -depthStep + nLayerToDraw*voxelDepth);

    glTexCoord3f(r_max,
                 s_max,
                 t_max*nLayerToDraw/(cubeDepth-1));
    glVertex3f(-widthStep + (nColToDraw+increment_width)*size_texture*voxelWidth,
               heightStep - (nRowToDraw+increment_height)*size_texture*voxelHeight,
               -depthStep + nLayerToDraw*voxelDepth);

    glTexCoord3f(0,
                 s_max,
                 t_max*nLayerToDraw/(cubeDepth-1));
    glVertex3f(-widthStep + nColToDraw*size_texture*voxelWidth,
               heightStep - (nRowToDraw+increment_height)*size_texture*voxelHeight,
               -depthStep + nLayerToDraw*voxelDepth);
    glEnd();

  }
  else
    glColor3f(0.0,0.0,1.0);


  glDisable(GL_TEXTURE_3D);
//   glColor3f(0.0,0.0,0.7);
  glBegin(GL_LINE_LOOP);
  glVertex3f(-widthStep + nColToDraw*size_texture*voxelWidth,
             heightStep - nRowToDraw*size_texture*voxelHeight,
             -depthStep + nLayerToDraw*voxelDepth);
  glVertex3f(-widthStep + (nColToDraw+increment_width)*size_texture*voxelWidth,
             heightStep - nRowToDraw*size_texture*voxelHeight,
             -depthStep + nLayerToDraw*voxelDepth);
  glVertex3f(-widthStep + (nColToDraw+increment_width)*size_texture*voxelWidth,
             heightStep - (nRowToDraw+increment_height)*size_texture*voxelHeight,
             -depthStep + nLayerToDraw*voxelDepth);
  glVertex3f(-widthStep + nColToDraw*size_texture*voxelWidth,
             heightStep - (nRowToDraw+increment_height)*size_texture*voxelHeight,
             -depthStep + nLayerToDraw*voxelDepth);
  glEnd();
//   glColor3f(0.0,0.0,1.0);
  glBegin(GL_LINES);

  glEnd();

  //Draws the coordinates

}

void Cube_C::draw_layer_tile_XZ(float nLayerToDraw, int color)
{
//   draw_orientation_grid();
  glEnable(GL_TEXTURE_3D);
  glBindTexture(GL_TEXTURE_3D, wholeTexture);
  GLint max_texture_size = 0;
  glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_texture_size);
  max_texture_size = D_MAX_TEXTURE_SIZE;
  int size_texture = max_texture_size;

  if(nLayerToDraw < 0)
    {
      printf("Cube::draw_layer: invalid nLayerToDraw %f\n", nLayerToDraw);
      nLayerToDraw = 0;
    }
  if( (nLayerToDraw > min((int)cubeHeight-1, (int)cubeHeight - nRowToDraw*size_texture) ) )
    {
      printf("Cube::draw_layer: invalid nLayerToDraw %f\n", nLayerToDraw);
      nLayerToDraw = min((int)cubeHeight-1, (int)cubeHeight - nRowToDraw*size_texture);
    }

  //In object coordinates, we will draw the cube
  GLfloat widthStep = float(cubeWidth)*voxelWidth/2;
  GLfloat heightStep = float(cubeHeight)*voxelHeight/2;
  GLfloat depthStep = float(cubeDepth)*voxelDepth/2;

  GLfloat increment_depth =
    min(float(cubeDepth), float(size_texture));
  increment_depth = increment_depth / size_texture;
  GLfloat increment_width =
    min(float(cubeWidth - nColToDraw*size_texture), float(size_texture));
  increment_width = increment_width / size_texture;

  float y_max = min(size_texture, (int)cubeHeight - nRowToDraw*size_texture);

  if(color == 0){
    glColor3f(1.0,1.0,1.0);
    glColor3f(v_r,v_g,v_b);
    glBegin(GL_QUADS);
    glTexCoord3f(0, s_max*nLayerToDraw/y_max, 0);
    glVertex3f(-widthStep + nColToDraw*max_texture_size*voxelWidth,
               heightStep - nRowToDraw*max_texture_size*voxelHeight
               - nLayerToDraw*voxelHeight,
               -depthStep);
    glTexCoord3f(0, s_max*nLayerToDraw/y_max, t_max);
    glVertex3f(-widthStep + nColToDraw*max_texture_size*voxelWidth,
               heightStep - nRowToDraw*max_texture_size*voxelHeight
               - nLayerToDraw*voxelHeight,
               depthStep);
    glTexCoord3f(r_max, s_max*nLayerToDraw/y_max, t_max);
    glVertex3f(-widthStep + (nColToDraw+increment_width)*max_texture_size*voxelWidth,
               heightStep - nRowToDraw*max_texture_size*voxelHeight
               - nLayerToDraw*voxelHeight,
               depthStep);
    glTexCoord3f(r_max, s_max*nLayerToDraw/y_max , 0);
    glVertex3f(-widthStep + (nColToDraw+increment_width)*max_texture_size*voxelWidth,
               heightStep - nRowToDraw*max_texture_size*voxelHeight
               - nLayerToDraw*voxelHeight,
               -depthStep);
    glEnd();
  }
  else
    glColor3f(1.0,0.0,0.0);



  glDisable(GL_TEXTURE_3D);
//   glColor3f(0.7,0.0,0.0);
  glBegin(GL_LINE_LOOP);
  glVertex3f(-widthStep + nColToDraw*max_texture_size*voxelWidth,
             heightStep - nRowToDraw*max_texture_size*voxelHeight
             - nLayerToDraw*voxelHeight,
             -depthStep);
  glVertex3f(-widthStep + nColToDraw*max_texture_size*voxelWidth,
             heightStep - nRowToDraw*max_texture_size*voxelHeight
             - nLayerToDraw*voxelHeight,
             depthStep);
  glVertex3f(-widthStep + (nColToDraw+increment_width)*max_texture_size*voxelWidth,
             heightStep - nRowToDraw*max_texture_size*voxelHeight
             - nLayerToDraw*voxelHeight,
             depthStep);
  glVertex3f(-widthStep + (nColToDraw+increment_width)*max_texture_size*voxelWidth,
             heightStep - nRowToDraw*max_texture_size*voxelHeight
             - nLayerToDraw*voxelHeight,
             -depthStep);
  glEnd();

}

void Cube_C::draw_layer_tile_YZ(float nLayerToDraw, int color)
{
//   draw_orientation_grid();
  glEnable(GL_TEXTURE_3D);
  glBindTexture(GL_TEXTURE_3D, wholeTexture);
  GLint max_texture_size = 0;
  glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_texture_size);
  max_texture_size = D_MAX_TEXTURE_SIZE;
  int size_texture = max_texture_size;

  if(nLayerToDraw < 0)
    {
      printf("Cube::draw_layer: invalid nLayerToDraw %f\n", nLayerToDraw);
      nLayerToDraw = 0;
    }
  if( (nLayerToDraw > min((int)cubeWidth-1, (int)cubeWidth - nColToDraw*size_texture) ) )
    {
      printf("Cube::draw_layer: invalid nLayerToDraw %f\n", nLayerToDraw);
      nLayerToDraw = min((int)cubeWidth-1, (int)cubeWidth - nColToDraw*size_texture);
    }

  //In object coordinates, we will draw the cube
  GLfloat widthStep = float(cubeWidth)*voxelWidth/2;
  GLfloat heightStep = float(cubeHeight)*voxelHeight/2;
  GLfloat depthStep = float(cubeDepth)*voxelDepth/2;

  GLfloat increment_depth =
    min(float(cubeDepth), float(size_texture));
  increment_depth = increment_depth / size_texture;
  GLfloat increment_height =
    min(float(cubeHeight - nRowToDraw*size_texture), float(size_texture));
  increment_height = increment_height / size_texture;

  float x_max = min(size_texture, (int)cubeWidth - nColToDraw*size_texture);
  GLfloat depth_texture = nLayerToDraw/r_max;

  if(color == 0){
//     glColor3f(1.0,1.0,1.0);
    glColor3f(v_r,v_g,v_b);
    glBegin(GL_QUADS);
    glTexCoord3f(r_max*nLayerToDraw/x_max, 0, 0);
    glVertex3f(-widthStep + nColToDraw*max_texture_size*voxelWidth
               + nLayerToDraw*voxelWidth,
               heightStep - nRowToDraw*max_texture_size*voxelHeight,
               -depthStep);
    glTexCoord3f(r_max*nLayerToDraw/x_max, s_max, 0);
    glVertex3f(-widthStep + nColToDraw *max_texture_size*voxelWidth
               + nLayerToDraw*voxelWidth,
               heightStep - (nRowToDraw+increment_height)*max_texture_size*voxelHeight,
               -depthStep);
    glTexCoord3f(r_max*nLayerToDraw/x_max, s_max, t_max);
    glVertex3f(-widthStep + nColToDraw*max_texture_size*voxelWidth
               + nLayerToDraw*voxelWidth,
               heightStep - (nRowToDraw+increment_height)*max_texture_size*voxelHeight,
               depthStep);
    glTexCoord3f(r_max*nLayerToDraw/x_max, 0, t_max);
    glVertex3f(-widthStep + nColToDraw*max_texture_size*voxelWidth
               + nLayerToDraw*voxelWidth,
               heightStep - nRowToDraw*max_texture_size*voxelHeight,
               depthStep);
    glEnd();
  }
  else
    glColor3f(0.0,1.0,0.0);


  glDisable(GL_TEXTURE_3D);
//   glColor3f(0.0,0.7,0.0);
  glBegin(GL_LINE_LOOP);
  glVertex3f(-widthStep + nColToDraw*max_texture_size*voxelWidth
             + nLayerToDraw*voxelWidth,
             heightStep - nRowToDraw*max_texture_size*voxelHeight,
             -depthStep);
  glVertex3f(-widthStep + nColToDraw *max_texture_size*voxelWidth
             + nLayerToDraw*voxelWidth,
             heightStep - (nRowToDraw+increment_height)*max_texture_size*voxelHeight,
             -depthStep);
  glVertex3f(-widthStep + nColToDraw*max_texture_size*voxelWidth
             + nLayerToDraw*voxelWidth,
             heightStep - (nRowToDraw+increment_height)*max_texture_size*voxelHeight,
             depthStep);
  glVertex3f(-widthStep + nColToDraw*max_texture_size*voxelWidth
             + nLayerToDraw*voxelWidth,
             heightStep - nRowToDraw*max_texture_size*voxelHeight,
             depthStep);
  glEnd();

}



/** From here on it is some data analysis that will not be implemented.*/

void Cube_C::draw
(int x0, int y0, int z0, int x1, int y1, int z1,
 float rotx, float roty, float nPlanes, int min_max, float threshold)
{
  draw(0,0,200,v_draw_projection,0);
}


void Cube_C::print_size()
{
}

void Cube_C::min_max(float* min, float* max){}

Cube_P* Cube_C::threshold(float thres, string outputName,
                  bool putHigherValuesTo, bool putLowerValuesTo,
                  float highValue, float lowValue){}

void Cube_C::print_statistics(string name){}

void Cube_C::histogram(string name){}

void Cube_C::save_as_image_stack(string dirname){}

vector< vector< int > > Cube_C::decimate
(float threshold, int window_xy, int window_z, string filemane,
 bool save_boosting_response){}

vector< vector< int > > Cube_C::decimate_log
(float threshold, int window_xy, int window_z, string filemane,
 bool save_boosting_response){}

/** Produces a vector of the NMS in the layer indicated.*/
vector< vector< int > > Cube_C::decimate_layer
(int nLayer, float threshold, int window_xy, string filename){}

void Cube_C::allocate_alphas(int ni, int nj, int nk){}

void Cube_C::delete_alphas(int ni, int nj, int nk){}


float Cube_C::getValueAsFloat(int x, int y, int z){}
