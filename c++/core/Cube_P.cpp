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

////////////////////////////////////////////////////////////////////////////////
// DRAWING FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

GLfloat* Cube_P::matrix_vector_product(GLfloat* m, GLfloat* v)
{
  GLfloat* b = (GLfloat*)malloc(4*sizeof(GLfloat));
//   GLfloat* b = (GLfloat*)malloc(4*4);
  b[3] =  m[3]*v[0] + m[7]*v[1] + m[11]*v[2] + m[15]*v[3];
  b[0] = (m[0]*v[0] + m[4]*v[1] + m[8 ]*v[2] + m[12]*v[3])/b[3];
  b[1] = (m[1]*v[0] + m[5]*v[1] + m[9 ]*v[2] + m[13]*v[3])/b[3];
  b[2] = (m[2]*v[0] + m[6]*v[1] + m[10]*v[2] + m[14]*v[3])/b[3];
  b[3] = 1;
  if(sizeof(b)/sizeof(GLfloat) < 2){
//     b = (GLfloat*)malloc(4*4);
    GLfloat* b = (GLfloat*)malloc(4*sizeof(GLfloat));
    b[0]=0; b[1] =0; b[2] = 0; b[3]=0;
  }
  return b;
}

GLfloat* Cube_P::create_vector(GLfloat x, GLfloat y, GLfloat z, GLfloat w)
{
  GLfloat* b = (GLfloat*)malloc(4*sizeof(GLfloat));
  b[0]=x; b[1] = y; b[2] = z; b[3] = w;
  return b;
}


void Cube_P::draw
(float nPlanes,
 int min_max, int microm_voxels)
{
  //Parches one bug with the matrices
  int nMatrices = 0;

  GLint max_texture_size = 0;
//   glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_texture_size);
  max_texture_size = D_MAX_TEXTURE_SIZE;

  //  draw_orientation_grid(false, min_max);

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

  if(microm_voxels == 0){
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
  } else {
    widthStep = float(cubeWidth)*voxelWidth/2;
    heightStep = float(cubeHeight)*voxelWidth/2;
    depthStep = float(cubeDepth)*voxelWidth/2;

    cubePoints[0] = create_vector
      (-widthStep + nColTotal*max_texture_size*voxelWidth,
       heightStep - nRowTotal*max_texture_size*voxelWidth, -depthStep, 1.0f);
    cubePoints[1] = create_vector
      (-widthStep + nColTotal*max_texture_size*voxelWidth,
       heightStep - nRowTotal*max_texture_size*voxelWidth,  depthStep, 1.0f);
    cubePoints[2] = create_vector
      (-widthStep + end_x*voxelWidth,
       heightStep - nRowTotal*max_texture_size*voxelWidth,  depthStep, 1.0f);
    cubePoints[3] = create_vector
      (-widthStep + end_x*voxelWidth,
       heightStep - nRowTotal*max_texture_size*voxelWidth, -depthStep, 1.0f);
    cubePoints[4] = create_vector
      (-widthStep + nColTotal*max_texture_size*voxelWidth,
       heightStep - end_y*voxelWidth,
       -depthStep, 1.0f);
    cubePoints[5] = create_vector
      (-widthStep + nColTotal*max_texture_size*voxelWidth,
       heightStep - end_y*voxelWidth,  depthStep, 1.0f);
    cubePoints[6] = create_vector
      (-widthStep + end_x*voxelWidth,
       heightStep - end_y*voxelWidth,  depthStep, 1.0f);
    cubePoints[7] = create_vector
      (-widthStep + end_x*voxelWidth,
       heightStep - end_y*voxelWidth, -depthStep, 1.0f);
  }

  // We will get the coordinates of the vertex of the cube in the modelview coordinates
  glLoadIdentity();
//   GLfloat* cubePoints_c[8];
  GLfloat** cubePoints_c = (GLfloat**) malloc(8*sizeof(GLfloat*));
  // glColor3f(0,0,0);
  for(int i=0; i < 8; i++)
    cubePoints_c[i] = matrix_vector_product(pModelViewMatrix, cubePoints[i]);

  //Draws the points numbers and the coordinates of the textures
  if(0){
    for(int i=0; i < 8; i++)
      {
        glColor3f(0.0,1.0,0.0);
        glPushMatrix();
        nMatrices++;
        glTranslatef(cubePoints_c[i][0], cubePoints_c[i][1], cubePoints_c[i][2]);
        renderString("%i",i);
        glPopMatrix();
        nMatrices--;
      }
    glPushMatrix();
    nMatrices++;
    glTranslatef(cubePoints_c[0][0], cubePoints_c[0][1], cubePoints_c[0][2]);
    // glRotatef(rotx, 1.0,0,0);
    // glRotatef(roty, 0,1.0,0);
    //Draw The Z axis
    glColor3f(0.0, 0.0, 1.0);
    glutSolidCone(1.0, 10, 20, 20);
    glPushMatrix();
    nMatrices++;
    glTranslatef(0,0,10);
    renderString("T");
    glPopMatrix();
    nMatrices--;
    //Draw the x axis
    glColor3f(1.0, 0.0, 0.0);
    glRotatef(90, 0.0, 1.0, 0.0);
    glutSolidCone(1.0, 10, 20, 20);
    glPushMatrix();
    nMatrices++;
    glTranslatef(0,0,10);
    renderString("R");
    glPopMatrix();
    nMatrices--;
    //Draw the y axis
    glColor3f(0.0, 1.0, 0.0);
    glRotatef(90, 1.0, 0.0, 0.0);
    glutSolidCone(1.0, 10, 20, 20);
    glPushMatrix();
    nMatrices++;
    glTranslatef(0,0,10);
    renderString("S");
    glPopMatrix();
    nMatrices--;
    glPopMatrix();
    nMatrices--;
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
    nMatrices++;
    glTranslatef(cubePoints_c[closest_point_idx][0], cubePoints_c[closest_point_idx][1], cubePoints_c[closest_point_idx][2]);
    glColor3f(0.0,1.0,0.0);
    glutWireSphere(5,10,10);
    glPopMatrix();
    nMatrices--;

    glPushMatrix();
    nMatrices++;
    glTranslatef(cubePoints_c[furthest_point_idx][0], cubePoints_c[furthest_point_idx][1], cubePoints_c[furthest_point_idx][2]);
    glColor3f(0.0,0.0,1.0);
    glutWireSphere(5,10,10);
    glPopMatrix();
    nMatrices--;
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
      GLfloat intersectionPoints[6][6];
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

              //Draws spheres in the intersection points
              if(0){
                glPushMatrix();
                nMatrices++;
                glTranslatef(x_point, y_point, z_point);
                glutWireSphere(5,10,10);
                glPopMatrix();
                nMatrices--;
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

//       if(min_max==0)
      glColor3f(v_r,v_g,v_b);
//       if(min_max==1)
//         glColor3f(1.0,0.0,0.0);
//       if(min_max==2)
//         glColor3f(0.0,1.0,0.0);
//       if(min_max==3)
//         glColor3f(0.0,0.0,1.0);
//       if(min_max==4)
//         glColor3f(1.0,1.0,1.0);

      glEnable(GL_BLEND);
      #ifdef WITH_GLEW
      if(blendFunction == MIN_MAX)
        {
          if(min_max == 0)
            glBlendEquation(GL_MIN);
          else
            glBlendEquation(GL_MAX);
        }
      else
        {
          glEnable(GL_ALPHA_TEST);
          //glAlphaFunc(GL_GEQUAL, min_alpha);
          glAlphaFunc(GL_GREATER, min_alpha);
          //glBlendFunc(GL_SRC_ALPHA, GL_ONE);
          //glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }

      //glEnable(GL_STENCIL_TEST);
      //glStencilFunc(GL_GREATER,1.0f,~0);
      //glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);

      glEnable(GL_TEXTURE_3D);
      glBindTexture(GL_TEXTURE_3D, wholeTexture);
#endif
      glEdgeFlag(GL_FALSE);
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
//       glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

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
#ifdef WITH_GLEW
      glDisable(GL_TEXTURE_3D);
      glDisable(GL_BLEND);
      glDisable(GL_ALPHA_TEST); // Test AL
      //glDisable(GL_STENCIL_TEST);
#endif
      //Draws an sphere on all the intersection points
      if(0)
        {
          glColor3f(0.0,1.0,1.0);
          for(int i = 0; i < intersectionPointsIdx; i++)
            {
              glPushMatrix();
              nMatrices++;
              glTranslatef(intersectionPoints[indexes[i]][0],
                           intersectionPoints[indexes[i]][1],
                           intersectionPoints[indexes[i]][2]);
              glutSolidSphere(1,5,5);
              glPopMatrix();
              nMatrices--;
            }
        }

      //Draws the texture coordinates of the intersection points - NOT WORKING
      if(0)
        {
          glColor3f(0.0,0.0,0.0);
          for(int i = 0; i < intersectionPointsIdx; i++)
            {
              glPushMatrix();
              nMatrices++;
              glTranslatef(intersectionPoints[indexes[i]][0],
                           intersectionPoints[indexes[i]][1],
                           intersectionPoints[indexes[i]][2]);
              renderString("(%.2f %.2f %.2f)",
                            intersectionPoints[indexes[i]][3],
                            intersectionPoints[indexes[i]][4],
                            intersectionPoints[indexes[i]][5]);
              glPopMatrix();
              nMatrices--;
            }
          glColor3f(1.0,1.0,1.0);
         }
    } //depth loop

  if(nMatrices!=0)
    printf("nMatrices = %i\n", nMatrices);

  //Put back the modelView matrix
  glLoadIdentity();
  glMultMatrixf(pModelViewMatrix);
}
//End of draw

void Cube_P::draw(){
  // printf("I am beauty\n");
  draw(200,v_draw_projection,0);
//   draw_layers_parallel();
}


void Cube_P::draw_layer_tile_XY(float nLayerToDraw, int texturize)
{

  GLint max_texture_size = 0;
//   draw_orientation_grid();
#ifdef WITH_GLEW
//   glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_texture_size);
  glEnable(GL_TEXTURE_3D);
  glBindTexture(GL_TEXTURE_3D, wholeTexture);
#endif
  max_texture_size = D_MAX_TEXTURE_SIZE;
  int size_texture = max_texture_size;

  if(nLayerToDraw < 0){
    printf("Cube::draw_layer: invalid nLayerToDraw %f\n", nLayerToDraw);
    nLayerToDraw = 0;
  }
  if(nLayerToDraw > cubeDepth -1){
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

  if(texturize == 0){
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

#ifdef WITH_GLEW
  glDisable(GL_TEXTURE_3D);
#endif
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
//  glBegin(GL_LINES);

  glEnd();

  //Draws the coordinates
}

void Cube_P::draw_layer_tile_XZ(float nLayerToDraw, int color)
{
//   draw_orientation_grid();
#ifdef WITH_GLEW
  glEnable(GL_TEXTURE_3D);
  glBindTexture(GL_TEXTURE_3D, wholeTexture);
#endif
  GLint max_texture_size = 0;
//   glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_texture_size);
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


#ifdef WITH_GLEW
  glDisable(GL_TEXTURE_3D);
#endif
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

void Cube_P::draw_layer_tile_YZ(float nLayerToDraw,int color)
{
//   draw_orientation_grid();
#ifdef WITH_GLEW
  glEnable(GL_TEXTURE_3D);
  glBindTexture(GL_TEXTURE_3D, wholeTexture);
#endif
  GLint max_texture_size = 0;
//   glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_texture_size);
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

#ifdef WITH_GLEW
  glDisable(GL_TEXTURE_3D);
//   glColor3f(0.0,0.7,0.0);
#endif
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





//////////////////////////////////////////////////////////////////////
// STATISTICAL FUNCTIONS
//////////////////////////////////////////////////////////////////////
void Cube_P::min_max(float& min, float& max)
{
  min = 1e24;
  max = -1e24;

  printf("%s calculating min and max [", filenameParameters.c_str());
  for(int z = 0; z < cubeDepth; z++){
    for(int y = 0; y < cubeHeight; y++){
      for(int x = 0; x < cubeWidth; x++){
        if( get(x,y,z) > max)
          max = get(x,y,z);
        if( (get(x,y,z) < min) )
          min = get(x,y,z);
      }
    }
    printf("#");fflush(stdout);
  }
  printf("]\n");
}


void Cube_P::histogram(string filename)
{

  float max, min;
  min_max(min, max);

  float range = max - min;

  vector< int > boxes(100);
  for(int i = 0; i < 100; i++)
    boxes[i] = 0;

  for(int z = 0; z < cubeDepth; z++){
    for(int y = 0; y < cubeHeight; y++){
      for(int x = 0; x < cubeWidth; x++){
        boxes[(int)(floor(100*(this->get(x,y,z)-min)/range))] += 1;
      }
    }
    printf("#"); fflush(stdout);
  }
  printf("]\n");

  int total =0;
  for(int i = 0; i < boxes.size(); i++)
    total += boxes[i];

  if(filename == ""){
    int totalToNow = 0;
    for(int i =0; i < boxes.size(); i++){
      totalToNow = totalToNow + boxes[i];
      printf("[%f %f ] - %i ttn-%i tte=%i\n",
             min + i*range/100, min + (i+1)*range/100,
             boxes[i], totalToNow, total-totalToNow);}
    printf("\n");
  }
  else{
    std::ofstream out(filename.c_str());
    int totalToNow = 0;
    for(int i =0; i < boxes.size(); i++){
      totalToNow = totalToNow + boxes[i];
      printf("[%f %f] - %i\n", min + i*range/100, min + (i+1)*range/100,  boxes[i]);
      out << min + i*range/100 << " " << min + (i+1)*range/100
          << " " << boxes[i] << " " << totalToNow << std::endl;
    }
    out.close();
  }
}

void Cube_P::print_statistics(string filename)
{
//   printf("%f %f %f\n", voxels[0][0][0], voxels[112][511][511], voxels[30][400][40]);
  //Will find the mean and the variance and print it. Also the max and the min
  float max = 1e-12;
  float min = 1e12;
  float mean = 0;
  for(int z = 0; z < cubeDepth; z++)
    for(int y = 0; y < cubeHeight; y++)
      for(int x = 0; x < cubeWidth; x++)
        {
          mean += get(x,y,z);
          if(get(x,y,z) > max)
            max = get(x,y,z);
          if(get(x,y,z) < min)
            min = get(x,y,z);
        }

  mean = mean / (cubeDepth*cubeHeight*cubeWidth);
  printf("Cube mean value is %06.015f, max = %06.015f, min = %06.015f\n", mean, max, min);
}

