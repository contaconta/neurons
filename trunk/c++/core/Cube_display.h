
#include "Cube_drawBetween.h"

//#define USE_ALPHA

/** We will have some tuning of the parameters for the rendering here
    as defines. At some point they should be put as variables, so that
    they can be accessed from the program.
 */

#define D_MAX_TEXTURE_SIZE      1024
// #define D_TEXTURE_INTERPOLATION GL_LINEAR
#define D_TEXTURE_INTERPOLATION GL_NEAREST


template <class T, class U>
void Cube<T,U>::load_whole_texture()
{

  nColToDraw = -1;
  nRowToDraw = -1;

  GLint max_texture_size = 0;
  glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_texture_size);
  max_texture_size = D_MAX_TEXTURE_SIZE;
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
              //if(tf!=0)
              //  texels[z*max_texture_size*max_texture_size + y*max_texture_size + x ] = tf[(int)voxel];
              //else
              texels[z*max_texture_size*max_texture_size + y*max_texture_size + x ] = voxel;
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
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, D_TEXTURE_INTERPOLATION);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, D_TEXTURE_INTERPOLATION);
//   glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, D_TEXTURE_INTERPOLATION);
//   glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, D_TEXTURE_INTERPOLATION);

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
  void Cube<T,U>::load_texture_brick(int row, int col, float scale)
{
  nColToDraw = col;
  nRowToDraw = row;
  GLint max_texture_size = 0;
  glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_texture_size);
//   max_texture_size = min(max_texture_size,max((int)cubeWidth,(int)cubeHeight));;
  max_texture_size = D_MAX_TEXTURE_SIZE;
  printf("Max texture size %i\n", max_texture_size);

  #if debug
  printf("Cube::load_texture_brick() max_texture_size = %i\n", max_texture_size);
  printf("Cube::load_texture_brick() creating the texture buffer\n");
  #endif

  int limit_x = min((int)cubeWidth,
                    min((int)max_texture_size,
                        (int)cubeWidth - (nColToDraw*max_texture_size)));
  int limit_y = min((int)cubeHeight,
                    min((int)max_texture_size,
                        (int)cubeHeight - (nRowToDraw*max_texture_size)));
  int limit_z =  (int)min(max_texture_size, (int)cubeDepth);

  int texture_size_x = (int) pow(2, ceil(log(limit_x)/log(2)) );
  int texture_size_y = (int) pow(2, ceil(log(limit_y)/log(2)) );
  int texture_size_z = (int) pow(2, ceil(log(limit_z)/log(2)) );

  //Limit od the textures. They are object variables
  r_max = (double)limit_x/texture_size_x;
  s_max = (double)limit_y/texture_size_y;
  t_max = (double)limit_z/texture_size_z;

  printf("Load_texture_brick: texture size %i, limit_x = %i, limit_y = %i limit_z = %i\n               texture_size: x=%i y=%i z=%i\n",
         max_texture_size, limit_x, limit_y, limit_z,
         texture_size_x, texture_size_y,texture_size_z);


  if( (limit_x<0) || (limit_y<0))
    {
      printf("Cube<T,U>::load_texture_brick requested col %i row %i out of range, loading 0,0\n", nColToDraw, nRowToDraw);
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

  //   Creates the array with the texture. Coded to avoid float multiplications

  int col_offset = (col)*max_texture_size;
  int row_offset = (row)*max_texture_size;

  int size_texture = max_texture_size;

  /*
  if(sizeof(T) == 1)
    {
      GLubyte* texels =(GLubyte*)(malloc(texture_size_x*texture_size_y*texture_size_z*sizeof(GLubyte)));
      for(int t = 0; t < texture_size_x*texture_size_y*texture_size_z; t++)
            texels[t] = 0;

      for(int z = 0; z < limit_z; z++)
        {
          int depth_z = z*texture_size_x*texture_size_y;
          for(int y = 0; y < limit_y; y++)
            {
              int depth_y = y*texture_size_x;
              for(int x = 0; x < limit_x; x++)
                {
                  texels[depth_z + depth_y + x] =
                    (GLubyte)at(col_offset+x,row_offset+y,z)*scale;
                }
            }
          printf("#");
          fflush(stdout);
//           printf("%u\n", texels[depth_z + 200*limit_x + 120]);
        }
      printf("]\n");
      glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE8,
                   texture_size_x, texture_size_y, texture_size_z, 0, GL_LUMINANCE,
                   GL_UNSIGNED_BYTE, texels);
      free(texels);
    }
*/

  if(sizeof(T) == 1)
    {
      GLubyte voxel;
      
      //GLubyte* texel =(GLubyte*)(malloc(texture_size_x*texture_size_y*texture_size_z*sizeof(GLubyte)));
      //for(int t = 0; t < texture_size_x*texture_size_y*texture_size_z; t++)
      //      texels[t] = 0;

      int texture_size;
#ifdef USE_ALPHA
      texture_size = texture_size_x*texture_size_y*texture_size_z*2;
#else
      texture_size = texture_size_x*texture_size_y*texture_size_z;
#endif
      GLubyte* texels =(GLubyte*)(malloc(texture_size*sizeof(GLubyte)));
      for(int t = 0; t < texture_size; t++)
        texels[t] = 0;

      int depth_y;
      int depth_z;
      for(int z = 0; z < limit_z; z++)
        {
#ifdef USE_ALPHA
          depth_z = z*texture_size_x*texture_size_y*2;
#else
          depth_z = z*texture_size_x*texture_size_y;
#endif
          for(int y = 0; y < limit_y; y++)
            {
#ifdef USE_ALPHA
              depth_y = y*texture_size_x*2;
#else
              depth_y = y*texture_size_x;
#endif
              for(int x = 0; x < limit_x; x++)
                {
                  voxel = (GLubyte)at(col_offset+x,row_offset+y,z)*scale;

#ifdef USE_ALPHA
                  texels[depth_z + depth_y + x*2] = voxel;

                  if(alphas)
                    {
                      if(alphas[x][y][z]!=0)
                        printf("Alpha=%d\n", alphas[x][y][z]);
                      texels[depth_z + depth_y + x*2+1] = alphas[x][y][z];
                    }
                  else
                    {
                      //printf("Null alpha\n");
                      texels[depth_z + depth_y + x*2+1] = 0;
                    }
#else
                  //if(tf!=0)
                  //  texels[depth_z + depth_y + x] = tf[voxel];
                  //else
                    texels[depth_z + depth_y + x] = voxel;
#endif
                }
            }
          printf("#");
          fflush(stdout);
//           printf("%u\n", texels[depth_z + 200*limit_x + 120]);
        }
      printf("]\n");
      //printf("Cube::load_whole_texture() 8 bits\n");
#ifdef USE_ALPHA
      glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE8_ALPHA8,
                   texture_size_x,texture_size_y, texture_size_z, 0, GL_LUMINANCE_ALPHA,
                   GL_UNSIGNED_BYTE, texels);
#else
      glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE8,
                   texture_size_x, texture_size_y, texture_size_z, 0, GL_LUMINANCE,
                   GL_UNSIGNED_BYTE, texels);
#endif
      free(texels);
    }


  if(sizeof(T) == 4)
    {
      float voxel;
      int texture_size;
#ifdef USE_ALPHA
      texture_size = texture_size_x*texture_size_y*texture_size_z*2;
#else
      texture_size = texture_size_x*texture_size_y*texture_size_z;
#endif
      float* texels =(float*)(calloc(texture_size, sizeof(float)));
      /*
      for(int t = 0; t < texture_size; t++)
      float voxel;
      int texture_size;
#ifdef USE_ALPHA
      texture_size = texture_size_x*texture_size_y*texture_size_z*2;
#else
      texture_size = texture_size_x*texture_size_y*texture_size_z;
#endif
      float* texels =(float*)(calloc(texture_size,sizeof(float)));
      /*
      for(int t = 0; t < texture_size; t++)
            texels[t] = 0;
      */

      printf("Cube::load_texture_brick() creating texture from floats\n");
      float max_texture = -10e6;
      float min_texture = 10e6;
      for(int z = 0; z < limit_z; z++)
        {
          for(int y = 0; y < limit_y; y++)
            {
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
      int depth_y;
      int depth_z;
      for(int z = 0; z < limit_z; z++)
        {
#ifdef USE_ALPHA
          depth_z = z*texture_size_x*texture_size_y*2;
#else
          depth_z = z*texture_size_x*texture_size_y;
#endif
          for(int y = 0; y < limit_y; y++)
            {
#ifdef USE_ALPHA
              depth_y = y*texture_size_x*2;
#else
              depth_y = y*texture_size_x;
#endif
              for(int x = 0; x < limit_x; x++)
                {
//                   if((y<20) || (z>86) || (z<10) ){
//                     texels[depth_z + depth_y + x] = 0;
//                   }
//                   else{
                  voxel = scale*(this->at(col_offset+x,row_offset+y,z) - min_texture)
                    / (max_texture - min_texture);

#ifdef USE_ALPHA
                  texels[depth_z + depth_y + x*2] = voxel;
                  if(alphas)
                    {
                      if(alphas[x][y][z]!=0)
                        printf("Alphaf=%d\n", alphas[x][y][z]);
                      texels[depth_z + depth_y + x*2+1] = alphas[x][y][z];
                    }
                  else
                    {
                      //printf("Null alpha\n");
                      texels[depth_z + depth_y + x*2+1] = 0;
                    }
#else
                  if(tf != 0)
                    texels[depth_z + depth_y + x] = tf[(int)voxel];
                  else
                      texels[depth_z + depth_y + x] = voxel;                      
#endif                  
//                   }
                }
            }
          printf("#");
          fflush(stdout);
        }
      printf("]\n");
#ifdef USE_ALPHA
      //printf("Cube::load_whole_texture() USE_ALPHA float\n");
      glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE_ALPHA,
                   texture_size_x,texture_size_y, texture_size_z, 0, GL_LUMINANCE_ALPHA,
                   GL_FLOAT, texels);
#else
      //printf("Cube::load_whole_texture() NO_ALPHA float\n");
      glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE,
                   texture_size_x,texture_size_y, texture_size_z, 0, GL_LUMINANCE,
                   GL_FLOAT, texels);
#endif
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
  max_texture_size = D_MAX_TEXTURE_SIZE;
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
  max_texture_size = D_MAX_TEXTURE_SIZE;
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
  max_texture_size = D_MAX_TEXTURE_SIZE;
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

template< class T, class U>
void Cube<T,U>::draw(){
  draw(0,0,200,v_draw_projection,0);
}

template <class T, class U>
void Cube<T,U>::draw
(float rotx, float roty, float nPlanes,
 int min_max, int microm_voxels)
{
  //Parches one bug with the matrices
  int nMatrices = 0;

  GLint max_texture_size = 0;
  glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_texture_size);
  max_texture_size = D_MAX_TEXTURE_SIZE;

  draw_orientation_grid(false, min_max);

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
        render_string("%i",i);
        glPopMatrix();
        nMatrices--;
      }
    glPushMatrix();
    nMatrices++;
    glTranslatef(cubePoints_c[0][0], cubePoints_c[0][1], cubePoints_c[0][2]);
    glRotatef(rotx, 1.0,0,0);
    glRotatef(roty, 0,1.0,0);
    //Draw The Z axis
    glColor3f(0.0, 0.0, 1.0);
    glutSolidCone(1.0, 10, 20, 20);
    glPushMatrix();
    nMatrices++;
    glTranslatef(0,0,10);
    render_string("T");
    glPopMatrix();
    nMatrices--;
    //Draw the x axis
    glColor3f(1.0, 0.0, 0.0);
    glRotatef(90, 0.0, 1.0, 0.0);
    glutSolidCone(1.0, 10, 20, 20);
    glPushMatrix();
    nMatrices++;
    glTranslatef(0,0,10);
    render_string("R");
    glPopMatrix();
    nMatrices--;
    //Draw the y axis
    glColor3f(0.0, 1.0, 0.0);
    glRotatef(90, 1.0, 0.0, 0.0);
    glutSolidCone(1.0, 10, 20, 20);
    glPushMatrix();
    nMatrices++;
    glTranslatef(0,0,10);
    render_string("S");
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

      if(blendFunction == MIN_MAX)
        {
          glEnable(GL_BLEND);
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
      glDisable(GL_ALPHA_TEST); // Test AL
      //glDisable(GL_STENCIL_TEST);

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
              render_string("(%.2f %.2f %.2f)",
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
  max_texture_size = D_MAX_TEXTURE_SIZE;
  draw_orientation_grid(false, min_max);

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
void Cube<T,U>::draw_layer_tile_XY(float nLayerToDraw, int color)
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

template <class T, class U>
void Cube<T,U>::draw_layer_tile_XZ(float nLayerToDraw, int color)
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

template <class T, class U>
void Cube<T,U>::draw_layer_tile_YZ(float nLayerToDraw,int color)
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


template <class T, class U>
void Cube<T,U>::draw_orientation_grid(bool include_split, bool min_max)
{
  GLint max_texture_size = 0;
  glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_texture_size);
  max_texture_size = D_MAX_TEXTURE_SIZE;
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
  if(min_max)
    glColor3f(1.0,1.0,1.0);
  else
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
      for(float i = -widthStep; i <= widthStep; i+= voxelWidth*D_MAX_TEXTURE_SIZE)
        {
          glBegin(GL_LINES);
          glVertex3f(i, heightStep, -depthStep);
          glVertex3f(i, -heightStep, -depthStep);
          glEnd();
        }
      for(float i = heightStep; i >= -heightStep; i-= voxelHeight*D_MAX_TEXTURE_SIZE)
        {
          glBegin(GL_LINES);
          glVertex3f(-widthStep, i, -depthStep);
          glVertex3f(widthStep, i, -depthStep);
          glEnd();
        }
    }
  glColor3f(1.0,1.0,1.0);

}



template <class T, class U>
void Cube<T,U>::render_string(const char* format, ...)
{
 va_list args;
 char    buffer[1024];
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
void Cube<T,U>::allocate_alphas(int ni, int nj, int nk)
{
  if(alphas==0)
    {
      printf("Allocating alphas\n");
      alphas = new GLubyte** [ni];
      for(int i = 0;i<ni;i++)
        {
          alphas[i] = new GLubyte*[nj];
          for(int j = 0;j<nj;j++)
            {
              alphas[i][j] = new GLubyte[nk];

              /*
                for(int k = 0;k<nk;k++)
                {
                alphas[i][j][k] = new GLubyte;
                }
              */
            }
        }
    }
}

template <class T, class U>
void Cube<T,U>::delete_alphas(int ni, int nj, int nk)
{
  if(alphas!=0)
    {
      for(int i = 0;i<ni;i++)
	{
	  for(int j = 0;j<nj;j++)
	    {
	      delete alphas[i][j];
	    }
	  delete alphas[i];
	}
      delete[] alphas;
    }
  alphas=0;
}
