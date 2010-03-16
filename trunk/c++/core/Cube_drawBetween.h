template <class T, class U>
void Cube<T,U>::draw(int x0, int y0, int z0, int x1, int y1, int z1, float rotx, float roty, float nPlanes, int min_max, float threshold)
{

//   draw_orientation_grid(false);

  if( (x1<x0) || (y1<y0) || (z1<z0))
    return;

  int max_texture_size = 1024;

  //Checks if I need to reload the 3D texture
  if(x1 > cubeWidth) x1 = cubeWidth;
  if(y1 > cubeHeight) y1 = cubeHeight;
  if(z1 > cubeDepth) z1 = cubeDepth;
  int width = min(max_texture_size, abs(x1-x0));
  int height = min(max_texture_size, abs(y1-y0));
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
#ifdef WITH_GLEW
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
#endif
      printf("P0=[%i %i %i] P1=[%i %i %i] D=[%i %i %i]\n",
             x0, y0, z0, x1, y1, z1, width, height, depth);
      if(sizeof(T)==1){
        uchar* texels =(uchar*)( malloc(width*height*depth*sizeof(uchar)));
        float x_step = max(1.0,(double)fabs((float)x1-(float)x0)/max_texture_size);
        float y_step = max(1.0,(double)fabs((float)y1-(float)y0)/max_texture_size);

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
                      texels[z_t*width*height + y_t*width + x_t] = at((int)x,(int)y,(int)z);
                    else{
                      if(at((int)x,(int)y,(int)z) < threshold)
                        texels[z_t*width*height + y_t*width + x_t] = at((int)x,(int)y,(int)z);
                      else
                        texels[z_t*width*height + y_t*width + x_t] = 255;
                    }
                  }
              }
            printf("#");
            fflush(stdout);
          }
        printf("]\n");
#ifdef WITH_GLEW
        glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE8, width, height, depth, 0, GL_LUMINANCE,
                     GL_UNSIGNED_BYTE, texels);
#endif
        free(texels);
      }
      // if(sizeof(T)==4){
        // float* texels =(float*)( malloc(width*height*depth*sizeof(float)));
        // float x_step = max(1.0,(double)fabs((float)x1-(float)x0)/max_texture_size);
        // float y_step = max(1.0,(double)fabs((float)y1-(float)y0)/max_texture_size);
        // float x = 0;
        // float y = 0;
        // float z = 0;

        // //Calculates the max and the min of the texture to be loaded
        // float text_min = 1e6;
        // float text_max = -1e6;
        // for(int z_t = 0; z_t < depth; z_t++){
            // for(int y_t = 0; y_t < height; y_t++){
                // for(int x_t = 0; x_t < width; x_t++){
                    // x = x0 + x_step*x_t;
                    // y = y0 + y_step*y_t;
                    // z = z0 + z_t;
                    // if(at((int)x,(int)y,(int)z) > text_max)
                      // text_max = at((int)x,(int)y,(int)z);
                    // if(at((int)x,(int)y,(int)z) < text_min)
                      // text_min = at((int)x,(int)y,(int)z);
                  // }
              // }
          // }

        // for(int z_t = 0; z_t < depth; z_t++){
            // for(int y_t = 0; y_t < height; y_t++){
                // for(int x_t = 0; x_t < width; x_t++){
                    // x = x0 + x_step*x_t;
                    // y = y0 + y_step*y_t;
                    // z = z0 + z_t;
                    // if(threshold == -1e6)
                      // texels[z_t*width*height + y_t*width + x_t] = (at((int)x,(int)y,(int)z)-text_min)/(text_max-text_min);
                    // else{
                      // if(at((int)x,(int)y,(int)z) > threshold)
                        // texels[z_t*width*height + y_t*width + x_t] = (at((int)x,(int)y,(int)z)-threshold)/(text_max-threshold);
                      // else
                        // texels[z_t*width*height + y_t*width + x_t] = 0.0;
                    // }
                  // }
              // }
            // printf("#");
            // fflush(stdout);
          // }
        // printf("]\n");
        // glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE, width, height, depth, 0, GL_LUMINANCE,
                     // GL_FLOAT, texels);
        // free(texels);
      // }


    }


  //In object coordinates, we will draw the cube
  GLfloat widthStep = float(cubeWidth)*voxelWidth/2;
  GLfloat heightStep = float(cubeHeight)*voxelHeight/2;
  GLfloat depthStep = float(cubeDepth)*voxelDepth/2;

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

  int nColTotal = nColToDraw;
  int nRowTotal = nRowToDraw;

  int end_x = min((nColTotal+1)*max_texture_size, (int)cubeWidth);
  int end_y = min((nRowTotal+1)*max_texture_size, (int)cubeHeight);

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
#ifdef WITH_GLEW
      if(min_max == 0)
        glBlendEquation(GL_MIN);
//         glBlendEquation(GL_FUNC_REVERSE_SUBTRACT);
//         glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
      else
        glBlendEquation(GL_MAX);

      glEnable(GL_TEXTURE_3D);
      glBindTexture(GL_TEXTURE_3D, wholeTexture);
#endif
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
#ifdef WITH_GLEW
      glDisable(GL_TEXTURE_3D);
      glDisable(GL_BLEND);
#endif
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
