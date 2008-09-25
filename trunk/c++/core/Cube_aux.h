template <class T, class U>
GLfloat* Cube<T,U>::invert_matrix(GLfloat* a)
{
  GLfloat* b = (GLfloat*) malloc(16*sizeof(GLfloat));

  double t14 = a[0]*a[5];
  double t15 = a[10]*a[15];
  double t17 = a[14]*a[11];
  double t19 = a[0]*a[6];
  double t20 = a[9]*a[15];
  double t22 = a[13]*a[11];
  double t24 = a[0]*a[7];
  double t25 = a[9]*a[14];
  double t27 = a[13]*a[10];
  double t29 = a[1]*a[4];
  double t32 = a[1]*a[6];
  double t33 = a[8]*a[15];
  double t35 = a[12]*a[11];
  double t37 = a[1]*a[7];
  double t38 = a[8]*a[14];
  double t40 = a[12]*a[10];
  double t42 = t14*t15-t14*t17-t19*t20+t19*t22+t24*t25-t24*t27-t29*t15+t29*t17+t32
    *t33-t32*t35-t37*t38+t37*t40;
  double t43 = a[2]*a[4];
  double t46 = a[2]*a[5];
  double t49 = a[2]*a[7];
  double t50 = a[8]*a[13];
  double t52 = a[12]*a[9];
  double t54 = a[3]*a[4];
  double t57 = a[3]*a[5];
  double t60 = a[3]*a[6];
  double t63 = t43*t20-t43*t22-t46*t33+t46*t35+t49*t50-t49*t52-t54*t25+t54*t27+t57
    *t38-t57*t40-t60*t50+t60*t52;
  double t65 = 1/(t42+t63);
  double t71 = a[8]*a[6];
  double t73 = a[12]*a[6];
  double t75 = a[8]*a[7];
  double t77 = a[12]*a[7];
  double t81 = a[4]*a[9];
  double t83 = a[4]*a[13];
  double t85 = a[8]*a[5];
  double t87 = a[12]*a[5];
  double t101 = a[1]*a[10];
  double t103 = a[1]*a[14];
  double t105 = a[2]*a[9];
  double t107 = a[2]*a[13];
  double t109 = a[3]*a[9];
  double t111 = a[3]*a[13];
  double t115 = a[0]*a[10];
  double t117 = a[0]*a[14];
  double t119 = a[2]*a[8];
  double t121 = a[2]*a[12];
  double t123 = a[3]*a[8];
  double t125 = a[3]*a[12];
  double t129 = a[0]*a[9];
  double t131 = a[0]*a[13];
  double t133 = a[1]*a[8];
  double t135 = a[1]*a[12];

  //b will be returned in the opengl ordening
  b[0] = (a[5]*a[10]*a[15]-a[5]*a[14]*a[11]-a[6]*a[9]*a[15]+a[6]*a[13]*a[11]+a[7]*a[9]*a[14]-a[7]*a[13]*
          a[10])*t65;

  b[4] = -(a[4]*a[10]*a[15]-a[4]*a[14]*a[11]-t71*a[15]+t73*a[11]+t75*a[14]-t77*a[10])*t65;
  b[8] = (t81*a[15]-t83*a[11]-t85*a[15]+t87*a[11]+t75*a[13]-t77*a[9])*t65;
  b[12] = -(t81*a[14]-t83*a[10]-t85*a[14]+t87*a[10]+t71*a[13]-t73*a[9])*t65;
  b[1] = -(t101*a[15]-t103*a[11]-t105*a[15]+t107*a[11]+t109*a[14]-t111*a[10])*t65;
  b[5] = (t115*a[15]-t117*a[11]-t119*a[15]+t121*a[11]+t123*a[14]-t125*a[10])*t65;
  b[9] = -(t129*a[15]-t131*a[11]-t133*a[15]+t135*a[11]+t123*a[13]-t125*a[9])*t65;
  b[13] = (t129*a[14]-t131*a[10]-t133*a[14]+t135*a[10]+t119*a[13]-t121*a[9])*t65;
  b[2] = (t32*a[15]-t103*a[7]-t46*a[15]+t107*a[7]+t57*a[14]-t111*a[6])*t65;
  b[6] = -(t19*a[15]-t117*a[7]-t43*a[15]+t121*a[7]+t54*a[14]-t125*a[6])*t65;
  b[10] = (t14*a[15]-t131*a[7]-t29*a[15]+t135*a[7]+t54*a[13]-t125*a[5])*t65;
  b[14] = -(t14*a[14]-t131*a[6]-t29*a[14]+t135*a[6]+t43*a[13]-t121*a[5])*t65;
  b[3] = -(t32*a[11]-t101*a[7]-t46*a[11]+t105*a[7]+t57*a[10]-t109*a[6])*t65;
  b[7] = (t19*a[11]-t115*a[7]-t43*a[11]+t119*a[7]+t54*a[10]-t123*a[6])*t65;
  b[11] = -(t14*a[11]-t129*a[7]-t29*a[11]+t133*a[7]+t54*a[9]-t123*a[5])*t65;
  b[15] = (t14*a[10]-t129*a[6]-t29*a[10]+t133*a[6]+t43*a[9]-t119*a[5])*t65;

  return b;
}

template <class T, class U>
GLfloat* Cube<T,U>::matrix_vector_product(GLfloat* m, GLfloat* v)
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

template <class T, class U>
GLfloat* Cube<T,U>::get_matrix_angles(GLfloat* m)
{
  GLfloat* m2 = (GLfloat*)malloc(16*sizeof(GLfloat));
  memcpy(m2, m, 16*sizeof(GLfloat));
  m2[12]=0;
  m2[13]=0;
  m2[14]=0;

//   for(int i = 0; i < 16; i++)
//     printf("%f ", m2[i]);
//   printf("\n");

  GLfloat* b = (GLfloat*)malloc(4*sizeof(GLfloat));
  GLfloat* x_r = matrix_vector_product(m2, create_vector(1,0,0,1));
  float angle_x = atan2(-x_r[2],x_r[0]);
  float angle_y = atan2(-x_r[1],x_r[2]);
  float angle_z = atan2(x_r[1],x_r[0]);

  b[1] =  angle_x*180.0/3.14159;
  b[2] =  angle_y*180.0/3.14159;
  b[0] =  angle_z*180.0/3.14159;
  b[3] =  0;

  free(m2);

  return b;
}

template <class T, class U>
GLfloat* Cube<T,U>::create_vector(GLfloat x, GLfloat y, GLfloat z, GLfloat w)
{
  GLfloat* b = (GLfloat*)malloc(4*sizeof(GLfloat));
  b[0]=x; b[1] = y; b[2] = z; b[3] = w;
  return b;
}

template <class T, class U>
int Cube<T,U>::find_value_in_ordered_vector(vector< T >& vector_ord, T value)
{
  int begin = 0;
  int end = vector_ord.size() -1;
  int middle = floor((begin + end)/2);

  while( (end-begin)>1 )
    {
      middle = floor((begin + end)/2);
//       printf("%i %i %i %f %f\n", begin, end, middle, vector_ord[middle], value);
      if( vector_ord[middle] < value)
        end = middle;
      if( vector_ord[middle] > value)
        begin = middle;
      if( fabsf(vector_ord[middle] - value) < 1e-8 )
        break;
    }

  if( fabsf(vector_ord[begin] - value) < 1e-8)
    {
//       printf("Error: %f ", vector_ord[begin] - value);
      return begin;
    }
  if( fabsf(vector_ord[end] - value) < 1e-8)
    {
//       printf("Error: %f ", vector_ord[end] - value);
      return end;
    }
//   printf("Error: %f ", vector_ord[middle] - value);
  return middle;
}

template <class T, class U>
vector< T > Cube<T,U>::sort_values()
{
  vector<T> toSort;

  //We do the +- 3 to avoid layers without boosting result
  for(int z = 0 + 3; z < cubeDepth-3; z++)
    for(int y = 0; y < cubeHeight; y++)
      for(int x = 0; x < cubeWidth; x++)
        toSort.push_back(this->at2(x,y,z));

  sort(toSort.begin(), toSort.end());
  reverse(toSort.begin(), toSort.end());

  return toSort;
}
