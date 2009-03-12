template <class T, class U>
void Cube<T,U>::subsampleMean(string dirname)
{

  // // Subsampling /6 /6
  char buff_nfo[1024];
  char buff_vl [1024];

  sprintf(buff_nfo, "%s/volume_subsampled.nfo",dirname.c_str());
  std::ofstream out(buff_nfo);
  out << "cubeWidth " << this->cubeWidth/6 << std::endl;
  out << "cubeHeight " << this->cubeHeight/6 << std::endl;
  out << "cubeDepth " << this->cubeDepth << std::endl;
  out << "parentCubeWidth " << this->cubeWidth/6 << std::endl;
  out << "parentCubeHeight " << this->cubeHeight/6 << std::endl;
  out << "parentCubeDepth " << this->cubeDepth << std::endl;
  out << "voxelWidth " << this->voxelWidth*6 << std::endl;
  out << "voxelHeight " << this->voxelHeight*6 << std::endl;
  out << "voxelDepth " << this->voxelDepth << std::endl;
  out << "rowOffset 0\n";
  out << "colOffset 0\n";
  out << "x_offset 0\n";
  out << "y_offset 0\n";
  out << "z_offset 0\n";
  out.close();

  sprintf(buff_vl, "%s/volume_subsampled.vl", dirname.c_str());
  printf("Creating volume file in %s\n", buff_vl);
  FILE* fp = fopen(buff_vl, "w");
  int line_length = floor((int)cubeWidth/6);
  T buff[line_length];
  for(int i = 0; i < line_length; i++)
    buff[i] = 0;
  for(int i = 0; i < floor((int)cubeHeight/6)*(int)cubeDepth; i++)
    {
      int err = fwrite(buff, sizeof(T), line_length, fp);
      if(err == 0)
        printf("Cube::create_volume_file(%s): error writing the layer %i\n", buff_vl, i);
    }
  fclose(fp);

  Cube<uchar,ulong>* pepe = new Cube<uchar,ulong>(buff_nfo, buff_vl);

  printf("Subsampling mean[");
  //Outer loop, for all the pixels
  for(int z = 0; z < cubeDepth; z++)
    {
      for(int y = 0; y < cubeHeight-6-1; y+=6)
        {
          for(int x = 0; x < cubeWidth-6-1; x+=6)
            {
              int value = 0;
                for(int y2 = 0; y2 < 6; y2++)
                  for(int x2 = 0; x2 < 6; x2++)
                    value+= this->at(x+x2, y+y2, z);
              pepe->put(x/6,y/6,z, (uchar)(value/36));
            }
        }
      printf("#"); fflush(stdout);
    }
  printf("]\n");

//Subsampling /12 /12 /2
//   string filename = "/media/neurons/neuron1/volume_subsampled_12.vl";
//   printf("Creating volume file in %s\n", filename.c_str());
//   FILE* fp = fopen(filename.c_str(), "w");
//   int line_length = floor((int)cubeWidth/12);
//   T buff[line_length];
//   for(int i = 0; i < line_length; i++)
//     buff[i] = 0;
//   or(int i = 0; i < floor((int)cubeHeight/12)*floor((int)cubeDepth/2); i++)
//     {
//       int err = fwrite(buff, sizeof(T), line_length, fp);
//       if(err == 0)
//         printf("Cube::create_volume_file(%s): error writing the layer %i\n", filename.c_str(), i);
//     }
//   fclose(fp);
//   Cube< uchar >* pepe = new Cube< uchar >("/media/neurons/neuron1/volume_subsampled_12.nfo",
//                                          "/media/neurons/neuron1/volume_subsampled_12.vl");

//   printf("Subsampling mean[");
//   fflush(stdout);
//   //Outer loop, for all the pixels
//   for(int z = 0; z < cubeDepth-3; z+=2)
//     {
//       for(int y = 0; y < cubeHeight-12; y+=12)
//         {
//           for(int x = 0; x < cubeWidth-12; x+=12)
//             {
//               int value = 0;
//               for(int z2=0; z2<2; z2++)
//                 for(int y2 = 0; y2 < 12; y2++)
//                   for(int x2 = 0; x2 < 12; x2++){
// //                     printf("%i %i %i\n", x+x2, y+y2, z+z2);
//                     value+= this->at(x+x2, y+y2, z+z2);
//                   }
//               pepe->put(x/12,y/12,z/2, (uchar)(value/288));
//             }
//         }
//       printf("#"); fflush(stdout);
//     }
//   printf("]\n");


}


template <class T, class U>
void Cube<T,U>::subsampleMinimum()
{
  string filename = "/media/neurons/neuron1/volume_subsampled_minimum.vl";
  printf("Creating volume file in %s\n", filename.c_str());
  FILE* fp = fopen(filename.c_str(), "w");
  int line_length = floor((int)cubeWidth/6);
  T buff[line_length];
  for(int i = 0; i < line_length; i++)
    buff[i] = 0;
  for(int i = 0; i < floor((int)cubeHeight/6)*cubeDepth; i++)
    {
      int err = fwrite(buff, sizeof(T), line_length, fp);
      if(err == 0)
        printf("Cube::create_volume_file(%s): error writing the layer %i\n", filename.c_str(), i);
    }
  fclose(fp);
  Cube< uchar,ulong >* pepe = new Cube< uchar,ulong >("/media/neurons/neuron1/volume_subsampled.nfo",
                                         "/media/neurons/neuron1/volume_subsampled_minimum.vl");

  printf("Subsampling minimum[");
  //Outer loop, for all the pixels
  for(int z = 0; z < cubeDepth; z++)
    {
      for(int y = 0; y < cubeHeight; y+=6)
        {
          for(int x = 0; x < cubeWidth; x+=6)
            {
              uchar value = 255;
              for(int y2 = 0; y2 < 6; y2++)
                for(int x2 = 0; x2 < 6; x2++)
                  if (this->at(x+x2, y+y2, z)< value)
                    value = this->at(x+x2, y+y2, z)   ;
              pepe->put(x/6,y/6,z, (uchar)(value));
            }
        }
      printf("#"); fflush(stdout);
    }
  printf("]\n");
}

template <class T, class U>
void Cube<T,U>::substract_mean(string name)
{
  Cube<float,double>* output = create_blank_cube(name);

  double mean_value = 0;
  double mean_value_tmp = 0;

  for(int z = 0; z < cubeDepth; z++){
    for(int y = 0; y < cubeHeight; y++)
      for(int x = 0; x < cubeWidth; x++)
        mean_value_tmp += this->at(x,y,z);
    mean_value += mean_value_tmp/(cubeDepth*cubeHeight*cubeWidth);
    mean_value_tmp = 0;
  }

  printf("The mean value is %f\n",mean_value);

  for(int z = 0; z < cubeDepth; z++)
    for(int y = 0; y < cubeHeight; y++)
      for(int x = 0; x < cubeWidth; x++)
        output->put(x,y,z,this->at(x,y,z) - mean_value);
  delete output;
}


// This is stolen from libvision and modified
#define MIN_RATIO .05
#define MAX_WIDTH  20
#define VERB  0
template< class T, class U >
int Cube<T,U>::gaussian_mask(float sigma, vector< float >& Mask0, vector< float >& Mask1)
{
  float val,sum,Aux0[MAX_WIDTH],Aux1[MAX_WIDTH],coeff=-2.0/(sigma*sigma);
  int i,j,k,n;

  for(i=0;i<MAX_WIDTH;i++) Aux0[i]=0;

  val=sum=Aux0[0]=1.0;
  Aux1[0]=0.0;

  for(n=1;n<MAX_WIDTH;n++){
    val  = n/sigma;
    Aux0[n] = val = exp (- val*val);
    Aux1[n] = coeff * val * n;
    sum += (2.0*val);
    if(val<MIN_RATIO)
      break;
  }

  if(MAX_WIDTH==n){
    fprintf(stderr,"GaussianMask: %f too large, truncating mask.\n",sigma);
    n--;
  }

    Mask0.resize(1+2*n);
    Mask0[n]=1.0/sum;
    Mask1.resize(1+2*n);
    Mask1[n]=0.0;

#if 0
  fprintf(stderr,"--> %f\n",sum);
  for(i=0;i<=n;i++) fprintf(stderr,"%f ",Aux1[i]);
#endif


  for(i=n+1,j=n-1,k=1;k<=n;i++,j--,k++){
      Mask0[i]=Mask0[j]=Aux0[k]/sum;
      Mask1[i]=Aux1[k]/sum;
      Mask1[j]=-(Mask1[i]);
  }
return(n);
}


template <class T, class U>
int Cube<T,U>::gaussian_mask_second
(float sigma, vector< float >& Mask0, vector< float >& Mask1)
{
  float val,sum,Aux0[MAX_WIDTH],Aux1[MAX_WIDTH],coeff=-2.0/(sigma*sigma),sum2;
  int i,j,k,n;

  for(i=0;i<MAX_WIDTH;i++) Aux0[i]=0;

  val=sum=Aux0[0]=1.0;
  sum2 = 0;
  Aux1[0]=0.0;

  for(n=1;n<MAX_WIDTH;n++){
    val  = float(n)/sigma;
    Aux0[n] = exp (- val*val/2);
    Aux1[n] = Aux0[n]*(float(n*n)/(sigma*sigma)-1)/
      (sigma*sigma*sigma*sqrt(2*3.14159));
    sum += (2.0*Aux0[n]);
    sum2 += Aux1[n];
    if(Aux0[n]<MIN_RATIO)
      break;
  }

  if(MAX_WIDTH==n){
    fprintf(stderr,"GaussianMask: %f too large, truncating mask.\n",sigma);
    n--;
  }

    Mask0.resize(1+2*n);
    Mask0[n]=1.0/sum;
    Mask1.resize(1+2*n);
    Mask1[n]=-1.0/(sigma*sigma*sigma*sqrt(2*3.14159));
//     Mask1[n]=-1.0/(2*sum2);

#if 0
  fprintf(stderr,"--> %f\n",sum);
  for(i=0;i<=n;i++) fprintf(stderr,"%f ",Aux1[i]);
#endif


  for(i=n+1,j=n-1,k=1;k<=n;i++,j--,k++){
      Mask0[i]=Mask0[j]=Aux0[k]/sum;
      Mask1[i]=Aux1[k];
      Mask1[j]=(Mask1[i]);
  }
return(n);
}


template <class T, class U>
void Cube<T,U>::convolve_horizontally(vector< float >& mask, Cube< float,double >* output, bool use_borders)
{
  printf("Here\n");
  assert(mask.size() > 0);

  int mask_side = mask.size()/2;
  int mask_size = mask.size();
  printf("Cube<T,U>::convolve_horizontally [");

  #ifdef WITH_OPENMP
  #pragma omp parallel for
  #endif
  for(int z = 0; z < cubeDepth; z++){
    int x,q;
    float result;
    for(int y = 0; y < cubeHeight; y++)
      {
        // Beginning of the line
        for(x = 0; x < mask_size; x++){
          result = 0;
          for(q = -mask_side; q <=mask_side; q++){
            if(x+q<0)
              result+=this->at(0,y,z)*mask[mask_side + q];
            else
              result += this->at(x+q,y,z)*mask[mask_side + q];
          }
          output->put(x,y,z,result);
        }

       //Middle of the line
        for(x = mask_size; x <= cubeWidth-mask_size-1; x++)
          {
            result = 0;
            for(q = -mask_side; q <=mask_side; q++)
              result += this->at(x+q,y,z)*mask[mask_side + q];
            output->put(x,y,z,result);
            // printf("%i %i %i\n", x, y, z);
          }
        //End of the line
        for(x = cubeWidth-mask_size; x < cubeWidth; x++){
          result = 0;
          for(q = -mask_side; q <=mask_side; q++){
            if(x+q >= cubeWidth)
              result+=this->at(cubeWidth-1,y,z)*mask[mask_side + q];
            else
              result += this->at(x+q,y,z)*mask[mask_side + q];
          }
          output->put(x,y,z,result);
        }
      }
    if(z%(cubeDepth/20)==0)
      printf("#");fflush(stdout);
  }
  printf("]\n");
}

template <class T, class U>
void Cube<T,U>::convolve_vertically(vector< float >& mask, Cube<float,double>* output, bool use_borders)
{
  assert(mask.size() > 0);
  int mask_side = mask.size()/2;
  int mask_size = mask.size();

  printf("Cube<T,U>::convolve_vertically [");
  #ifdef WITH_OPENMP
  #pragma omp parallel for
  #endif
  for(int z = 0; z < cubeDepth; z++){
    float result = 0;
    int q = 0;
    int y = 0;
    for(int x = 0; x < cubeWidth; x++)
      {

        //Beginning of the line
        for(y = 0; y < mask_size; y++){
          result = 0;
          for(q = -mask_side; q <=mask_side; q++){
            if(y+q<0)
              result+=this->at(x,0,z)*mask[mask_side + q];
            else
              result += this->at(x,y+q,z)*mask[mask_side + q];
          }
          output->put(x,y,z,result);
        }

        //Middle of the line
        for(y = mask_size; y <= cubeHeight-mask_size-1; y++)
          {
            result = 0;
            for(q = -mask_side; q <=mask_side; q++)
              result += this->at(x,y+q,z)*mask[mask_side + q];
            output->put(x,y,z,result);
          }

        //End of the line
        for(y = cubeHeight-mask_size; y < cubeHeight; y++){
          result = 0;
          for(q = -mask_side; q <=mask_side; q++){
            if(y+q >= cubeHeight)
              result+=this->at(x,cubeHeight-1,z)*mask[mask_side + q];
            else
              result += this->at(x,y+q,z)*mask[mask_side + q];
          }
          output->put(x,y,z,result);
        }
      }
    if(z%(cubeDepth/20)==0)
      printf("#");fflush(stdout);
  }
  printf("]\n");
}

template <class T, class U>
void Cube<T,U>::convolve_depth(vector< float >& mask, Cube<float,double>* output, bool use_borders)
{
  assert(mask.size() > 0);
  int mask_side = mask.size()/2;
  int mask_size = mask.size();

  printf("Cube<T,U>::convolve_depth [");
  #ifdef WITH_OPENMP
  #pragma omp parallel for
  #endif
  for(int y = 0; y < cubeHeight; y++){
    float result = 0;
    int q = 0;
    int z = 0;
    for(int x = 0; x < cubeWidth; x++){

      for(z = 0; z < mask_size; z++){
        result = 0;
        for(q = -mask_side; q <=mask_side; q++){
          if(z+q<0)
            result+=this->at(x,y,0)*mask[mask_side + q];
          else if (z+q >= cubeDepth)
            result+=this->at(x,y,cubeDepth-1)*mask[mask_side + q];
          else
            result += this->at(x,y,z+q)*mask[mask_side + q];
        }
        output->put(x,y,z,result);
      }

      for(z = mask_size; z <cubeDepth - mask_size; z++){
        result = 0;
        for(q = -mask_side; q <=mask_side; q++)
          result += this->at(x,y,z+q)*mask[mask_side + q];
        output->put(x,y,z,result);
      }

      for(z = cubeDepth-mask_size; z < cubeDepth; z++){
        result = 0;
        for(q = -mask_side; q <=mask_side; q++){
          if(z+q >= cubeDepth)
            result+=this->at(x,y,cubeDepth-1)*mask[mask_side + q];
          else if(z+q<0)
            result+=this->at(x,y,0)*mask[mask_side + q];
          else
            result += this->at(x,y,z+q)*mask[mask_side + q];
        }
        output->put(x,y,z,result);
      }
    }
    if(y%(cubeHeight/20)==0)
      printf("#");fflush(stdout);
  }
  printf("]\n");
}

template <class T, class U>
void Cube<T,U>::gradient_x(float sigma_xy, float sigma_z, Cube<float,double>* output, Cube< float,double>* tmp)
{
  vector< float > mask0;
  vector< float > mask1;
  gaussian_mask(sigma_xy, mask0, mask1);
  vector< float > mask0_z;
  vector< float > mask1_z;
  gaussian_mask(sigma_z, mask0_z, mask1_z);


  this->convolve_depth(mask0_z, output);
  output->convolve_vertically(mask0, tmp);
  tmp->convolve_horizontally(mask1, output);
}

template <class T, class U>
void Cube<T,U>::gradient_y(float sigma_xy, float sigma_z, Cube<float,double>* output, Cube<float,double>* tmp)
{
  vector< float > mask0;
  vector< float > mask1;
  gaussian_mask(sigma_xy, mask0, mask1);
  vector< float > mask0_z;
  vector< float > mask1_z;
  gaussian_mask(sigma_z, mask0_z, mask1_z);

  this->convolve_depth(mask0_z, output);
  output->convolve_horizontally(mask0, tmp);
  tmp->convolve_vertically(mask1, output);
}

template <class T, class U>
void Cube<T,U>::gradient_z(float sigma_xy, float sigma_z, Cube<float,double>* output, Cube<float,double>* tmp)
{
  vector< float > mask0;
  vector< float > mask1;
  gaussian_mask(sigma_xy, mask0, mask1);
  vector< float > mask0_z;
  vector< float > mask1_z;
  gaussian_mask(sigma_z, mask0_z, mask1_z);

  this->convolve_vertically(mask0, output);
  output->convolve_horizontally(mask0, tmp);
  tmp->convolve_depth(mask1_z, output);
}

template <class T, class U>
void  Cube<T,U>::second_derivate_xx
(float sigma_xy, float sigma_z, Cube<float,double>* output, Cube<float,double>* tmp)
{
  vector< float > mask0;
  vector< float > mask1;
  gaussian_mask_second(sigma_xy, mask0, mask1);
  vector< float > mask0_z;
  vector< float > mask1_z;
  gaussian_mask(sigma_z, mask0_z, mask1_z);

  this->convolve_depth(mask0_z, output);
  output->convolve_vertically(mask0, tmp);
  tmp->convolve_horizontally(mask1, output);
}

template <class T, class U>
void  Cube<T,U>::second_derivate_yy
(float sigma_xy, float sigma_z, Cube<float,double>* output, Cube<float,double>* tmp)
{
  vector< float > mask0;
  vector< float > mask1;
  gaussian_mask_second(sigma_xy, mask0, mask1);
  vector< float > mask0_z;
  vector< float > mask1_z;
  gaussian_mask(sigma_z, mask0_z, mask1_z);

  this->convolve_depth(mask0_z, output);
  output->convolve_horizontally(mask0, tmp);
  tmp->convolve_vertically(mask1, output);
}

template <class T, class U>
void  Cube<T,U>::second_derivate_zz
(float sigma_xy, float sigma_z, Cube<float,double>* output, Cube<float,double>* tmp)
{
  vector< float > mask0;
  vector< float > mask1;
  gaussian_mask_second(sigma_xy, mask0, mask1);
  vector< float > mask0_z;
  vector< float > mask1_z;
  gaussian_mask_second(sigma_z, mask0_z, mask1_z);

  this->convolve_vertically(mask0, output);
  output->convolve_horizontally(mask0, tmp);
  tmp->convolve_depth(mask1_z, output);
}

template <class T, class U>
void  Cube<T,U>::second_derivate_xy
(float sigma_xy, float sigma_z, Cube<float,double>* output, Cube<float,double>* tmp)
{
  vector< float > mask0;
  vector< float > mask1;
  gaussian_mask(sigma_xy, mask0, mask1);
  vector< float > mask0_z;
  vector< float > mask1_z;
  gaussian_mask(sigma_z, mask0_z, mask1_z);

  this->convolve_depth(mask0_z, output);
  output->convolve_vertically(mask1, tmp);
  tmp->convolve_horizontally(mask1, output);
}

template <class T, class U>
void  Cube<T,U>::second_derivate_xz
(float sigma_xy, float sigma_z, Cube<float,double>* output, Cube<float,double>* tmp)
{
  vector< float > mask0;
  vector< float > mask1;
  gaussian_mask(sigma_xy, mask0, mask1);
  vector< float > mask0_z;
  vector< float > mask1_z;
  gaussian_mask(sigma_z, mask0_z, mask1_z);

  this->convolve_vertically(mask0, output);
  output->convolve_horizontally(mask1, tmp);
  tmp->convolve_depth(mask1_z,output);
}

template <class T, class U>
void  Cube<T,U>::second_derivate_yz
(float sigma_xy, float sigma_z, Cube<float,double>* output, Cube<float,double>* tmp)
{
  vector< float > mask0;
  vector< float > mask1;
  gaussian_mask(sigma_xy, mask0, mask1);
  vector< float > mask0_z;
  vector< float > mask1_z;
  gaussian_mask(sigma_z, mask0_z, mask1_z);

  this->convolve_horizontally(mask0, output);
  output->convolve_vertically(mask1, tmp);
  tmp->convolve_depth(mask1_z,output);
}

template <class T, class U>
void Cube<T,U>::blur(float sigma,
                     Cube<float, double>* output,
                     Cube<float,double>* tmp)
{
  vector< float > mask0;
  vector< float > mask1;
  gaussian_mask(sigma, mask0, mask1);

  this->convolve_horizontally(mask0, output,false);
  output->convolve_vertically(mask0, tmp, false);
  tmp->convolve_depth(mask0,output,false);
}

template <class T, class U>
void Cube<T,U>::blur_2D(float sigma,
                     Cube<float, double>* output,
                     Cube<float,double>* tmp)
{
  vector< float > mask0;
  vector< float > mask1;
  gaussian_mask(sigma, mask0, mask1);

  this->convolve_horizontally(mask0, tmp,false);
  tmp->convolve_vertically(mask0, output, false);
}


template <class T, class U>
void Cube<T,U>::calculate_derivative
(int nx, int ny, int nz,
 float sigma_x, float sigma_y, float sigma_z,
 Cube<float, double>* output, Cube<float, double>* tmp)
{
  vector< float > mask_x = Mask::gaussian_mask(nx, sigma_x, true);
  vector< float > mask_y = Mask::gaussian_mask(ny, sigma_y, true);
  vector< float > mask_z = Mask::gaussian_mask(nz, sigma_z, true);

  this->convolve_horizontally(mask_x, output, true);
  output->convolve_vertically(mask_y, tmp,    true);
  tmp->convolve_depth(        mask_z, output, true);
}


template <class T, class U>
void Cube<T,U>::calculate_second_derivates(float sigma_xy, float sigma_z)
{
  char file_cubeSteer[1024];
  sprintf(file_cubeSteer, "%scubeSteer_%02.2f_%02.2f.nfo", 
          directory.c_str(), sigma_xy, sigma_z);

  std::ofstream out(file_cubeSteer);

  printf("Cube<T,U>::calculate_second_derivates(): creating the temporary directory\n");

  Cube<float,double>* tmp = create_blank_cube("tmp");

  string vl = ".vl";
  string nfo = ".nfo";
  char vol_name[512];
  vector< string > names(6);
  names[0] = "gxx";
  names[1] = "gxy";
  names[2] = "gxz";
  names[3] = "gyy";
  names[4] = "gyz";
  names[5] = "gzz";

  Cube<float,double>* derivates;

  for(int i = 0; i < 6; i++){

    out << names[i] << " ";
    sprintf(vol_name, "%s_%02.2f_%02.2f", names[i].c_str(),sigma_xy,sigma_z);

    char all_name[1024];
    sprintf(all_name, "%s%s.nfo", directory.c_str(), vol_name);

    if(fileExists(all_name))
      continue;

    derivates = create_blank_cube(vol_name);

    out << directory << vol_name << ".nfo" << std::endl;

    printf("Cube<T,U>::calculate_second_derivates(): creating %s\n", names[i].c_str());

    if(i==0)
      this->second_derivate_xx(sigma_xy, sigma_z, derivates, tmp);
    if(i==1)
      this->second_derivate_xy(sigma_xy, sigma_z, derivates, tmp);
    if(i==2)
      this->second_derivate_xz(sigma_xy, sigma_z, derivates, tmp);
    if(i==3)
      this->second_derivate_yy(sigma_xy, sigma_z, derivates, tmp);
    if(i==4)
      this->second_derivate_yz(sigma_xy, sigma_z, derivates, tmp);
    if(i==5)
      this->second_derivate_zz(sigma_xy, sigma_z, derivates, tmp);

    delete derivates;
  }
  out.close();
}



template <class T, class U>
void Cube<T,U>::calculate_eigen_values
(float sigma_xy, float sigma_z, bool calculate_eigne_vectors)
{

  vector< float > Mask0, Mask1;
  gaussian_mask_second(sigma_xy, Mask0, Mask1);
  int margin = Mask0.size()/2;

  char vol_name[1024];

  //Load the input of the hessian
  sprintf(vol_name, "%sgxx_%02.2f_%02.2f.nfo", directory.c_str(), sigma_xy, sigma_z);
  Cube<float,double>* gxx = new Cube<float,double>(vol_name);
  sprintf(vol_name, "%sgxy_%02.2f_%02.2f.nfo", directory.c_str(), sigma_xy, sigma_z);
  Cube<float,double>* gxy = new Cube<float,double>(vol_name);
  sprintf(vol_name, "%sgxz_%02.2f_%02.2f.nfo", directory.c_str(), sigma_xy, sigma_z);
  Cube<float,double>* gxz = new Cube<float,double>(vol_name);
  sprintf(vol_name, "%sgyy_%02.2f_%02.2f.nfo", directory.c_str(), sigma_xy, sigma_z);
  Cube<float,double>* gyy = new Cube<float,double>(vol_name);
  sprintf(vol_name, "%sgyz_%02.2f_%02.2f.nfo", directory.c_str(), sigma_xy, sigma_z);
  Cube<float,double>* gyz = new Cube<float,double>(vol_name);
  sprintf(vol_name, "%sgzz_%02.2f_%02.2f.nfo", directory.c_str(), sigma_xy, sigma_z);
  Cube<float,double>* gzz = new Cube<float,double>(vol_name);

  vector< Cube<float,double>* > eign_values(3);
  vector< Cube<float,double>* > eign_vector(9);

  string nfo = ".nfo";
  string vl  = ".vl";
  vector< string > l_names(3);
  sprintf(vol_name, "lambda1_%02.2f_%02.2f", sigma_xy, sigma_z);
  l_names[0] = vol_name;
  sprintf(vol_name, "lambda2_%02.2f_%02.2f", sigma_xy, sigma_z);
  l_names[1] = vol_name;
  sprintf(vol_name, "lambda3_%02.2f_%02.2f", sigma_xy, sigma_z);
  l_names[2] = vol_name;

  //Checks if the files exists, if so, return
  //FIXME
  bool all_exist = true;
  for(int i = 1; i < 3; i++){
    // spprintf(vol_name, "%
  }

  vector< string > v_names(9);
  sprintf(vol_name, "lambda1_vx_%02.2f_%02.2f", sigma_xy, sigma_z);
  v_names[0] = vol_name;
  sprintf(vol_name, "lambda1_vy_%02.2f_%02.2f", sigma_xy, sigma_z);
  v_names[1] = vol_name;
  sprintf(vol_name, "lambda1_vz_%02.2f_%02.2f", sigma_xy, sigma_z);
  v_names[2] = vol_name;
  sprintf(vol_name, "lambda2_vx_%02.2f_%02.2f", sigma_xy, sigma_z);
  v_names[3] = vol_name;
  sprintf(vol_name, "lambda2_vy_%02.2f_%02.2f", sigma_xy, sigma_z);
  v_names[4] = vol_name;
  sprintf(vol_name, "lambda2_vz_%02.2f_%02.2f", sigma_xy, sigma_z);
  v_names[5] = vol_name;
  sprintf(vol_name, "lambda3_vx_%02.2f_%02.2f", sigma_xy, sigma_z);
  v_names[6] = vol_name;
  sprintf(vol_name, "lambda3_vy_%02.2f_%02.2f", sigma_xy, sigma_z);
  v_names[7] = vol_name;
  sprintf(vol_name, "lambda3_vz_%02.2f_%02.2f", sigma_xy, sigma_z);
  v_names[8] = vol_name;

  for(int i = 0; i < 3; i++){
    printf("Cube<T,U>::calculate_eigen_values(): creating %s\n", l_names[i].c_str());
    eign_values[i] = create_blank_cube(l_names[i]);
  }

  if(calculate_eigne_vectors){
    for(int i = 0; i < 9; i++){
      printf("Cube<T,U>::calculate_eigen_values(): creating %s\n", v_names[i].c_str());
      eign_vector[i] = create_blank_cube(v_names[i]);
    }
  }

  gsl_vector *eign = gsl_vector_alloc (3);
  gsl_matrix *evec = gsl_matrix_alloc (3, 3);

  gsl_eigen_symm_workspace* w =  gsl_eigen_symm_alloc (3);
  gsl_eigen_symmv_workspace* w2 =  gsl_eigen_symmv_alloc (3);

  double data[9];

  float l1 = 0;
  float l2 = 0;
  float l3 = 0;

  float l1_tmp = 0;
  float l2_tmp = 0;
  float l3_tmp = 0;

  printf("Cube<T,U>::calculate_eigen_values[ ");
  for(int z = margin; z < cubeDepth-margin; z++){
    for(int y = margin; y < cubeHeight-margin; y++){
      for(int x = margin; x < cubeWidth-margin; x++){

        data[0] = gxx->at(x,y,z);
        data[1] = gxy->at(x,y,z);
        data[2] = gxz->at(x,y,z);
        data[3] = data[1];
        data[4] = gyy->at(x,y,z);
        data[5] = gyz->at(x,y,z);
        data[6] = data[2];
        data[7] = data[5];
        data[8] = gzz->at(x,y,z);

        gsl_matrix_view M
          = gsl_matrix_view_array (data, 3, 3);

        if(calculate_eigne_vectors){
          gsl_eigen_symmv (&M.matrix, eign, evec, w2);}
        else{
          gsl_eigen_symm (&M.matrix, eign, w);}

        l1 = gsl_vector_get (eign, 0);
        l2 = gsl_vector_get (eign, 1);
        l3 = gsl_vector_get (eign, 2);

        eign_values[0]->put(x,y,z, l1 );
        eign_values[1]->put(x,y,z, l2 );
        eign_values[2]->put(x,y,z, l3 );

        if(calculate_eigne_vectors){
          eign_vector[0]->put(x,y,z,gsl_matrix_get(&M.matrix, 0,0));
          eign_vector[1]->put(x,y,z,gsl_matrix_get(&M.matrix, 1,0));
          eign_vector[2]->put(x,y,z,gsl_matrix_get(&M.matrix, 2,0));
          eign_vector[3]->put(x,y,z,gsl_matrix_get(&M.matrix, 0,1));
          eign_vector[4]->put(x,y,z,gsl_matrix_get(&M.matrix, 1,1));
          eign_vector[5]->put(x,y,z,gsl_matrix_get(&M.matrix, 2,1));
          eign_vector[6]->put(x,y,z,gsl_matrix_get(&M.matrix, 0,2));
          eign_vector[7]->put(x,y,z,gsl_matrix_get(&M.matrix, 1,2));
          eign_vector[8]->put(x,y,z,gsl_matrix_get(&M.matrix, 2,2));
        }
      }
    }
    printf("#"); fflush(stdout);
  }
  printf("]\n");
  gsl_vector_free (eign);

  delete gxx;
  delete gxy;
  delete gxz;
  delete gyy;
  delete gyz;
  delete gzz;
  delete eign_values[0];
  delete eign_values[1];
  delete eign_values[2];

  if(calculate_eigne_vectors)
    for (int i = 0; i < 9; i++)
      delete eign_vector[i];

//   gsl_eigen_symmv_free(w);
}



//Implements the f-measure as described in "multiscale vessel enhancement filtering", by Alejandro F. Frangi
//The third eigenvalue will be scaled by 4 to account for the ellipsis of the filament, it is not a tube in the 
// subsampled images, but an ellipsoid
template <class T, class U>
void Cube<T,U>::calculate_f_measure(float sigma_xy, float sigma_z)
{

  vector< float > Mask0, Mask1;
  gaussian_mask_second(sigma_xy, Mask0, Mask1);
  int margin = Mask0.size()/2;

  char buff[1024];
  sprintf(buff, "%slambda1_%02.2f_%02.2f.nfo", directory.c_str(), sigma_xy, sigma_z);
  Cube<float,double>* eign1 = new Cube<float,double>(buff);
  sprintf(buff, "%slambda2_%02.2f_%02.2f.nfo", directory.c_str(), sigma_xy, sigma_z);
  Cube<float,double>* eign2 = new Cube<float,double>(buff);
  sprintf(buff, "%slambda3_%02.2f_%02.2f.nfo", directory.c_str(), sigma_xy, sigma_z);
  Cube<float,double>* eign3 = new Cube<float,double>(buff);

  sprintf(buff, "f_measure_%02.2f_%02.2f", sigma_xy, sigma_z);
  Cube<float,double>* f_measure = create_blank_cube(buff);

  printf("Cube<T,U>::calculate_f_measure, getting the max norm of the hessians [");

  float max_s = 0;
  float s;

  for(int z = margin; z < cubeDepth-margin; z++){
    for(int y = margin; y < cubeHeight-margin; y++){
      for(int x = margin; x < cubeWidth-margin; x++){
        s = eign1->at(x,y,z)*eign1->at(x,y,z) +
            eign2->at(x,y,z)*eign2->at(x,y,z) +
            eign3->at(x,y,z)*eign3->at(x,y,z);
        f_measure->put(x,y,z,s);
        if (s > max_s) max_s = s;
//         printf("%f %f %f -> %f\n", eign1->at(x,y,z), eign2->at(x,y,z), eign3->at(x,y,z),s);
      }
    }
    printf("#");fflush(stdout);
  }
  printf("]\n   - the max is %f\n", max_s);

  float l1,l2,l3,l1_t,l2_t,l3_t;
  printf("Cube<T,U>::calculate_F_measure: [");
  for(int z = margin; z < cubeDepth-margin; z++){
    for(int y =  margin; y < cubeHeight-margin; y++){
      for(int x =  margin; x < cubeWidth-margin; x++){
        l1_t = eign1->at(x,y,z);
        l2_t = eign2->at(x,y,z);
        l3_t = eign3->at(x,y,z);

        if( (l1_t <= l2_t) && (l2_t <= l3_t)){
          l1 = l1_t;
          l2 = l2_t;
          l3 = l3_t;
        }
        if( (l1_t <= l3_t) && (l3_t <= l2_t)){
          l1 = l1_t;
          l2 = l3_t;
          l3 = l2_t;
        }
        if( (l2_t <= l1_t) && (l1_t <= l3_t)){
          l1 = l2_t;
          l2 = l1_t;
          l3 = l3_t;
        }
        if( (l2_t <= l3_t) && (l3_t <= l1_t)){
          l1 = l2_t;
          l2 = l3_t;
          l3 = l1_t;
        }
        if( (l3_t <= l2_t) && (l2_t <= l1_t)){
          l1 = l3_t;
          l2 = l2_t;
          l3 = l1_t;
        }
        if( (l3_t <= l1_t) && (l1_t <= l2_t)){
          l1 = l3_t;
          l2 = l1_t;
          l3 = l2_t;
        }
//         if( (l2 < 0) || (l3 < 0)){
//             f_measure->put(x,y,z,0);
//             continue;
//         }
        s = ( 1 - exp( -l2*l2/(l3*l3*0.5) ) )*
            exp(-l1*l1/(fabs(l2*l3)*0.5) )*
            (1 - exp(-2*f_measure->at(x,y,z)/(max_s)));
        f_measure->put(x,y,z,s);

//         printf("%f %f %f \n", l1, l2, l3);
      }
    }
    printf("#");fflush(stdout);
  }
  printf("]\n");
}


template <class T, class U>
void Cube<T,U>::calculate_aguet(float sigma_xy, float sigma_z)
{

  if(sigma_z == 0) sigma_z = sigma_xy;

  char vol_name[1024];

  vector< float > Mask0, Mask1;
  gaussian_mask_second(sigma_xy, Mask0, Mask1);
  // int margin = Mask0.size()/2;
  int margin = 0;

  calculate_second_derivates(sigma_xy, sigma_z);

  //Load the input of the hessian
  sprintf(vol_name, "%sgxx_%02.2f_%02.2f.nfo", directory.c_str(), sigma_xy, sigma_z);
  Cube<float,double>* gxx = new Cube<float,double>(vol_name);
  sprintf(vol_name, "%sgxy_%02.2f_%02.2f.nfo", directory.c_str(), sigma_xy, sigma_z);
  Cube<float,double>* gxy = new Cube<float,double>(vol_name);
  sprintf(vol_name, "%sgxz_%02.2f_%02.2f.nfo", directory.c_str(), sigma_xy, sigma_z);
  Cube<float,double>* gxz = new Cube<float,double>(vol_name);
  sprintf(vol_name, "%sgyy_%02.2f_%02.2f.nfo", directory.c_str(), sigma_xy, sigma_z);
  Cube<float,double>* gyy = new Cube<float,double>(vol_name);
  sprintf(vol_name, "%sgyz_%02.2f_%02.2f.nfo", directory.c_str(), sigma_xy, sigma_z);
  Cube<float,double>* gyz = new Cube<float,double>(vol_name);
  sprintf(vol_name, "%sgzz_%02.2f_%02.2f.nfo", directory.c_str(), sigma_xy, sigma_z);
  Cube<float,double>* gzz = new Cube<float,double>(vol_name);

  sprintf(vol_name, "aguet_%02.2f_%02.2f", sigma_xy, sigma_z);
  Cube<float,double>* aguet_l = create_blank_cube(vol_name);
  sprintf(vol_name, "aguet_%02.2f_%02.2f_theta", sigma_xy, sigma_z);
  Cube<float,double>* aguet_theta = create_blank_cube(vol_name);
  sprintf(vol_name, "aguet_%02.2f_%02.2f_phi", sigma_xy, sigma_z);
  Cube<float,double>* aguet_phi = create_blank_cube(vol_name);



  int nthreads = 1;
#ifdef WITH_OPENMP
  nthreads = omp_get_max_threads();
  printf("Cube<T,U>::calculate_aguet: using %i threads\n", nthreads);
#endif

  printf("Cube<T,U>::calculate_aguet [");
  fflush(stdout);

  //Initialization of the places where each thread will work
  vector< gsl_vector* > eign(nthreads);
  vector< gsl_matrix* > evec(nthreads);
  vector< gsl_eigen_symmv_workspace* > w2(nthreads);
  for(int i = 0; i < nthreads; i++){
    eign[i] = gsl_vector_alloc (3);
    evec[i] = gsl_matrix_alloc (3, 3);
    w2[i]   = gsl_eigen_symmv_alloc (3);
  }

#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
  for(int z = margin; z < cubeDepth-margin; z++){
    int tn = omp_get_thread_num();
    //Variables defined in the loop for easy parallel processing
    float l1,l2,l3, theta, phi, r;
    double data[9];
    int higher_eival = 0;

    for(int y = margin; y < cubeHeight-margin; y++){
      for(int x = margin; x < cubeWidth-margin; x++){
        // data[0] = -2.0*gxx->at(x,y,z)/3.0
          // + gyy->at(x,y,z)
          // + gzz->at(x,y,z);
        // data[1] = -5.0*gxy->at(x,y,z)/3.0;
        // data[2] = -5.0*gxz->at(x,y,z)/3.0;
        // data[3] = data[1];
        // data[4] = gxx->at(x,y,z)
          // - 2.0*gyy->at(x,y,z)/3.0
          // + gzz->at(x,y,z);
        // data[5] = -5.0*gyz->at(x,y,z)/3.0;
        // data[6] = data[2];
        // data[7] = data[5];
        // data[8] = -2.0*gzz->at(x,y,z)/3.0
          // + gyy->at(x,y,z)
          // + gxx->at(x,y,z);

        //There is an screwed sign in the computation of gy, therefore
        // the inversion of all the coefficients with one derivative
        // in y

        data[0] = -2.0*gxx->at(x,y,z)/3.0
          + gyy->at(x,y,z)
          + gzz->at(x,y,z);
        data[1] = 5.0*gxy->at(x,y,z)/3.0;
        data[2] = -5.0*gxz->at(x,y,z)/3.0;
        data[3] = data[1];
        data[4] = gxx->at(x,y,z)
          - 2.0*gyy->at(x,y,z)/3.0
          + gzz->at(x,y,z);
        data[5] = 5.0*gyz->at(x,y,z)/3.0;
        data[6] = data[2];
        data[7] = data[5];
        data[8] = -2.0*gzz->at(x,y,z)/3.0
          + gyy->at(x,y,z)
          + gxx->at(x,y,z);


        gsl_matrix_view M
          = gsl_matrix_view_array (data, 3, 3);

        gsl_eigen_symmv (&M.matrix, eign[tn], evec[tn], w2[tn]);

        l1 = gsl_vector_get (eign[tn], 0);
        l2 = gsl_vector_get (eign[tn], 1);
        l3 = gsl_vector_get (eign[tn], 2);

        if( (l1>=l2)&&(l1>=l3)){
          higher_eival = 0;
          aguet_l->put(x,y,z,l1);
        }
        if( (l2>=l1)&&(l2>=l3)){
          higher_eival = 1;
          aguet_l->put(x,y,z,l2);
        }
        if( (l3>=l2)&&(l3>=l1)){
          higher_eival = 2;
          aguet_l->put(x,y,z,l3);
        }

        r = sqrt(gsl_matrix_get(evec[tn],0,higher_eival)*
                 gsl_matrix_get(evec[tn],0,higher_eival)+
                 gsl_matrix_get(evec[tn],1,higher_eival)*
                 gsl_matrix_get(evec[tn],1,higher_eival)+
                 gsl_matrix_get(evec[tn],2,higher_eival)*
                 gsl_matrix_get(evec[tn],2,higher_eival)
                 );

        theta = atan2(gsl_matrix_get(evec[tn],1,higher_eival),
                      gsl_matrix_get(evec[tn],0,higher_eival));
        phi   = acos(gsl_matrix_get(evec[tn],2,higher_eival)/r);

        aguet_theta->put(x,y,z,theta);
        aguet_phi->put(x,y,z,theta);
      }
    }
    printf("#");fflush(stdout);
  }
  printf("]\n");
}

template <class T, class U>
void Cube<T,U>::calculate_aguet_f_(float sigma_xy, float sigma_z)
{

  if(sigma_z == 0) sigma_z = sigma_xy;

  char vol_name[1024];

  vector< float > Mask0, Mask1;
  gaussian_mask_second(sigma_xy, Mask0, Mask1);
  // int margin = Mask0.size()/2;
  int margin = 0;

  calculate_second_derivates(sigma_xy, sigma_z);

  //Load the input of the hessian
  sprintf(vol_name, "%sgxx_%02.2f_%02.2f.nfo", directory.c_str(), sigma_xy, sigma_z);
  Cube<float,double>* gxx = new Cube<float,double>(vol_name);
  sprintf(vol_name, "%sgxy_%02.2f_%02.2f.nfo", directory.c_str(), sigma_xy, sigma_z);
  Cube<float,double>* gxy = new Cube<float,double>(vol_name);
  sprintf(vol_name, "%sgxz_%02.2f_%02.2f.nfo", directory.c_str(), sigma_xy, sigma_z);
  Cube<float,double>* gxz = new Cube<float,double>(vol_name);
  sprintf(vol_name, "%sgyy_%02.2f_%02.2f.nfo", directory.c_str(), sigma_xy, sigma_z);
  Cube<float,double>* gyy = new Cube<float,double>(vol_name);
  sprintf(vol_name, "%sgyz_%02.2f_%02.2f.nfo", directory.c_str(), sigma_xy, sigma_z);
  Cube<float,double>* gyz = new Cube<float,double>(vol_name);
  sprintf(vol_name, "%sgzz_%02.2f_%02.2f.nfo", directory.c_str(), sigma_xy, sigma_z);
  Cube<float,double>* gzz = new Cube<float,double>(vol_name);

  sprintf(vol_name, "aguet_f_%02.2f_%02.2f", sigma_xy, sigma_z);
  Cube<float,double>* aguet_l = create_blank_cube(vol_name);

  int nthreads = 1;
#ifdef WITH_OPENMP
  nthreads = omp_get_max_threads();
  printf("Cube<T,U>::calculate_aguet: using %i threads\n", nthreads);
#endif
  printf("Cube<T,U>::calculate_aguet_f [");
  fflush(stdout);

  //Initialization of the places where each thread will work
  vector< gsl_vector* > eign(nthreads);
  vector< gsl_matrix* > evec(nthreads);
  vector< gsl_eigen_symmv_workspace* > w2(nthreads);
  for(int i = 0; i < nthreads; i++){
    eign[i] = gsl_vector_alloc (3);
    evec[i] = gsl_matrix_alloc (3, 3);
    w2[i]   = gsl_eigen_symmv_alloc (3);
  }

  float max_s = 0;
  float s = 0;
  //Get the maximum of the absolute value of the eigenvalues of the hessian like matrix
#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
  for(int z = margin; z < cubeDepth-margin; z++){
    int tn = omp_get_thread_num();
    //Variables defined in the loop for easy parallel processing
    float l1,l2,l3, theta, phi, r, l1_t, l2_t, l3_t;
    double data[9];
    int higher_eival = 0;

    for(int y = margin; y < cubeHeight-margin; y++){
      for(int x = margin; x < cubeWidth-margin; x++){
        //There is an screwed sign in the computation of gy, therefore
        // the inversion of all the coefficients with one derivative
        // in y

        data[0] = -2.0*gxx->at(x,y,z)/3.0
          + gyy->at(x,y,z)
          + gzz->at(x,y,z);
        data[1] = 5.0*gxy->at(x,y,z)/3.0;
        data[2] = -5.0*gxz->at(x,y,z)/3.0;
        data[3] = data[1];
        data[4] = gxx->at(x,y,z)
          - 2.0*gyy->at(x,y,z)/3.0
          + gzz->at(x,y,z);
        data[5] = 5.0*gyz->at(x,y,z)/3.0;
        data[6] = data[2];
        data[7] = data[5];
        data[8] = -2.0*gzz->at(x,y,z)/3.0
          + gyy->at(x,y,z)
          + gxx->at(x,y,z);


        gsl_matrix_view M
          = gsl_matrix_view_array (data, 3, 3);

        gsl_eigen_symmv (&M.matrix, eign[tn], evec[tn], w2[tn]);

        l1_t = gsl_vector_get (eign[tn], 0);
        l2_t = gsl_vector_get (eign[tn], 1);
        l3_t = gsl_vector_get (eign[tn], 2);
        s = l1_t*l1_t +  l2_t*l2_t + l3_t*l3_t;

        aguet_l->put(x,y,z,s);
        if(s > max_s)
          max_s = s;

      }
    }
  }


#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
  for(int z = margin; z < cubeDepth-margin; z++){
    int tn = omp_get_thread_num();
    //Variables defined in the loop for easy parallel processing
    float l1,l2,l3, theta, phi, r, l1_t, l2_t, l3_t;
    double data[9];
    int higher_eival = 0;

    for(int y = margin; y < cubeHeight-margin; y++){
      for(int x = margin; x < cubeWidth-margin; x++){
        //There is an screwed sign in the computation of gy, therefore
        // the inversion of all the coefficients with one derivative
        // in y

        data[0] = -2.0*gxx->at(x,y,z)/3.0
          + gyy->at(x,y,z)
          + gzz->at(x,y,z);
        data[1] = 5.0*gxy->at(x,y,z)/3.0;
        data[2] = -5.0*gxz->at(x,y,z)/3.0;
        data[3] = data[1];
        data[4] = gxx->at(x,y,z)
          - 2.0*gyy->at(x,y,z)/3.0
          + gzz->at(x,y,z);
        data[5] = 5.0*gyz->at(x,y,z)/3.0;
        data[6] = data[2];
        data[7] = data[5];
        data[8] = -2.0*gzz->at(x,y,z)/3.0
          + gyy->at(x,y,z)
          + gxx->at(x,y,z);


        gsl_matrix_view M
          = gsl_matrix_view_array (data, 3, 3);

        gsl_eigen_symmv (&M.matrix, eign[tn], evec[tn], w2[tn]);

        l1_t = gsl_vector_get (eign[tn], 0);
        l2_t = gsl_vector_get (eign[tn], 1);
        l3_t = gsl_vector_get (eign[tn], 2);


        if( (l1_t <= l2_t) && (l2_t <= l3_t)){
          l1 = l1_t;
          l2 = l2_t;
          l3 = l3_t;
        }
        if( (l1_t <= l3_t) && (l3_t <= l2_t)){
          l1 = l1_t;
          l2 = l3_t;
          l3 = l2_t;
        }
        if( (l2_t <= l1_t) && (l1_t <= l3_t)){
          l1 = l2_t;
          l2 = l1_t;
          l3 = l3_t;
        }
        if( (l2_t <= l3_t) && (l3_t <= l1_t)){
          l1 = l2_t;
          l2 = l3_t;
          l3 = l1_t;
        }
        if( (l3_t <= l2_t) && (l2_t <= l1_t)){
          l1 = l3_t;
          l2 = l2_t;
          l3 = l1_t;
        }
        if( (l3_t <= l1_t) && (l1_t <= l2_t)){
          l1 = l3_t;
          l2 = l1_t;
          l3 = l2_t;
        }

        r = l1*l1 + l2*l2 + l3*l3;
        s = ( 1 - exp( -l2*l2/(l3*l3*0.5) ) )*
            exp(-l1*l1/(fabs(l2*l3)*0.5) )*
            (1 - exp(-2*r/(max_s)));
        aguet_l->put(x,y,z,s);

      }
    }
    printf("#");fflush(stdout);
  }
  printf("]\n");
}





template <class T, class U>
void Cube<T,U>::calculate_aguet_flat(float sigma_xy, float sigma_z)
{

  if(sigma_z == 0) sigma_z = sigma_xy;

  char vol_name[1024];

  vector< float > Mask0, Mask1;
  gaussian_mask_second(sigma_xy, Mask0, Mask1);
  // int margin = Mask0.size()/2;
  int margin = 0;

  calculate_second_derivates(sigma_xy, sigma_z);

  //Load the input of the hessian
  sprintf(vol_name, "%sgxx_%02.2f_%02.2f.nfo", directory.c_str(), sigma_xy, sigma_z);
  Cube<float,double>* gxx = new Cube<float,double>(vol_name);
  sprintf(vol_name, "%sgxy_%02.2f_%02.2f.nfo", directory.c_str(), sigma_xy, sigma_z);
  Cube<float,double>* gxy = new Cube<float,double>(vol_name);
  sprintf(vol_name, "%sgxz_%02.2f_%02.2f.nfo", directory.c_str(), sigma_xy, sigma_z);
  Cube<float,double>* gxz = new Cube<float,double>(vol_name);
  sprintf(vol_name, "%sgyy_%02.2f_%02.2f.nfo", directory.c_str(), sigma_xy, sigma_z);
  Cube<float,double>* gyy = new Cube<float,double>(vol_name);
  sprintf(vol_name, "%sgyz_%02.2f_%02.2f.nfo", directory.c_str(), sigma_xy, sigma_z);
  Cube<float,double>* gyz = new Cube<float,double>(vol_name);
  sprintf(vol_name, "%sgzz_%02.2f_%02.2f.nfo", directory.c_str(), sigma_xy, sigma_z);
  Cube<float,double>* gzz = new Cube<float,double>(vol_name);

  sprintf(vol_name, "aguet_%02.2f_%02.2f", sigma_xy, sigma_z);
  Cube<float,double>* aguet_l = create_blank_cube(vol_name);
  sprintf(vol_name, "aguet_%02.2f_%02.2f_theta", sigma_xy, sigma_z);
  Cube<float,double>* aguet_theta = create_blank_cube(vol_name);
  sprintf(vol_name, "aguet_%02.2f_%02.2f_phi", sigma_xy, sigma_z);
  Cube<float,double>* aguet_phi = create_blank_cube(vol_name);



  int nthreads = 1;
#ifdef WITH_OPENMP
  nthreads = omp_get_max_threads();
  printf("Cube<T,U>::calculate_aguet: using %i threads\n", nthreads);
#endif

  printf("Cube<T,U>::calculate_aguet [");
  fflush(stdout);

  //Initialization of the places where each thread will work
  vector< gsl_vector* > eign(nthreads);
  vector< gsl_matrix* > evec(nthreads);
  vector< gsl_eigen_symmv_workspace* > w2(nthreads);
  for(int i = 0; i < nthreads; i++){
    eign[i] = gsl_vector_alloc (3);
    evec[i] = gsl_matrix_alloc (3, 3);
    w2[i]   = gsl_eigen_symmv_alloc (3);
  }

#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
  for(int z = margin; z < cubeDepth-margin; z++){
    int tn = omp_get_thread_num();
    //Variables defined in the loop for easy parallel processing
    float l1,l2,l3, theta, phi, r;
    double data[9];
    int higher_eival = 0;

    for(int y = margin; y < cubeHeight-margin; y++){
      for(int x = margin; x < cubeWidth-margin; x++){
        // data[0] = -2.0*gxx->at(x,y,z)/3.0
          // + gyy->at(x,y,z)
          // + gzz->at(x,y,z);
        // data[1] = -5.0*gxy->at(x,y,z)/3.0;
        // data[2] = -5.0*gxz->at(x,y,z)/3.0;
        // data[3] = data[1];
        // data[4] = gxx->at(x,y,z)
          // - 2.0*gyy->at(x,y,z)/3.0
          // + gzz->at(x,y,z);
        // data[5] = -5.0*gyz->at(x,y,z)/3.0;
        // data[6] = data[2];
        // data[7] = data[5];
        // data[8] = -2.0*gzz->at(x,y,z)/3.0
          // + gyy->at(x,y,z)
          // + gxx->at(x,y,z);

        //There is an screwed sign in the computation of gy, therefore
        // the inversion of all the coefficients with one derivative
        // in y

        data[0] = 4.0*gxx->at(x,y,z)
          - gyy->at(x,y,z)
          - gzz->at(x,y,z);
        data[1] = 5.0*gxy->at(x,y,z);
        data[2] = 5.0*gxz->at(x,y,z);
        data[3] = data[1];
        data[4] = -gxx->at(x,y,z)
          + 4.0*gyy->at(x,y,z)
          - gzz->at(x,y,z);
        data[5] = 5.0*gyz->at(x,y,z);
        data[6] = data[2];
        data[7] = data[5];
        data[8] = 4.0*gzz->at(x,y,z)
          - gyy->at(x,y,z)
          - gxx->at(x,y,z);


        gsl_matrix_view M
          = gsl_matrix_view_array (data, 3, 3);

        gsl_eigen_symmv (&M.matrix, eign[tn], evec[tn], w2[tn]);

        l1 = gsl_vector_get (eign[tn], 0);
        l2 = gsl_vector_get (eign[tn], 1);
        l3 = gsl_vector_get (eign[tn], 2);

        if( (l1>=l2)&&(l1>=l3)){
          higher_eival = 0;
          aguet_l->put(x,y,z,l1);
        }
        if( (l2>=l1)&&(l2>=l3)){
          higher_eival = 1;
          aguet_l->put(x,y,z,l2);
        }
        if( (l3>=l2)&&(l3>=l1)){
          higher_eival = 2;
          aguet_l->put(x,y,z,l3);
        }

        r = sqrt(gsl_matrix_get(evec[tn],0,higher_eival)*
                 gsl_matrix_get(evec[tn],0,higher_eival)+
                 gsl_matrix_get(evec[tn],1,higher_eival)*
                 gsl_matrix_get(evec[tn],1,higher_eival)+
                 gsl_matrix_get(evec[tn],2,higher_eival)*
                 gsl_matrix_get(evec[tn],2,higher_eival)
                 );

        theta = atan2(gsl_matrix_get(evec[tn],1,higher_eival),
                      gsl_matrix_get(evec[tn],0,higher_eival));
        phi   = acos(gsl_matrix_get(evec[tn],2,higher_eival)/r);

        aguet_theta->put(x,y,z,theta);
        aguet_phi->put(x,y,z,theta);
      }
    }
    printf("#");fflush(stdout);
  }
  printf("]\n");
}



template <class T, class U>
void Cube<T,U>::order_eigen_values(float sigma_xy, float sigma_z)
{
  char buff[1024];
  sprintf(buff, "%slambda1_%02.2f_%02.2f.nfo", directory.c_str(), sigma_xy, sigma_z);
  Cube<float,double>* eign1 = new Cube<float,double>(buff);
  sprintf(buff, "%slambda2_%02.2f_%02.2f.nfo", directory.c_str(), sigma_xy, sigma_z);
  Cube<float,double>* eign2 = new Cube<float,double>(buff);
  sprintf(buff, "%slambda3_%02.2f_%02.2f.nfo", directory.c_str(), sigma_xy, sigma_z);
  Cube<float,double>* eign3 = new Cube<float,double>(buff);

  sprintf(buff, "lambda1_o_%02.2f_%02.2f", sigma_xy, sigma_z);
  Cube<float,double>* lambda1_o = create_blank_cube(buff);
  sprintf(buff, "lambda2_o_%02.2f_%02.2f", sigma_xy, sigma_z);
  Cube<float,double>* lambda2_o = create_blank_cube(buff);
  sprintf(buff, "lambda3_o_%02.2f_%02.2f", sigma_xy, sigma_z);
  Cube<float,double>* lambda3_o = create_blank_cube(buff);

  float l1,l2,l3,l1_t,l2_t,l3_t;
  printf("Cube<T,U>::calculate_F_measure: [");
  for(int z = 0; z < cubeDepth; z++){
    for(int y = 0; y < cubeHeight; y++){
      for(int x = 0; x < cubeWidth; x++){
        l1_t = eign1->at(x,y,z);
        l2_t = eign2->at(x,y,z);
        l3_t = eign3->at(x,y,z);

        if( (l1_t >= l2_t) && (l2_t >= l3_t)){
          lambda1_o->put(x,y,z,l1_t);
          lambda2_o->put(x,y,z,l2_t);
          lambda3_o->put(x,y,z,l3_t);
        }
        if( (l1_t >= l3_t) && (l3_t >= l2_t)){
          lambda1_o->put(x,y,z,l1_t);
          lambda2_o->put(x,y,z,l3_t);
          lambda3_o->put(x,y,z,l2_t);
        }
        if( (l2_t >= l1_t) && (l1_t >= l3_t)){
          lambda1_o->put(x,y,z,l2_t);
          lambda2_o->put(x,y,z,l1_t);
          lambda3_o->put(x,y,z,l3_t);
        }
        if( (l2_t >= l3_t) && (l3_t >= l1_t)){
          lambda1_o->put(x,y,z,l2_t);
          lambda2_o->put(x,y,z,l3_t);
          lambda3_o->put(x,y,z,l1_t);
        }
        if( (l3_t >= l2_t) && (l2_t >= l1_t)){
          lambda1_o->put(x,y,z,l3_t);
          lambda2_o->put(x,y,z,l2_t);
          lambda3_o->put(x,y,z,l1_t);
        }
        if( (l3_t >= l1_t) && (l1_t >= l2_t)){
          lambda1_o->put(x,y,z,l3_t);
          lambda2_o->put(x,y,z,l1_t);
          lambda3_o->put(x,y,z,l2_t);
        }
      }
    }
  }
  delete eign1;
  delete eign2;
  delete eign3;
  delete lambda1_o;
  delete lambda2_o;
  delete lambda3_o;
}



template <class T, class U>
vector< vector< int > > Cube<T,U>::decimate_layer
(int nLayer, float threshold, int window_xy, string filename)
{

  vector< vector < int > > toReturn;
  int cubeCardinality = cubeWidth*cubeHeight;
  bool* visitedPoints = (bool*)malloc(cubeCardinality*sizeof(bool));
  for(int i = 0; i < cubeCardinality; i++)
    visitedPoints[i] = false;

  //Creates a map from the values of the cube to its coordinates
  multimap< T, int > valueToCoords;
  printf("Cube<T,U>::decimate Creating the map[\n");

  T min_value = 255;
  T max_value = 0;
  for(int y = 0; y < cubeHeight; y++)
    for(int x = 0; x < cubeWidth; x++){
      if(at(x,y,nLayer) > max_value)
        max_value = at(x,y,nLayer);
      if(at(x,y,nLayer) < min_value)
        min_value = at(x,y,nLayer);
    }

  if(sizeof(T) == 1)
    printf("Layer %i with max_value = %u and min_value = %u\n", nLayer, max_value, min_value);
  else
    printf("Layer %i with max_value = %f and min_value = %f\n", nLayer, max_value, min_value);

  int position = 0;
  for(int y = window_xy*2; y < cubeHeight-window_xy*2; y++){
    for(int x = window_xy*2; x < cubeWidth-window_xy*2; x++)
      {
        position = x + y*cubeWidth;
        if( (this->at(x,y,nLayer) > threshold) &&
            (visitedPoints[position] == false))
          {
            valueToCoords.insert(pair<T, int >(this->at(x,y,nLayer), position));
          }
      }
  }


    typename multimap< T, int >::iterator iter = valueToCoords.begin();
    T min_value_it = (*iter).first;
    if(sizeof(T)==1)
      printf("\nCube<T,U>:: threshold: %u min_value = %u[", threshold, min_value_it);
    else
      printf("\nCube<T,U>:: threshold: %f min_value = %f[", threshold, min_value_it);
    fflush(stdout);

    typename multimap< T, int >::reverse_iterator riter = valueToCoords.rbegin();
    int pos;
    int counter = 0;
    int print_step = valueToCoords.size()/50;

    int x_p = 0;
    int y_p = 0;
    int z_p = 0;
    for( riter = valueToCoords.rbegin(); riter != valueToCoords.rend(); riter++)
      {
        counter++;
        if(counter%print_step == 0)
          printf("#");fflush(stdout);
        pos = (*riter).second;
        if(visitedPoints[pos] == true)
          continue;
        z_p = nLayer;
        y_p = (pos)/cubeWidth;
        x_p = pos - y_p*cubeWidth;
//         //       printf("%i %i %i %i \n", pos, x_p, y_p, z_p);
        for(int y = max(y_p-window_xy,0); y < min(y_p+window_xy, (int)cubeHeight); y++)
          for(int x = max(x_p-window_xy,0); x < min(x_p+window_xy, (int)cubeWidth); x++)
            visitedPoints[x + y*cubeWidth] = true;
        vector< int > coords(3);
        coords[0] = x_p;
        coords[1] = y_p;
        coords[2] = nLayer;
        toReturn.push_back(coords);
      }
    printf("] %i points \n", toReturn.size());


  if(filename!="")
    {
      printf("Cube<T,U>::decimating_layer : saving the points in %s\n", filename.c_str());
      std::ofstream out(filename.c_str());
      for(int i = 0; i < toReturn.size(); i++){
        out << toReturn[i][0] << " " << toReturn[i][1] << " " << toReturn[i][2] << std::endl ;
      }
      out.close();
    }
  return toReturn;
}




template <class T, class U>
vector< vector< int > > Cube<T,U>::decimate
(float threshold, int window_xy, int window_z,
 string filename, bool save_boosting_response)
{

  vector< vector < int > > toReturn;
  int cubeCardinality = cubeWidth*cubeHeight*cubeDepth;
  bool* visitedPoints = (bool*)malloc(cubeCardinality*sizeof(bool));
  for(int i = 0; i < cubeCardinality; i++)
    visitedPoints[i] = false;

  //Creates a map from the values of the cube to its coordinates
  multimap< T, int > valueToCoords;
  printf("Cube<T,U>::decimate Creating the map[\n");

  int min_layer = 0;
  int max_layer = (int)cubeDepth-1;
  for(int z = 0; z < min_layer; z++)
    for(int y = 0; y < cubeHeight; y++)
      for(int x = 0; x < cubeWidth; x++)
        visitedPoints[x + y*cubeWidth + z*cubeWidth*cubeHeight] = true;

  for(int z = max_layer+1; z < cubeDepth; z++)
    for(int y = 0; y < cubeHeight; y++)
      for(int x = 0; x < cubeWidth; x++)
        visitedPoints[x + y*cubeWidth + z*cubeWidth*cubeHeight] = true;

  float min_value, max_value;
  min_max(&min_value, &max_value);

  if(sizeof(T) == 1)
    printf("Cube with max_value = %u and min_value = %u\n", max_value, min_value);
  else
    printf("Cube with max_value = %f and min_value = %f\n", max_value, min_value);

  double step_size = (max_value - min_value) / 10;
  double current_threshold = max_value - step_size;

  int position;

  while(
        (current_threshold > min_value) &&
        (current_threshold > threshold - step_size)
        ){

    if( fabs(threshold - current_threshold) < step_size)
      current_threshold = threshold;

    valueToCoords.erase(valueToCoords.begin(), valueToCoords.end() );

    // Find the non-visited points above the threshold
    for(int z = min_layer; z <= max_layer; z++){
      for(int y = 0; y < cubeHeight; y++){
        for(int x = 0; x < cubeWidth; x++)
          {
            position = x + y*cubeWidth + z*cubeWidth*cubeHeight;
            if( (this->at(x,y,z) > current_threshold) &&
                (visitedPoints[position] == false))
              {
                valueToCoords.insert(pair<T, int >(this->at(x,y,z), position));
              }
          }
      }
      printf("Threshold: %f, Layer %02i and %07i points\r",
             current_threshold, z, valueToCoords.size()); fflush(stdout);
    }

    typename multimap< T, int >::iterator iter = valueToCoords.begin();
    T min_value_it = (*iter).first;
    // if(sizeof(T)==1)
      // printf("\nCube<T,U>:: threshold: %u min_value = %u[", current_threshold, min_value_it);
    // else
      // printf("\nCube<T,U>:: threshold: %f min_value = %f[", current_threshold, min_value_it);
    // fflush(stdout);

    typename multimap< T, int >::reverse_iterator riter = valueToCoords.rbegin();
    int pos;
    int counter = 0;
    int print_step = valueToCoords.size()/50;

    int x_p = 0;
    int y_p = 0;
    int z_p = 0;
    for( riter = valueToCoords.rbegin(); riter != valueToCoords.rend(); riter++)
      {
        counter++;
        pos = (*riter).second;
        if(visitedPoints[pos] == true)
          continue;
        z_p = pos / (cubeWidth*cubeHeight);
        y_p = (pos - z_p*cubeWidth*cubeHeight)/cubeWidth;
        x_p = pos - z_p*cubeWidth*cubeHeight - y_p*cubeWidth;

        for(int z = max(z_p-window_z,min_layer); z <= min(z_p+window_z, (int)max_layer); z++)
          for(int y = max(y_p-window_xy,0); y < min(y_p+window_xy, (int)cubeHeight); y++)
            for(int x = max(x_p-window_xy,0); x < min(x_p+window_xy, (int)cubeWidth); x++)
              visitedPoints[x + y*cubeWidth + z*cubeWidth*cubeHeight] = true;
        vector< int > coords(3);
        coords[0] = x_p;
        coords[1] = y_p;
        coords[2] = z_p;
        toReturn.push_back(coords);
      }
    current_threshold = current_threshold - step_size;
  }

  if(filename!="")
    {
      printf("Cube<T,U>::decimating : saving the points in %s\n", filename.c_str());
      std::ofstream out(filename.c_str());
      vector< float > coordsMicrom(3);
      for(int i = 0; i < toReturn.size(); i++){
        indexesToMicrometers(toReturn[i], coordsMicrom);
        out << coordsMicrom[0] << " " << coordsMicrom[1] << " " << coordsMicrom[2];
        if(save_boosting_response && (sizeof(T)==1) ){
          out << " " << (int)at(toReturn[i][0], toReturn[i][1], toReturn[i][2]);
        }
        if(save_boosting_response && (sizeof(T)==4) ){
          out << " " << at(toReturn[i][0], toReturn[i][1], toReturn[i][2]);
        }
        out << std::endl;
      }
      out.close();
    }
  return toReturn;

}

template <class T, class U>
vector< vector< int > > Cube<T,U>::decimate_log(float threshold, int window_xy, int window_z,  string filename, bool save_boosting_response)
{

  vector< vector < int > > toReturn;
  int cubeCardinality = cubeWidth*cubeHeight*cubeDepth;
  bool* visitedPoints = (bool*)malloc(cubeCardinality*sizeof(bool));
  for(int i = 0; i < cubeCardinality; i++)
    visitedPoints[i] = false;

  //Creates a map from the values of the cube to its coordinates
  multimap< T, int > valueToCoords;
  printf("Cube<T,U>::decimate Creating the map[\n");

  int min_layer = 10;
  int max_layer = 86;
  for(int z = 0; z < min_layer; z++)
    for(int y = 0; y < cubeHeight; y++)
      for(int x = 0; x < cubeWidth; x++)
        visitedPoints[x + y*cubeWidth + z*cubeWidth*cubeHeight] = true;

  for(int z = max_layer; z < cubeDepth; z++)
    for(int y = 0; y < cubeHeight; y++)
      for(int x = 0; x < cubeWidth; x++)
        visitedPoints[x + y*cubeWidth + z*cubeWidth*cubeHeight] = true;

  T min_value = 255;
  T max_value = 0;
  for(int z = min_layer; z < max_layer; z++)
    for(int y = 0; y < cubeHeight; y++)
      for(int x = 0; x < cubeWidth; x++){
        if(at(x,y,z) > max_value)
          max_value = at(x,y,z);
        if(at(x,y,z) < min_value)
          min_value = at(x,y,z);
      }

  if(sizeof(T) == 1)
    printf("Cube with max_value = %u and min_value = %u\n", max_value, min_value);
  else
    printf("Cube with max_value = %f and min_value = %f\n", max_value, min_value);

  double step_size = 0.1;
  double current_threshold = max_value/2;

  int position;

  while( (current_threshold > min_value) && (current_threshold > 1e-8) ){

    valueToCoords.erase(valueToCoords.begin(), valueToCoords.end() );

    for(int z = min_layer; z < max_layer; z++){
      for(int y = 20; y < cubeHeight-20; y++){
        for(int x = 20; x < cubeWidth-20; x++)
          {
            position = x + y*cubeWidth + z*cubeWidth*cubeHeight;
            if( (this->at(x,y,z) > current_threshold) &&
                (visitedPoints[position] == false))
              {
                valueToCoords.insert(pair<T, int >(this->at(x,y,z), position));
              }
          }
      }
      printf("iteration %02i and %07i points\r", z, valueToCoords.size()); fflush(stdout);
    }

    typename multimap< T, int >::iterator iter = valueToCoords.begin();
    T min_value_it = (*iter).first;
    if(sizeof(T)==1)
      printf("\nCube<T,U>:: threshold: %u min_value = %u[", current_threshold, min_value_it);
    else
      printf("\nCube<T,U>:: threshold: %f min_value = %f[", current_threshold, min_value_it);
    fflush(stdout);

    typename multimap< T, int >::reverse_iterator riter = valueToCoords.rbegin();
    int pos;
    int counter = 0;
    int print_step = valueToCoords.size()/50;

    int x_p = 0;
    int y_p = 0;
    int z_p = 0;
    for( riter = valueToCoords.rbegin(); riter != valueToCoords.rend(); riter++)
      {
        counter++;
//         if(counter%print_step == 0)
//           printf("#");fflush(stdout);
        pos = (*riter).second;
        if(visitedPoints[pos] == true)
          continue;
        z_p = pos / (cubeWidth*cubeHeight);
        y_p = (pos - z_p*cubeWidth*cubeHeight)/cubeWidth;
        x_p = pos - z_p*cubeWidth*cubeHeight - y_p*cubeWidth;
        //       printf("%i %i %i %i \n", pos, x_p, y_p, z_p);
        for(int z = max(z_p-window_z,min_layer); z < min(z_p+window_z, (int)max_layer); z++)
          for(int y = max(y_p-window_xy,0); y < min(y_p+window_xy, (int)cubeHeight); y++)
            for(int x = max(x_p-window_xy,0); x < min(x_p+window_xy, (int)cubeWidth); x++)
              visitedPoints[x + y*cubeWidth + z*cubeWidth*cubeHeight] = true;
        vector< int > coords(3);
        coords[0] = x_p;
        coords[1] = y_p;
        coords[2] = z_p;
        toReturn.push_back(coords);
      }
    printf("] %i \n", toReturn.size());
    current_threshold = current_threshold*step_size;
  }

  if(filename!="")
    {
      printf("Cube<T,U>::decimating : saving the points in %s\n", filename.c_str());
      std::ofstream out(filename.c_str());
      for(int i = 0; i < toReturn.size(); i++){
        out << toReturn[i][0] << " " << toReturn[i][1] << " " << toReturn[i][2] ;
        if(save_boosting_response && (sizeof(T)==1) ){
          out << " " << (int)at(toReturn[i][0], toReturn[i][1], toReturn[i][2]);
        }
        if(save_boosting_response && (sizeof(T)==4) ){
          out << " " << at(toReturn[i][0], toReturn[i][1], toReturn[i][2]);
        }
        out << std::endl;
      }
      out.close();
    }
  return toReturn;

}

template <class T, class U>
void Cube<T,U>::norm_cube(Cube<float,double>* c1, Cube<float,double>* c2, Cube<float,double>* output)
{
  printf("Cube<T,U>::norm_cube[");
  for(int z = 0; z < output->cubeDepth; z++){
    for(int y = 0; y < output->cubeHeight; y++){
      for(int x = 0; x < output->cubeWidth; x++){
        output->put(x,y,z,
                    sqrt(c1->at(x,y,z)*c1->at(x,y,z) + c2->at(x,y,z)*c2->at(x,y,z)));
      }
    }
    printf("#");fflush(stdout);
  }
  printf("]\n");
}

template <class T, class U>
void Cube<T,U>::norm_cube(string volume_nfo, string volume_1, string volume_2, string volume_3, string volume_output)
{
  Cube<float,double>* output = new Cube<float,double>(volume_nfo, volume_output);
  Cube<float,double>* v1 = new Cube<float,double>(volume_nfo, volume_1);
  Cube<float,double>* v2 = new Cube<float,double>(volume_nfo, volume_2);
  Cube<float,double>* v3 = new Cube<float,double>(volume_nfo, volume_3);

  printf("Cube<T,U>::norm_cube[");
  for(int z = 0; z < output->cubeDepth; z++){
    for(int y = 0; y < output->cubeHeight; y++){
      for(int x = 0; x < output->cubeWidth; x++){
        output->put(x,y,z,
                    sqrt(v1->at(x,y,z)*v1->at(x,y,z) +
                         v2->at(x,y,z)*v2->at(x,y,z) +
                         v3->at(x,y,z)*v3->at(x,y,z) )
                     );
      }
    }
    printf("#");fflush(stdout);
  }
  printf("]\n");
  delete v1;
  delete v2;
  delete v3;
  delete output;
}

template <class T, class U>
void Cube<T,U>::get_ROC_curve
(string volume_nfo,
 string volume_positive,
 string volume_negative,
 string output_file,
 int nPoints
)
{
//   Cube<float,double>* positive = new Cube<float,double>(volume_nfo, volume_positive);
//   Cube<float,double>* negative = new Cube<float,double>(volume_nfo, volume_negative);

  Cube<uchar,ulong>* positive = new Cube<uchar,ulong>(volume_nfo, volume_positive);
  Cube<uchar,ulong>* negative = new Cube<uchar,ulong>(volume_nfo, volume_negative);


  float min = 1e6;
  float max = -1e6;
  int nPositivePoints = 0;
  int nNegativePoints = 0;

  printf("Cube<T,U>::getROC_curve: "); fflush(stdout);
  for(int z = 0; z < positive->cubeDepth; z++){
    for(int y = 0; y < positive->cubeHeight; y++){
      for(int x = 0; x < positive->cubeWidth; x++)
        {
          if(positive->at(x,y,z)!=0){
            if(positive->at(x,y,z)<min)
              min = positive->at(x,y,z);
            if(positive->at(x,y,z)>max)
              max = positive->at(x,y,z);
            nPositivePoints++;
          }
          if(negative->at(x,y,z)!=0){
            if(negative->at(x,y,z)<min)
              min = negative->at(x,y,z);
            if(negative->at(x,y,z)>max)
              max = negative->at(x,y,z);
            nNegativePoints++;
          }
        }
    }
  }
  printf("max:%f min:%f nPos:%i nNeg:%i [", max, min, nPositivePoints, nNegativePoints);
  fflush(stdout);
  std::ofstream out(output_file.c_str());

  float tp = 0;
  float fp = 0;

  for(float threshold = min; threshold < max; threshold += (max-min)/nPoints)
    {
      tp = 0;
      fp = 0;
      for(int z = 0; z < positive->cubeDepth; z++){
        for(int y = 0; y < positive->cubeHeight; y++){
          for(int x = 0; x < positive->cubeWidth; x++){
            if( (positive->at(x,y,z) != 0) &&
                (positive->at(x,y,z) > threshold) )
              tp++;
            if( (negative->at(x,y,z) != 0) &&
                (negative->at(x,y,z) > threshold) )
              fp++;
          }
        }
      }
      out << fp/nNegativePoints << " " << tp/nPositivePoints << std::endl;
      printf("#"); fflush(stdout);
    }
  printf("]\n");

  out.close();
  delete positive;
  delete negative;
}

template <class T, class U>
void Cube<T,U>::render_cylinder(vector<int> idx1, vector<int> idx2, float radius_micrometers)
{
  // First is to calculate a window where the cylinder will be drawn.
  int window_x = (int)(radius_micrometers/voxelWidth);
  int window_y = (int)(radius_micrometers/voxelHeight);
  int window_z = (int)(radius_micrometers/voxelDepth);

  int x_min = min(idx1[0],idx2[0]);
  int y_min = min(idx1[1],idx2[1]);
  int z_min = min(idx1[2],idx2[2]);
  int x_max = max(idx1[0],idx2[0]);
  int y_max = max(idx1[1],idx2[1]);
  int z_max = max(idx1[2],idx2[2]);

  vector<int> p2p1(3);
  p2p1[0] = idx2[0] - idx1[0];
  p2p1[1] = idx2[1] - idx1[1];
  p2p1[2] = idx2[2] - idx1[2];
  float p2p1mod =
    p2p1[0]*p2p1[0] +
    p2p1[1]*p2p1[1] +
    p2p1[2]*p2p1[2];

  vector<int> p1p0(3);

  float d;
  int p1p0mod;
  int dot_p1p0p2p1;
  float r2 = radius_micrometers*radius_micrometers;

  printf("Generating edge [");

  for(int z = max(0,z_min-window_z);
      z < min((int)cubeDepth, z_max + window_z);
      z++){
      for(int y = max(0,y_min-window_y);
          y < min((int)cubeHeight, y_max + window_y);
          y++){
        for(int x = max(0,x_min-window_x);
            x < min((int)cubeWidth, x_max + window_x);
            x++){

          p1p0[0] = idx1[0]-x;
          p1p0[1] = idx1[1]-y;
          p1p0[2] = idx1[2]-z;

          p1p0mod = p1p0[0]*p1p0[0] +
            p1p0[1]*p1p0[1] +
            p1p0[2]*p1p0[2];

          dot_p1p0p2p1 = p1p0[0]*p2p1[0] + p1p0[1]*p2p1[1] + p1p0[2]*p2p1[2];

          d = float(p1p0mod*p2p1mod - dot_p1p0p2p1*dot_p1p0p2p1)/p2p1mod;

          if(d < r2)
            this->put(x,y,z,(T)0);
        }
      }
      printf("#");fflush(stdout);
  }
  printf("]\n");
}


template <class T, class U>
void Cube<T,U>::min_max(float* min, float* max)
{
  *min = 1e6;
  *max = -1e6;

  printf("%s calculating min and max [", filenameParameters.c_str());
  for(int z = 0; z < cubeDepth; z++){
    for(int y = 0; y < cubeHeight; y++){
      for(int x = 0; x < cubeWidth; x++){
        if( at(x,y,z) > *max)
          *max = at(x,y,z);
        if( (at(x,y,z) < *min) )
          *min = at(x,y,z);
      }
    }
    printf("#");fflush(stdout);
  }
  printf("]\n");
}


template <class T, class U>
Cube_P*  Cube<T,U>::threshold
(float thres, string outputName,
bool putHigherValuesTo, bool putLowerValuesTo,
float highValue, float lowValue)
{
  Cube<T,U>* result = duplicate_clean(outputName);

  #pragma omp parallel for
  for(int z = 0; z < cubeDepth; z++)
    for(int y = 0; y < cubeHeight; y++)
      for(int x = 0; x < cubeWidth; x++)
        if (at(x,y,z) > thres)
          if(putHigherValuesTo)
            result->put(x,y,z, highValue);
          else
            result->put(x,y,z, at(x,y,z));
        else
          if(putLowerValuesTo)
            result->put(x,y,z, lowValue);
          else
            result->put(x,y,z,  at(x,y,z));

  return (Cube_P*)result;

}
