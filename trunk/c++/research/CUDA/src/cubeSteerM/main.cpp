 
 /*
 * This sample implements a separable convolution filter 
 * of a 2D image with an arbitrary kernel.
 */



#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cutil_inline.h>
#include "Image.h"
#include "Cube.h"
#include "Mask.h"

// Torch3 files



#include "GPUfunctions.h"
extern "C" void rtfSVM
(float*  d_output,
 float*  d_theta,
 float*  d_phi,
 float*  d_sv,
 float*  d_alphas,
 float   sk,
 int     n_sv,
 int     imageW,
 int     imageH,
 int     imageD
);

////////////////////////////////////////////////////////////////////////////////
// Torch3 SVM output file extraction. Kansal code
////////////////////////////////////////////////////////////////////////////////
void get_svm_features
(string svm_file,
 int& n_support_vectors,
 int& no_dimensions,
 float **alphas,
 float **support_vec)
{
  // printf("Init read of %s\n", svm_file.c_str());
  int tag_size,block_size,n_blocks;
  int n_support_vectors_bound;
  char tag[10];
  float b;
  FILE *f=fopen(svm_file.c_str(),"rb");
  if(!f){
    cout<<"out.svm doest not exist"<<endl;
    exit(0);
  }
  fread( &tag_size, sizeof(int), 1, f );
  fread( tag, sizeof(char), tag_size, f );
  fread( &block_size, sizeof(int), 1, f );
  fread( &n_blocks, sizeof(int), 1, f );
  fread( &b, sizeof(float), 1, f );
  // printf("b is %f\n", b);

  fread( &tag_size, sizeof(int), 1, f );
  fread( tag, sizeof(char), tag_size, f );
  fread( &block_size, sizeof(int), 1, f );
  fread( &n_blocks, sizeof(int), 1, f );
  fread( &n_support_vectors, sizeof(int), 1, f );
  // printf("n_support_vectors is %i\n", n_support_vectors);

  fread( &tag_size, sizeof(int), 1, f );
  fread( tag, sizeof(char), tag_size, f );
  fread( &block_size, sizeof(int), 1, f );
  fread( &n_blocks, sizeof(int), 1, f );
  fread( &n_support_vectors_bound, sizeof(int), 1, f );

  fread( &tag_size, sizeof(int), 1, f );
  fread( tag, sizeof(char), tag_size, f );
  fread( &block_size, sizeof(int), 1, f );
  fread( &n_blocks, sizeof(int), 1, f );

  *alphas=new float[n_support_vectors];
  for(int n=0;n<n_support_vectors;n++){
    fread( &((*alphas)[n]), sizeof(float), 1, f );
  }
  fread( &tag_size, sizeof(int), 1, f );
  fread( tag, sizeof(char), tag_size, f );
  fread( &block_size, sizeof(int), 1, f );
  fread( &n_blocks, sizeof(int), 1, f );
  fread( &n_support_vectors_bound, sizeof(int), 1, f );
  fread( &tag_size, sizeof(int), 1, f );
  fread( tag, sizeof(char), tag_size, f );
  fread( &block_size, sizeof(int), 1, f );
  fread( &n_blocks, sizeof(int), 1, f );
  fread( &no_dimensions, sizeof(int), 1, f );
  printf("n_dimensions is %i\n", no_dimensions);

  *support_vec=new float[(n_support_vectors)*(no_dimensions)];
  for(int i=0;i<n_support_vectors;i++){
    fread( &tag_size, sizeof(int), 1, f );
    fread( tag, sizeof(char), tag_size, f );
    fread( &block_size, sizeof(int), 1, f );
    fread( &n_blocks, sizeof(int), 1, f );
    fread( &n_support_vectors_bound, sizeof(int), 1, f );
    fread( &tag_size, sizeof(int), 1, f );
    fread( tag, sizeof(char), tag_size, f );
    fread( &block_size, sizeof(int), 1, f );
    fread( &n_blocks, sizeof(int), 1, f );
    for(int j=0;j<no_dimensions;j++){
      fread( &((*(support_vec))[i*(no_dimensions) + j]), sizeof(float), 1, f );
    }
  }
}

void printSupportVectors
(float* alphas, float* suppotVectors, int nVectors, int nDimensions)
{
  for(int i = 0; i < nVectors; i++){
    printf("%i: %f ", i, alphas[i]);;
    for(int j = 0; j < nDimensions; j++){
      printf("%f ", suppotVectors[i*nDimensions + j]);
    }
    printf("\n");
  }

}

/*****
//Order of the derivatives
Abs   Ret  Meaning
0     0    -> xx
1     1    -> xy
2     2    -> xz
3     3    -> yy
4     4    -> yz
5     5    -> zz
6     0    -> xxxx
7     1    -> xxxy
8     2    -> xxxz
9     3    -> xxyy
10    4    -> xxyz
11    5    -> xxzz
12    6    -> xyyy
13    7    -> yyyy
14    8    -> yyyz
15    9    -> xyyz
16    10   -> yyzz
17    11   -> xzzz
18    12   -> yzzz
19    13   -> zzzz
20    14   -> xyzz
****/

extern "C" void set_derivatives_pointer(float **h_deriv);

void compute_derivatives
(float** h_deriv,
 float*  d_Input,
 float   sigma_images,
 float*  d_buffer,
 int sizeX,
 int sizeY,
 int sizeZ
)
{
  vector<float> kernel_0 = Mask::gaussian_mask(0, sigma_images, 1);
  vector<float> kernel_1 = Mask::gaussian_mask(1, sigma_images, 1);
  vector<float> kernel_2 = Mask::gaussian_mask(2, sigma_images, 1);
  vector<float> kernel_3 = Mask::gaussian_mask(3, sigma_images, 1);
  vector<float> kernel_4 = Mask::gaussian_mask(4, sigma_images, 1);

  convolution_separable( h_deriv[0 ], d_Input, kernel_2, kernel_0, kernel_0,
                         sizeX, sizeY, sizeZ, d_buffer );
  convolution_separable( h_deriv[1 ], d_Input, kernel_1, kernel_1, kernel_0,
                         sizeX, sizeY, sizeZ, d_buffer );
  convolution_separable( h_deriv[2 ], d_Input, kernel_1, kernel_0, kernel_1,
                         sizeX, sizeY, sizeZ, d_buffer );
  convolution_separable( h_deriv[3 ], d_Input, kernel_0, kernel_2, kernel_0,
                         sizeX, sizeY, sizeZ, d_buffer );
  convolution_separable( h_deriv[4 ], d_Input, kernel_0, kernel_1, kernel_1,
                         sizeX, sizeY, sizeZ, d_buffer );
  convolution_separable( h_deriv[5 ], d_Input, kernel_0, kernel_0, kernel_2,
                         sizeX, sizeY, sizeZ, d_buffer );
  convolution_separable( h_deriv[6 ], d_Input, kernel_4, kernel_0, kernel_0,
                         sizeX, sizeY, sizeZ, d_buffer );
  convolution_separable( h_deriv[7 ], d_Input, kernel_3, kernel_1, kernel_0,
                         sizeX, sizeY, sizeZ, d_buffer );
  convolution_separable( h_deriv[8 ], d_Input, kernel_3, kernel_0, kernel_1,
                         sizeX, sizeY, sizeZ, d_buffer );
  convolution_separable( h_deriv[9 ], d_Input, kernel_2, kernel_2, kernel_0,
                         sizeX, sizeY, sizeZ, d_buffer );
  convolution_separable( h_deriv[10], d_Input, kernel_2, kernel_1, kernel_1,
                         sizeX, sizeY, sizeZ, d_buffer );
  convolution_separable( h_deriv[11], d_Input, kernel_2, kernel_0, kernel_2,
                         sizeX, sizeY, sizeZ, d_buffer );
  convolution_separable( h_deriv[12], d_Input, kernel_1, kernel_3, kernel_0,
                         sizeX, sizeY, sizeZ, d_buffer );
  convolution_separable( h_deriv[13], d_Input, kernel_0, kernel_4, kernel_0,
                         sizeX, sizeY, sizeZ, d_buffer );
  convolution_separable( h_deriv[14], d_Input, kernel_0, kernel_3, kernel_1,
                         sizeX, sizeY, sizeZ, d_buffer );
  convolution_separable( h_deriv[15], d_Input, kernel_1, kernel_2, kernel_1,
                         sizeX, sizeY, sizeZ, d_buffer );
  convolution_separable( h_deriv[16], d_Input, kernel_0, kernel_2, kernel_2,
                         sizeX, sizeY, sizeZ, d_buffer );
  convolution_separable( h_deriv[17], d_Input, kernel_1, kernel_0, kernel_3,
                         sizeX, sizeY, sizeZ, d_buffer );
  convolution_separable( h_deriv[18], d_Input, kernel_0, kernel_1, kernel_3,
                         sizeX, sizeY, sizeZ, d_buffer );
  convolution_separable( h_deriv[19], d_Input, kernel_0, kernel_0, kernel_4,
                         sizeX, sizeY, sizeZ, d_buffer );
  convolution_separable( h_deriv[20], d_Input, kernel_1, kernel_1, kernel_2,
                         sizeX, sizeY, sizeZ, d_buffer );

  // and now we need to pass the pointers to the constant memmory of the GPU
  set_derivatives_pointer(h_deriv);
}


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){

  if(argc!=6){
    printf("Usage: cubeSteerM volume.nfo sigma_images svm.svm sigma_svm_kernel out\n");
    exit(0);
  }

  Cube<uchar, ulong>*  cube = new Cube<uchar, ulong>(argv[1]);
  float sigma_images = atof(argv[2]);
  string svm_file(argv[3]);
  float sigma_kernel = atof(argv[4]);
  string outName(argv[5]);

  Cube<float, double>* res = cube->create_blank_cube(outName);

  // strips the svm file
  int n_sv;
  int n_dim;
  float *h_alphas, *h_sv;
  get_svm_features
    (svm_file, n_sv, n_dim, &h_alphas, &h_sv);


  float
    *h_Input,
    *h_OutputGPU,
    **h_deriv;

  //The derivatives are stored in one vector of vectors of derivatives.
  float
    *d_Input,
    *d_Output,
    *d_theta,
    *d_phi,
    *d_sv,
    *d_alphas
    ;

  printf("Initializing CUDA\n");
  unsigned int hTimer;
  cudaSetDevice( cutGetMaxGflopsDeviceId() );
  cutilCheckError(cutCreateTimer(&hTimer));

  int imageW = cube->cubeWidth;
  int imageH = cube->cubeHeight;
  int imageD = cube->cubeDepth;
  // const int maxTileSizeX = 128;
  // const int maxTileSizeY = 128;

  const int maxTileSizeX = 128;
  const int maxTileSizeY = 128;
  const int maxTileSizeZ = 64;

  int  maxLinearSize = maxTileSizeX * maxTileSizeY * maxTileSizeZ;


  printf("Allocating and intializing host arrays...\n");
  h_Input     = (float *)malloc( maxLinearSize * sizeof(float));
  h_OutputGPU = (float *)malloc( maxLinearSize * sizeof(float));
  h_deriv     = (float **)malloc(21 * sizeof(float*));
  srand(200);

  printf("Allocating CUDA arrays...\n");
  cutilSafeCall( cudaMalloc((void **)&d_Input,        maxLinearSize * sizeof(float)) );
  cutilSafeCall( cudaMalloc((void **)&d_Output,       maxLinearSize * sizeof(float)) );
  cutilSafeCall( cudaMalloc((void **)&d_theta, maxLinearSize * sizeof(float)) );
  cutilSafeCall( cudaMalloc((void **)&d_phi,   maxLinearSize * sizeof(float)) );
  for(int i = 0; i < 21; i++)
    cutilSafeCall( cudaMalloc((void **)&h_deriv[i],   maxLinearSize * sizeof(float)) );
  cutilSafeCall( cudaMalloc((void **)&d_sv,  n_sv * n_dim * sizeof(float)) );
  cutilSafeCall( cudaMalloc((void **)&d_alphas,  n_sv * sizeof(float)) );

  // copy the values of the svm to the memmory of the device
  cutilSafeCall( cudaMemcpy(d_sv, h_sv,
                            n_dim * n_sv * sizeof(float),
                            cudaMemcpyHostToDevice) );
  cutilSafeCall( cudaMemcpy(d_alphas, h_alphas,
                             n_sv * sizeof(float),
                            cudaMemcpyHostToDevice) );


  // // Here should come the loop
  // // Variables required to split the image into tiles
  vector< float > kernelSample = Mask::gaussian_mask(0, sigma_images, 1);
  int pad_x = ceil(kernelSample.size()/2);
  int pad_y = ceil(kernelSample.size()/2);
  int pad_z = ceil(kernelSample.size()/2);
  int tile_size_x  = maxTileSizeX - 2*pad_x;
  int tile_size_y  = maxTileSizeY - 2*pad_y;
  int tile_size_z  = maxTileSizeZ - 2*pad_z;
  int n_tiles_horiz = ceil(float(imageW) /tile_size_x);
  int n_tiles_vert  = ceil(float(imageH) /tile_size_y);
  int n_tiles_depth = ceil(float(imageD) /tile_size_z);

  printf("Sigma = %f, kernel size: %i, Tiles pads be: [%i,%i,%i]\n",
         sigma_images, kernelSample.size(), pad_x, pad_y, pad_z);

  printf("Tiles sizes be: [%i,%i,%i]\n",
         tile_size_x, tile_size_y, tile_size_z);


  printf("The number of tiles should be: [%i,%i,%i]\n",
         n_tiles_horiz, n_tiles_vert, n_tiles_depth);

  // for the tiles in horizontal
  // for(int tz = 0; tz < n_tiles_depth; tz++){
    // for(int ty = 0; ty < n_tiles_vert; ty++){
      // for(int tx = 0; tx < n_tiles_horiz; tx++){

  for(int tz = 0; tz < 1; tz++){
    for(int ty = 0; ty < 1; ty++){
      // for(int tx = 0; tx < n_tiles_horiz; tx++){
      for(int tx = 0; tx < 1; tx++){

        int x0, y0, z0, x1, y1, z1;
        x0 = tx*tile_size_x;
        y0 = ty*tile_size_y;
        z0 = tz*tile_size_z;
        x1 = min((tx+1)*tile_size_x -1, imageW-1);
        y1 = min((ty+1)*tile_size_y -1, imageH-1);
        z1 = min((tz+1)*tile_size_z  -1, cube->cubeDepth -1);
        printf(" iteration [%i,%i,%i], pad= [%i,%i,%i]->[%i,%i,%i]\n",
               tx, ty, tz, x0, y0, z0, x1, y1, z1);

        Cube<uchar, ulong>* padded =
          cube->get_padded_tile(x0, y0, z0, x1, y1, z1, pad_x, pad_y, pad_z);

        // puts the padded image into the array where the convolutions are going to be done
        for(int z = 0; z < padded->cubeDepth; z++)
          for(int y = 0; y < padded->cubeHeight; y++)
            for(int x = 0; x < padded->cubeWidth; x++)
              h_Input[(z*maxTileSizeX + y)*maxTileSizeY + x] = padded->at(x,y,z);

        // cutilSafeCall( cudaMemcpy(d_Input, h_Input,
                                  // maxLinearSize * sizeof(float),
                                  // cudaMemcpyHostToDevice) );
        cudaMemcpy(d_Input, h_Input,
                   maxLinearSize * sizeof(float),
                   cudaMemcpyHostToDevice);


        // Computes the set of 21 derivatives
        compute_derivatives(h_deriv, d_Input, sigma_images, d_Output,
                            maxTileSizeX, maxTileSizeY, maxTileSizeZ);
        // cutilSafeCall( cudaThreadSynchronize() );
        if(cudaSuccess != cudaThreadSynchronize()){
          printf("Error computing the derivatives\n");
        }

        hessian_orientation
          (d_Output, d_theta, d_phi,
           d_Input, sigma_images,
           h_deriv[0], h_deriv[1], h_deriv[2], h_deriv[3], h_deriv[4], h_deriv[5],
           maxTileSizeX, maxTileSizeY, maxTileSizeZ);
        // cutilSafeCall( cudaThreadSynchronize() );
        cudaError err = cudaThreadSynchronize();
        if(cudaSuccess != err){
          printf("Error computing the orienatations: %s\n", cudaGetErrorString(err));
        }

        // And now the big stuff is done
        rtfSVM
          (d_Output,
           d_theta, d_phi,
           d_sv, d_alphas, sigma_kernel, n_sv,
           maxTileSizeX, maxTileSizeY, maxTileSizeZ);
        // cutilSafeCall( cudaThreadSynchronize() );
        err = cudaThreadSynchronize();
        if(cudaSuccess != err){
          printf("Error computing the SVM: %s\n", cudaGetErrorString(err));
        }

        // cudaThreadSynchronize();

        // cutilSafeCall( cudaMemcpy(h_OutputGPU, d_Output,
                                  // maxLinearSize * sizeof(float),
                                  // cudaMemcpyDeviceToHost) );

        cudaMemcpy(h_OutputGPU, d_Output,
                   maxLinearSize * sizeof(float),
                   cudaMemcpyDeviceToHost);



        for(int z = pad_z; z < padded->cubeDepth-pad_z; z++)
          for(int y = pad_y; y < padded->cubeHeight-pad_y; y++)
            for(int x = pad_x; x < padded->cubeWidth-pad_x; x++){
              res->put(x0+x-pad_x, y0+y-pad_y, z0+z-pad_z,
                     h_OutputGPU[(z*maxTileSizeY + y)*maxTileSizeX + x]);

            }

        delete padded;
      }
    }
  }
  // printf("Done with the computations\n");


  printf("Shutting down...\n");
  free(h_OutputGPU);
  free(h_Input);

  printf("Freeing CUDA arrays...\n");
  cudaFree(d_Input) ;
  cudaFree(d_Output) ;
  cudaFree(d_theta) ;
  cudaFree(d_phi) ;
  for(int i = 0; i < 21; i++)
    cudaFree(h_deriv[i] ) ;
  cudaFree(d_alphas ) ;
  cudaFree(d_sv ) ;

  // cutilSafeCall( cudaFree(d_Input) );
  // cutilSafeCall( cudaFree(d_Output) );
  // cutilSafeCall( cudaFree(d_theta) );
  // cutilSafeCall( cudaFree(d_phi) );
  // for(int i = 0; i < 21; i++)
    // cutilSafeCall( cudaFree(h_deriv[i] ) );
  // cutilSafeCall( cudaFree(d_alphas ) );
  // cutilSafeCall( cudaFree(d_sv ) );

  cutilCheckError(cutDeleteTimer(hTimer));

  cudaThreadExit();

  exit(0);
}

