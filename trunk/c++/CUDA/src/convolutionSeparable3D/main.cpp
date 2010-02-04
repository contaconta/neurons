/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation and 
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 * 
 */
 
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

// extern "C" void set_horizontal_kernel(vector<float>& kernel);
// extern "C" void set_vertical_kernel(vector<float>& kernel);

extern "C" void setConvolutionKernel_horizontal(float *h_Kernel, int kernel_length);

extern "C" void setConvolutionKernel_vertical(float *h_Kernel, int kernel_length);

extern "C" void setConvolutionKernel_depth(float *h_Kernel, int kernel_length);

extern "C" void convolutionRowsGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int imageD,
    int kernel_radiusw
);

extern "C" void convolutionColumnsGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int imageD,
    int kernel_radius
);

extern "C" void convolutionDepthGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int imageD,
    int kernel_radius
);


extern "C" void hessianGPU
(
 float *d_output,
 float *d_gxx,
 float *d_gxy,
 float *d_gyy,
 int imageW,
 int imageH
 );


extern "C" void set_horizontal_kernel(vector<float>& kernel){
  float h_kernel_h[kernel.size()];
  for(unsigned int i = 0; i < kernel.size(); i++)
    h_kernel_h[i] = kernel[i];
  setConvolutionKernel_horizontal(h_kernel_h, kernel.size());
}

extern "C" void set_vertical_kernel(vector<float>& kernel){
  float h_kernel_h[kernel.size()];
  for(unsigned int i = 0; i < kernel.size(); i++)
    h_kernel_h[i] = kernel[i];
  setConvolutionKernel_vertical(h_kernel_h, kernel.size());
}

extern "C" void set_depth_kernel(vector<float>& kernel){
  float h_kernel_h[kernel.size()];
  for(unsigned int i = 0; i < kernel.size(); i++)
    h_kernel_h[i] = kernel[i];
  setConvolutionKernel_depth(h_kernel_h, kernel.size());
}


extern "C" void convolution_separable
( float* d_Dst,
  float* d_Src,
  vector< float >& kernel_h,
  vector< float >& kernel_v,
  vector< float >& kernel_d,
  int sizeX,
  int sizeY,
  int sizeZ,
  float* d_tmp
  )
{
  set_horizontal_kernel(kernel_h);
  set_vertical_kernel  (kernel_v);
  set_depth_kernel     (kernel_d);

  cutilSafeCall( cudaThreadSynchronize() );
  convolutionRowsGPU(d_Dst,
                     d_Src,
                     sizeX,
                     sizeY,
                     sizeZ,
                     floor(kernel_h.size()/2)
                     );

  convolutionColumnsGPU(
                        d_tmp,
                        d_Dst,
                        sizeX,
                        sizeY,
                        sizeZ,
                        floor(kernel_v.size()/2)
                        );

  convolutionDepthGPU(
                        d_Dst,
                        d_tmp,
                        sizeX,
                        sizeY,
                        sizeZ,
                        floor(kernel_d.size()/2)
                        );

  cutilSafeCall( cudaThreadSynchronize() );
}


extern "C" void hessian
( float* d_Buffer,
  float* d_Input,
  float sigma,
  int sizeX,
  int sizeY,
  int sizeZ
  )
{
  float *d_gxx, *d_gxy, *d_gxz, *d_gyy, *d_gyz, *d_gzz;

  cutilSafeCall( cudaMalloc((void **)&d_gxx,   sizeZ * sizeX * sizeY * sizeof(float)) );
  cutilSafeCall( cudaMalloc((void **)&d_gxy,   sizeZ * sizeX * sizeY * sizeof(float)) );
  cutilSafeCall( cudaMalloc((void **)&d_gxz,   sizeZ * sizeX * sizeY * sizeof(float)) );
  cutilSafeCall( cudaMalloc((void **)&d_gyy,   sizeZ * sizeX * sizeY * sizeof(float)) );
  cutilSafeCall( cudaMalloc((void **)&d_gyz,   sizeZ * sizeX * sizeY * sizeof(float)) );
  cutilSafeCall( cudaMalloc((void **)&d_gzz,   sizeZ * sizeX * sizeY * sizeof(float)) );

  vector<float> kernel_0 = Mask::gaussian_mask(0, sigma, 1);
  vector<float> kernel_1 = Mask::gaussian_mask(1, sigma, 1);
  vector<float> kernel_2 = Mask::gaussian_mask(2, sigma, 1);

  printf("  ->computing convolutions\n");
  convolution_separable( d_gxx, d_Input, kernel_2, kernel_0, kernel_0,
                         sizeX, sizeY, sizeZ, d_Buffer );
  convolution_separable( d_gxy, d_Input, kernel_1, kernel_1, kernel_0,
                         sizeX, sizeY, sizeZ, d_Buffer );
  convolution_separable( d_gxz, d_Input, kernel_1, kernel_0, kernel_1,
                         sizeX, sizeY, sizeZ, d_Buffer );
  convolution_separable( d_gyy, d_Input, kernel_0, kernel_2, kernel_0,
                         sizeX, sizeY, sizeZ, d_Buffer );
  convolution_separable( d_gyz, d_Input, kernel_0, kernel_1, kernel_1,
                         sizeX, sizeY, sizeZ, d_Buffer );
  convolution_separable( d_gzz, d_Input, kernel_0, kernel_0, kernel_2,
                         sizeX, sizeY, sizeZ, d_Buffer );


  printf("  ->computing the hessian\n");
  hessianGPU(d_Buffer, d_gxx, d_gxy, d_gxy, d_gyy, d_gyz, d_gzz, sizeX, sizeY, sizeZ);

  cutilSafeCall( cudaFree(d_gxx ) );
  cutilSafeCall( cudaFree(d_gxy ) );
  cutilSafeCall( cudaFree(d_gxz ) );
  cutilSafeCall( cudaFree(d_gyy ) );
  cutilSafeCall( cudaFree(d_gyz ) );
  cutilSafeCall( cudaFree(d_gzz ) );

}



////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){
  float
    *h_Input,
    *h_Buffer,
    *h_OutputGPU;

  float
    *d_Input,
    *d_Output,
    *d_Buffer;

  printf("Initializing CUDA\n");
  unsigned int hTimer;
  cudaSetDevice( cutGetMaxGflopsDeviceId() );
  cutilCheckError(cutCreateTimer(&hTimer));


  printf("Initializing image and result image\n");
  Cube<uchar, ulong>*  cube = new Cube<uchar, ulong>(argv[1]);
  float sigma = atof(argv[2]);
  Cube<float, double>* res = cube->create_blank_cube(argv[3]);
  int imageW = cube->cubeWidth;
  int imageH = cube->cubeHeight;
  int imageD = cube->cubeDepth;
  const int maxTileSizeX = 128;
  const int maxTileSizeY = 128;
  const int maxTileSizeZ = 64;
  int  maxLinearSize = maxTileSizeX * maxTileSizeY * maxTileSizeZ;

  printf("Allocating and intializing host arrays...\n");
  h_Input     = (float *)malloc( maxLinearSize * sizeof(float));
  h_Buffer    = (float *)malloc( maxLinearSize * sizeof(float));
  h_OutputGPU = (float *)malloc( maxLinearSize * sizeof(float));
  srand(200);

  printf("Allocating CUDA arrays...\n");
  cutilSafeCall( cudaMalloc((void **)&d_Input, maxLinearSize * sizeof(float)) );
  cutilSafeCall( cudaMalloc((void **)&d_Output,maxLinearSize * sizeof(float)) );
  cutilSafeCall( cudaMalloc((void **)&d_Buffer,maxLinearSize * sizeof(float)) );

  // Here should come the loop
  // Variables required to split the image into tiles
  vector< float > kernelSample = Mask::gaussian_mask(0, sigma, 1);
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
         sigma, kernelSample.size(), pad_x, pad_y, pad_z);

  printf("Tiles sizes be: [%i,%i,%i]\n",
         tile_size_x, tile_size_y, tile_size_z);


  printf("The number of tiles should be: [%i,%i,%i]\n",
         n_tiles_horiz, n_tiles_vert, n_tiles_depth);

  vector<float> kernel_0 = Mask::gaussian_mask(0, sigma, 1);
  vector<float> kernel_1 = Mask::gaussian_mask(1, sigma, 1);
  vector<float> kernel_2 = Mask::gaussian_mask(2, sigma, 1);

  // for the tiles in horizontal
  for(int tz = 0; tz < n_tiles_depth; tz++){
    for(int ty = 0; ty < n_tiles_vert; ty++){
      for(int tx = 0; tx < n_tiles_horiz; tx++){

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

        cutilSafeCall( cudaMemcpy(d_Input, h_Input,
                                  maxLinearSize * sizeof(float),
                                  cudaMemcpyHostToDevice) );

        // convolution_separable( d_Output, d_Input, kernel_1, kernel_1, kernel_0,
                               // maxTileSizeX, maxTileSizeY, maxTileSizeZ, d_Buffer );

        hessian(d_Buffer, d_Input, sigma, maxTileSizeX, maxTileSizeY, maxTileSizeZ);


        cutilSafeCall( cudaThreadSynchronize() );

        cutilSafeCall( cudaMemcpy(h_OutputGPU, d_Output,
                                  maxLinearSize * sizeof(float),
                                  cudaMemcpyDeviceToHost) );

        for(int z = pad_z; z < padded->cubeDepth-pad_z; z++)
          for(int y = pad_y; y < padded->cubeHeight-pad_y; y++)
            for(int x = pad_x; x < padded->cubeWidth-pad_x; x++)
              res->put(x0+x-pad_x, y0+y-pad_y, z0+z-pad_z,
                     h_OutputGPU[(z*maxTileSizeY + y)*maxTileSizeX + x]);

        delete padded;
      }
    }
  }
  printf("Done with the computations\n");


  printf("Shutting down...\n");
  cutilSafeCall( cudaFree(d_Buffer ) );
  cutilSafeCall( cudaFree(d_Input) );
  free(h_OutputGPU);
  free(h_Buffer);
  free(h_Input);

  cutilCheckError(cutDeleteTimer(hTimer));

  cudaThreadExit();

  exit(0);
}
