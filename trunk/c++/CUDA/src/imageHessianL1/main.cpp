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
#include "Mask.h"

// extern "C" void set_horizontal_kernel(vector<float>& kernel);
// extern "C" void set_vertical_kernel(vector<float>& kernel);

extern "C" void setConvolutionKernel_horizontal(float *h_Kernel, int kernel_length);

extern "C" void setConvolutionKernel_vertical(float *h_Kernel, int kernel_length);

extern "C" void convolutionRowsGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int kernel_radius
);

extern "C" void convolutionColumnsGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
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

extern "C" void convolution_separable
( float* d_Dst,
  float* d_Src,
  vector< float >& kernel_h,
  vector< float >& kernel_v,
  int sizeX,
  int sizeY,
  float* d_tmp
  )
{
  set_horizontal_kernel(kernel_h);
  set_vertical_kernel(kernel_v);

  cutilSafeCall( cudaThreadSynchronize() );
  convolutionRowsGPU(
                     d_tmp,
                     d_Src,
                     sizeX,
                     sizeY,
                     floor(kernel_h.size()/2)
                     );

  convolutionColumnsGPU(
                        d_Dst,
                        d_tmp,
                        sizeX,
                        sizeY,
                        floor(kernel_v.size()/2)
                        );
  cutilSafeCall( cudaThreadSynchronize() );
}


extern "C" void hessian
( float* d_Buffer,
  float* d_Input,
  float sigma,
  int sizeX,
  int sizeY
  )
{
  float *d_gxx, *d_gxy, *d_gyy;

  cutilSafeCall( cudaMalloc((void **)&d_gxx,   sizeX * sizeY * sizeof(float)) );
  cutilSafeCall( cudaMalloc((void **)&d_gxy,   sizeX * sizeY * sizeof(float)) );
  cutilSafeCall( cudaMalloc((void **)&d_gyy,   sizeX * sizeY * sizeof(float)) );

  vector<float> kernel_0 = Mask::gaussian_mask(0, sigma, 1);
  vector<float> kernel_1 = Mask::gaussian_mask(1, sigma, 1);
  vector<float> kernel_2 = Mask::gaussian_mask(2, sigma, 1);

  printf("  ->computing convolutions\n");
  convolution_separable( d_gxx, d_Input, kernel_2, kernel_0,
                         sizeX, sizeY, d_Buffer );
  convolution_separable( d_gxy, d_Input, kernel_1, kernel_1,
                         sizeX, sizeY, d_Buffer );
  convolution_separable( d_gyy, d_Input, kernel_0, kernel_2,
                         sizeX, sizeY, d_Buffer );

  printf("  ->computing the hessian\n");
  hessianGPU(d_Buffer, d_gxx, d_gxy, d_gyy, sizeX, sizeY);

  cutilSafeCall( cudaFree(d_gxx ) );
  cutilSafeCall( cudaFree(d_gxy ) );
  cutilSafeCall( cudaFree(d_gyy ) );
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
    *d_Buffer;

  float sigma = atof(argv[2]);

  printf("Initializing CUDA\n");
  unsigned int hTimer;
  cudaSetDevice( cutGetMaxGflopsDeviceId() );
  cutilCheckError(cutCreateTimer(&hTimer));


  printf("Initializing image and result image\n");
  Image<float>* img = new Image<float>(argv[1],0);
  Image<float>* res = img->create_blank_image_float(argv[3]);
  const int imageW = img->width;
  const int imageH = img->height;
  const int maxTileSize = 1024;

  printf("Allocating and intializing host arrays...\n");
  h_Input     = (float *)malloc(maxTileSize * maxTileSize * sizeof(float));
  h_Buffer    = (float *)malloc(maxTileSize * maxTileSize * sizeof(float));
  h_OutputGPU = (float *)malloc(maxTileSize * maxTileSize * sizeof(float));
  srand(200);

  printf("Allocating CUDA arrays...\n");
  cutilSafeCall( cudaMalloc((void **)&d_Input, maxTileSize * maxTileSize * sizeof(float)) );
  cutilSafeCall( cudaMalloc((void **)&d_Buffer,maxTileSize * maxTileSize * sizeof(float)) );

  // Here should come the loop
  // Variables required to split the image into tiles
  vector< float > kernelSample = Mask::gaussian_mask(sigma, 0);
  int pad_x = kernelSample.size();
  int pad_y = kernelSample.size();
  int tile_size_x  = maxTileSize - 2*pad_x;
  int tile_size_y  = maxTileSize - 2*pad_y;
  int n_tiles_horiz = ceil(float(imageW) /tile_size_x);
  int n_tiles_vert  = ceil(float(imageH) /tile_size_y);

  // for the tiles in horizontal
  for(int ty = 0; ty < n_tiles_vert; ty++){
    for(int tx = 0; tx < n_tiles_horiz; tx++){

      int x0, y0, x1, y1;
      x0 = tx*tile_size_x;
      y0 = ty*tile_size_y;
      x1 = min((tx+1)*tile_size_x -1, imageW-1);
      y1 = min((ty+1)*tile_size_y -1, imageH-1);
      printf(" iteration [%i,%i], pad= [%i,%i]->[%i,%i]\n",
             tx, ty, x0, y0,x1, y1);

      Image<float>* padded = img->get_padded_tile
        (x0,y0,x1,y1,pad_x, pad_y);

      //puts the padded image into the array where the convolutions are going to be done
      for(int y = 0; y < padded->height; y++)
        for(int x = 0; x < padded->width; x++)
          h_Input[y*maxTileSize + x] = padded->at(x,y);

      cutilSafeCall( cudaMemcpy(d_Input, h_Input,
                                maxTileSize * maxTileSize * sizeof(float),
                                cudaMemcpyHostToDevice) );


      hessian(d_Buffer, d_Input, sigma, maxTileSize, maxTileSize);

      cutilSafeCall( cudaThreadSynchronize() );

      cutilSafeCall( cudaMemcpy(h_OutputGPU, d_Buffer,
                                maxTileSize* maxTileSize * sizeof(float),
                                cudaMemcpyDeviceToHost) );

      for(int y = pad_y; y < padded->height-pad_y; y++)
        for(int x = pad_x; x < padded->width-pad_x; x++)
          res->put(x0+x-pad_x, y0+y-pad_y,
                   h_OutputGPU[y*maxTileSize + x]);

      delete padded;
    }
  }

  printf("Done with the computations\n");
  res->save();


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
