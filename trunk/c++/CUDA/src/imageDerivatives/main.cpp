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



////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){
    float
        *h_Kernel_h,
        *h_Kernel_v,
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
    Image<float>* img = new Image<float>(argv[1],0);
    Image<float>* res = img->create_blank_image_float(argv[2]);
    const int imageW = img->width;
    const int imageH = img->height;
    const int maxTileSize = 1024;

    printf("Initializing kernels\n");
    vector<float> kernel_h = Mask::gaussian_mask(2, 4, 1);
    vector<float> kernel_v = Mask::gaussian_mask(0, 4, 1);
    int kernel_radius_h = kernel_h.size()/2;
    int kernel_length_h = kernel_h.size();
    int kernel_radius_v = kernel_v.size()/2;
    int kernel_length_v = kernel_v.size();



    printf("%i x %i, kernel_radius = %i, %i\n", 
           imageW, imageH, kernel_radius_h, kernel_radius_v);


    printf("Allocating and intializing host arrays...\n");
    h_Kernel_h  = (float *)malloc(kernel_length_h * sizeof(float));
    h_Kernel_v  = (float *)malloc(kernel_length_v * sizeof(float));
    h_Input     = (float *)malloc(maxTileSize * maxTileSize * sizeof(float));
    h_Buffer    = (float *)malloc(maxTileSize * maxTileSize * sizeof(float));
    h_OutputGPU = (float *)malloc(maxTileSize * maxTileSize * sizeof(float));
    srand(200);

    printf("Configuring the kernels\n");
    for(unsigned int i = 0; i < kernel_length_h; i++)
      h_Kernel_h[i] = kernel_h[i];
    for(unsigned int i = 0; i < kernel_length_v; i++)
      h_Kernel_v[i] = kernel_v[i];
    setConvolutionKernel_horizontal(h_Kernel_h, kernel_length_h);
    setConvolutionKernel_vertical  (h_Kernel_v, kernel_length_v);

    printf("Allocating CUDA arrays...\n");
    cutilSafeCall( cudaMalloc((void **)&d_Input, maxTileSize * maxTileSize * sizeof(float)) );
    cutilSafeCall( cudaMalloc((void **)&d_Output,maxTileSize * maxTileSize * sizeof(float)) );
    cutilSafeCall( cudaMalloc((void **)&d_Buffer,maxTileSize * maxTileSize * sizeof(float)) );


    // Here should come the loop
    // Variables required to split the image into tiles
    int pad_x = kernel_length_h;
    int pad_y = kernel_length_v;
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

        printf("Running GPU convolution\n");
        cutilSafeCall( cudaThreadSynchronize() );
        convolutionRowsGPU(
                           d_Buffer,
                           d_Input,
                           maxTileSize,
                           maxTileSize,
                           kernel_radius_h
                           );

        convolutionColumnsGPU(
                              d_Output,
                              d_Buffer,
                              maxTileSize,
                              maxTileSize,
                              kernel_radius_v
                              );
        cutilSafeCall( cudaThreadSynchronize() );

        cutilSafeCall( cudaMemcpy(h_OutputGPU, d_Output,
                                  maxTileSize* maxTileSize * sizeof(float),
                                  cudaMemcpyDeviceToHost) );

        for(int y = pad_y; y < padded->height-pad_y; y++)
          for(int x = pad_x; x < padded->width-pad_x; x++)
            res->put(x0+x-pad_x, y0+y-pad_y,
                     h_OutputGPU[y*maxTileSize + x]);

        delete padded;
      }
    }

    res->save();


    printf("Shutting down...\n");
    cutilSafeCall( cudaFree(d_Buffer ) );
    cutilSafeCall( cudaFree(d_Output) );
    cutilSafeCall( cudaFree(d_Input) );
    free(h_OutputGPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Kernel_h);
    free(h_Kernel_v);

    cutilCheckError(cutDeleteTimer(hTimer));

    cudaThreadExit();

    exit(0);












    // for(int y = 0; y < imageH; y++)
      // for(int x = 0; x < imageW; x++)
        // h_Input[y*imageW + x] = img->at(x,y);




    // float gpuTime = cutGetTimerValue(hTimer) / (float)1.0;
    // printf("Average GPU convolution time : %f msec //%f Mpixels/sec\n",
           // gpuTime, 1e-6 * imageW * imageH / (gpuTime * 0.001));

    // printf("Reading back GPU results...\n");




    // for(int y = 0; y < imageH; y++)
      // for(int x = 0; x < imageW; x++)
        // res->put(x,y, h_OutputGPU[x+y*imageW]);
    // res->save();





}
