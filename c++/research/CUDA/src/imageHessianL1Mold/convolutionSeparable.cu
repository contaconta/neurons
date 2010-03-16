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
 
 

#include <assert.h>
#include <cutil_inline.h>
// #include <vector>


////////////////////////////////////////////////////////////////////////////////
// Convolution kernel storage
////////////////////////////////////////////////////////////////////////////////
__constant__ float c_Kernel_h[100];
__constant__ float c_Kernel_v[100];

extern "C" void setConvolutionKernel_horizontal(float *h_Kernel, int kernel_length){
  cudaMemcpyToSymbol(c_Kernel_h, h_Kernel, kernel_length * sizeof(float));
}
extern "C" void setConvolutionKernel_vertical(float *h_Kernel, int kernel_length){
  cudaMemcpyToSymbol(c_Kernel_v, h_Kernel, kernel_length * sizeof(float));
}

////////////////////////////////////////////////////////////////////////////////
// Constants
////////////////////////////////////////////////////////////////////////////////
#define   ROWS_BLOCKDIM_X 16
#define   ROWS_BLOCKDIM_Y 16
#define   ROWS_RESULT_STEPS 8
#define   ROWS_HALO_STEPS 3
#define   COLUMNS_BLOCKDIM_X 16
#define   COLUMNS_BLOCKDIM_Y 16
#define   COLUMNS_RESULT_STEPS 8
#define   COLUMNS_HALO_STEPS 3





////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowsKernel(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int pitch,
    int kernel_radius
){
    __shared__ float s_Data[ROWS_BLOCKDIM_Y]
                           [(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

    //Offset to the left halo edge
    const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) *
                      ROWS_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

    //Main data
    #pragma unroll
    for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
        s_Data[threadIdx.y]
              [threadIdx.x + i * ROWS_BLOCKDIM_X]
          = d_Src[i * ROWS_BLOCKDIM_X];

    //Left halo
    for(int i = 0; i < ROWS_HALO_STEPS; i++){
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] =
            (baseX >= -i * ROWS_BLOCKDIM_X ) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
    }

    //Right halo
    for(int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS;
        i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++){
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] =
            (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
    }

    //Compute and store results
    __syncthreads();
    #pragma unroll
    for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++){
        float sum = 0;

        #pragma unroll
        for(int j = -kernel_radius; j <= kernel_radius; j++)
            sum += c_Kernel_h[kernel_radius - j] * 
                   s_Data    [threadIdx.y]
                             [threadIdx.x + i * ROWS_BLOCKDIM_X + j];
        d_Dst[i * ROWS_BLOCKDIM_X] = sum;
    }
}

extern "C" void convolutionRowsGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int kernel_radius
){
    assert( ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= kernel_radius );
    //There is a rational division of the image into blocks
    assert( imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0 );
    assert( imageH % ROWS_BLOCKDIM_Y == 0 );

    dim3 blocks(imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), imageH / ROWS_BLOCKDIM_Y);
    dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);

    convolutionRowsKernel<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
        imageW,
        kernel_radius
    );
    cutilCheckMsg("convolutionRowsKernel() execution failed\n");
}



////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumnsKernel(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int pitch,
    int kernel_radius
){
    __shared__ float s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

    //Main data
    #pragma unroll
    for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];

    //Upper halo
    for(int i = 0; i < COLUMNS_HALO_STEPS; i++)
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = 
            (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;

    //Lower halo
    for(int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
            (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;

    //Compute and store results
    __syncthreads();
    #pragma unroll
    for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++){
        float sum = 0;
        #pragma unroll
        for(int j = -kernel_radius; j <= kernel_radius; j++)
            sum += c_Kernel_v[kernel_radius - j] * s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];

        d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
    }
}

extern "C" void convolutionColumnsGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int kernel_radius
){
    assert( COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= kernel_radius );
    assert( imageW % COLUMNS_BLOCKDIM_X == 0 );
    assert( imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0 );

    dim3 blocks(imageW / COLUMNS_BLOCKDIM_X, imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
    dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);

    convolutionColumnsKernel<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
        imageW,
        kernel_radius
    );
    cutilCheckMsg("convolutionColumnsKernel() execution failed\n");
}

////////////////////////////////////////////////////////////////////////////////
// Simple interface to compute a derivative
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// Computes the higher eigenvalue of the hessian
////////////////////////////////////////////////////////////////////////////////
__global__ void hessianKernel(
       float *d_output,
       float *d_gxx,
       float *d_gxy,
       float *d_gyy,
       float scale,
       int imageW,
       int imageH,
       int invert
){
  int i = (blockDim.y*blockIdx.y + threadIdx.y)*imageW +
    blockDim.x*blockIdx.x+threadIdx.x;
  float a, b, c;
  a = invert*d_gxx[i];
  b = invert*d_gxy[i];
  c = invert*d_gyy[i];
  d_output[i] = ((a+c)/2 + sqrt( (a-c)*(a-c) + 4*b*b)/2)*scale*scale;
  // d_output[i] = (a-c)*(a-c) + 4*b*b;
  // d_output[i] = b;
}



extern "C" void hessianGPU
(
 float *d_output,
 float *d_gxx,
 float *d_gxy,
 float *d_gyy,
 float scale,
 int imageW,
 int imageH,
 int invert
 )
{
  dim3 gird (ceil(float(imageW)/ROWS_BLOCKDIM_X),ceil(float(imageH)/ROWS_BLOCKDIM_Y));
  dim3 block(ROWS_BLOCKDIM_X,ROWS_BLOCKDIM_Y);
  hessianKernel<<<gird, block>>>( d_output, d_gxx, d_gxy, d_gyy, scale, imageW, imageH, int invert );
  cutilCheckMsg("hessianKernel() execution failed\n");
}

//////////////// MAX /////////
__global__ void maxKernel(
       float *d_output,
       float *d_isMaxThanOutput,
       int imageW,
       int imageH
){
  int i = (blockDim.y*blockIdx.y + threadIdx.y)*imageW +
    blockDim.x*blockIdx.x+threadIdx.x;

  if(d_isMaxThanOutput[i] >= d_output[i])
    d_output[i] = d_isMaxThanOutput[i];
}



extern "C" void maxGPU
(
 float *d_output,
 float *d_isMaxThanOutput,
 int imageW,
 int imageH
 )
{
  dim3 gird (ceil(float(imageW)/ROWS_BLOCKDIM_X),ceil(float(imageH)/ROWS_BLOCKDIM_Y));
  dim3 block(ROWS_BLOCKDIM_X,ROWS_BLOCKDIM_Y);
  maxKernel<<<gird, block>>>( d_output, d_isMaxThanOutput, imageW, imageH );
  cutilCheckMsg("maxKernel() execution failed\n");
}

__global__ void maxKernel_scale(
       float *d_output,
       float *d_scale,
       float *d_isMaxThanOutput,
       float scale,
       int imageW,
       int imageH
){
  int i = (blockDim.y*blockIdx.y + threadIdx.y)*imageW +
    blockDim.x*blockIdx.x+threadIdx.x;

  if(d_isMaxThanOutput[i] > d_output[i]){
    d_output[i] = d_isMaxThanOutput[i];
    if(d_output[i] > 30)
      d_scale[i]  = scale;
  }
}



extern "C" void maxGPU_scale
(
 float *d_output,
 float *d_scale,
 float *d_isMaxThanOutput,
 float scale,
 int imageW,
 int imageH
 )
{
  dim3 gird (ceil(float(imageW)/ROWS_BLOCKDIM_X),ceil(float(imageH)/ROWS_BLOCKDIM_Y));
  dim3 block(ROWS_BLOCKDIM_X,ROWS_BLOCKDIM_Y);
  maxKernel_scale<<<gird, block>>>( d_output, d_scale, d_isMaxThanOutput, scale,
                                    imageW, imageH );
  cutilCheckMsg("maxKernel() execution failed\n");

}



////////////////////////// PUT VALUE

__global__ void putKernel(
       float *d_output,
       float value,
       int imageW,
       int imageH
){
  int i = (blockDim.y*blockIdx.y + threadIdx.y)*imageW +
    blockDim.x*blockIdx.x+threadIdx.x;

    d_output[i] = value;

}



extern "C" void putGPU
(
 float *d_output,
 float value,
 int imageW,
 int imageH
 )
{
  dim3 gird (ceil(float(imageW)/ROWS_BLOCKDIM_X),ceil(float(imageH)/ROWS_BLOCKDIM_Y));
  dim3 block(ROWS_BLOCKDIM_X,ROWS_BLOCKDIM_Y);
  putKernel<<<gird, block>>>( d_output, value, imageW, imageH );
  cutilCheckMsg("maxKernel() execution failed\n");
}
