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
__constant__ float c_Kernel_d[100];

extern "C" void setConvolutionKernel_horizontal(float *h_Kernel, int kernel_length){
  cudaMemcpyToSymbol(c_Kernel_h, h_Kernel, kernel_length * sizeof(float));
}
extern "C" void setConvolutionKernel_vertical(float *h_Kernel, int kernel_length){
  cudaMemcpyToSymbol(c_Kernel_v, h_Kernel, kernel_length * sizeof(float));
}

extern "C" void setConvolutionKernel_depth(float *h_Kernel, int kernel_length){
  cudaMemcpyToSymbol(c_Kernel_d, h_Kernel, kernel_length * sizeof(float));
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
#define   DEPTH_BLOCKDIM_Y 16
#define   DEPTH_BLOCKDIM_Z 16
#define   DEPTH_RESULT_STEPS 4
#define   DEPTH_HALO_STEPS 3



////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowsKernel(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int imageD,
    int kernel_radius
){
    __shared__ float s_Data[ROWS_BLOCKDIM_Y]
                           [(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

    //Offset to the left halo edge
    int n_blocks_per_row = imageW/(ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X);
    int basez = floor(float(blockIdx.x)/n_blocks_per_row);

    int blockx = blockIdx.x - basez*n_blocks_per_row;
    // const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) *
                      // ROWS_BLOCKDIM_X + threadIdx.x;
    const int baseX = (blockx * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) *
                      ROWS_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

    d_Src += basez*imageW*imageH + baseY * imageW + baseX;
    d_Dst += basez*imageW*imageH + baseY * imageW + baseX;

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
    // #pragma unroll
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
    int imageD,
    int kernel_radius
){
    assert( ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= kernel_radius );
    //There is a rational division of the image into blocks
    assert( imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0 );
    assert( imageH % ROWS_BLOCKDIM_Y == 0 );

    dim3 blocks(imageD*(imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X)),
                imageH / ROWS_BLOCKDIM_Y);
    dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);
    convolutionRowsKernel<<<blocks, threads>>>(
                                               // &d_Dst[i*imageH*imageW],
                                               // &d_Src[i*imageH*imageW],
                                               d_Dst,
                                               d_Src,
                                               imageW,
                                               imageH,
                                               imageD,
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

    int n_blocks_per_column = imageH/(COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y);
    int basez = floor(float(blockIdx.y)/n_blocks_per_column);
    int blocky = blockIdx.y - basez*n_blocks_per_column;

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blocky * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
    d_Src += basez*imageH*imageW + baseY * imageH + baseX;
    d_Dst += basez*imageH*imageW + baseY * imageH + baseX;

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
    // #pragma unroll
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
    int imageD,
    int kernel_radius
){
    assert( COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= kernel_radius );
    assert( imageW % COLUMNS_BLOCKDIM_X == 0 );
    assert( imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0 );

    dim3 blocks(imageW / COLUMNS_BLOCKDIM_X, imageD * imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
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
// Depth convolution filter - Really naive implementation
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionDepthKernel(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int imageD,
    int kernel_radius
){
    __shared__ float s_Data[DEPTH_BLOCKDIM_Y]
                           [(DEPTH_RESULT_STEPS + 2 * DEPTH_HALO_STEPS) * DEPTH_BLOCKDIM_Z];

    //Offset to the left halo edge
    int n_blocks_per_depth = imageD / (DEPTH_RESULT_STEPS * DEPTH_BLOCKDIM_Z);
    int basex = floor(float(blockIdx.x)/n_blocks_per_depth);

    int blockz = blockIdx.x - basex*n_blocks_per_depth;

    const int baseZ = (blockz * DEPTH_RESULT_STEPS - DEPTH_HALO_STEPS)*DEPTH_BLOCKDIM_Z +
                      threadIdx.x;
    const int baseY = blockIdx.y * DEPTH_BLOCKDIM_Y + threadIdx.y;

    //Put the pointers to the beginning of the data
    d_Src += baseZ * imageW * imageH + baseY * imageW + basex;
    d_Dst += baseZ * imageW * imageH + baseY * imageW + basex;

    // //Main data
    #pragma unroll
    for(int i = DEPTH_HALO_STEPS; i < DEPTH_HALO_STEPS + DEPTH_RESULT_STEPS; i++)
        s_Data[threadIdx.y]
              [threadIdx.x + i * DEPTH_BLOCKDIM_Z]
          = d_Src[i * DEPTH_BLOCKDIM_Z * imageH * imageW];

    //Left halo
    for(int i = 0; i < DEPTH_HALO_STEPS; i++){
        s_Data[threadIdx.y][threadIdx.x + i * DEPTH_BLOCKDIM_Z] =
            (baseZ >= -i * DEPTH_BLOCKDIM_Z ) ? 
          d_Src[i * DEPTH_BLOCKDIM_Z * imageH * imageW] : 0;
    }

    // Right halo
    for(int i = DEPTH_HALO_STEPS + DEPTH_RESULT_STEPS;
        i < DEPTH_HALO_STEPS + DEPTH_RESULT_STEPS + DEPTH_HALO_STEPS; i++){
        s_Data[threadIdx.y][threadIdx.x + i * DEPTH_BLOCKDIM_Z] =
            (imageD - baseZ > i * DEPTH_BLOCKDIM_Z ) ? 
          d_Src[i * DEPTH_BLOCKDIM_Z * imageH * imageW] : 0;
    }

    // //Compute and store results
    __syncthreads();
    #pragma unroll
    for(int i = DEPTH_HALO_STEPS; i < DEPTH_HALO_STEPS + DEPTH_RESULT_STEPS; i++){
        float sum = 0;
        #pragma unroll
        for(int j = -kernel_radius; j <= kernel_radius; j++)
            sum += c_Kernel_d[kernel_radius - j] * 
                   s_Data    [threadIdx.y]
                             [threadIdx.x + i * DEPTH_BLOCKDIM_Z + j];
        d_Dst[i * DEPTH_BLOCKDIM_Z * imageH * imageW] = sum;
    }
}

extern "C" void convolutionDepthGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int imageD,
    int kernel_radius
){
    assert( DEPTH_BLOCKDIM_Z * DEPTH_HALO_STEPS >= kernel_radius );
    //There is a rational division of the image into blocks
    assert( imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0 );
    assert( imageH % ROWS_BLOCKDIM_Y == 0 );

    dim3 blocks(imageW*imageD / (DEPTH_RESULT_STEPS * DEPTH_BLOCKDIM_Z),
                imageH / DEPTH_BLOCKDIM_Y);
    dim3 threads(DEPTH_BLOCKDIM_Z, DEPTH_BLOCKDIM_Y);

    // for(int x = 0; x < imageW; x++)
      convolutionDepthKernel<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
        imageD,
        kernel_radius
    );
    cutilCheckMsg("convolutionRowsKernel() execution failed\n");
}



////////////////////////////////////////////////////////////////////////////////
// Computes the higher eigenvalue of the hessian
////////////////////////////////////////////////////////////////////////////////
__global__ void hessianKernel
(
 float *d_output,
 float *d_gxx,
 float *d_gxy,
 float *d_gxz,
 float *d_gyy,
 float *d_gyz,
 float *d_gzz,
 int imageW,
 int imageH,
 int imageD,
){
  int z = ceil(blockIdx.x/ROWS_BLOCKDIM_X);
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  int x = (blockIdx.x - z*ROWS_BLOCKDIM_X)*blockDim.x + threadIdx.x;
  int i = z*imageW*imageH + y*imageW + x
  float a, b, c;
  a = d_gxx[i];
  b = d_gxy[i];
  c = d_gyy[i];
  d_output[i] = (a+c)/2 + sqrt( (a-c)*(a-c) + 4*b*b)/2;
  // d_output[i] = b;
}



extern "C" void hessianGPU
(
 float *d_output,
 float *d_gxx,
 float *d_gxy,
 float *d_gxz,
 float *d_gyy,
 float *d_gyz,
 float *d_gzz,
 int imageW,
 int imageH,
 int imageD,
 )
{
  dim3 gird (imageD*ceil(float(imageW)/ROWS_BLOCKDIM_X),ceil(float(imageH)/ROWS_BLOCKDIM_Y));
  dim3 block(ROWS_BLOCKDIM_X,ROWS_BLOCKDIM_Y);
  hessianKernel<<<gird, block>>>( d_output, d_gxx, d_gxy, d_gxz,
                                  d_gyy, d_gyz, d_gzz, imageW, imageH, imageD );
  cutilCheckMsg("hessianKernel() execution failed\n");
}
