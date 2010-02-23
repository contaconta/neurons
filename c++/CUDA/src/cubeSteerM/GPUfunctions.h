#ifndef GPU_FUNCTIONS_H_
#define GPU_FUNCTIONS_H_

#include <vector>
#include <cutil_inline.h>
#include "Mask.h"

using namespace std;

extern "C" void set_horizontal_kernel(vector<float>& kernel);

extern "C" void set_vertical_kernel(vector<float>& kernel);

extern "C" void set_depth_kernel(vector<float>& kernel);

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
 float *d_gxz,
 float *d_gyy,
 float *d_gyz,
 float *d_gzz,
 int imageW,
 int imageH,
 int imageD
 );


extern "C" void hessianGPU_orientation
(
 float *d_output,
 float *d_output_theta,
 float *d_output_phi,
 float *d_gxx,
 float *d_gxy,
 float *d_gxz,
 float *d_gyy,
 float *d_gyz,
 float *d_gzz,
 int imageW,
 int imageH,
 int imageD
 );

extern "C" void hessian
( float* d_Buffer,
  float* d_Input,
  float sigma,
  float *d_gxx,
  float *d_gxy,
  float *d_gxz,
  float *d_gyy,
  float *d_gyz,
  float *d_gzz,
  int sizeX,
  int sizeY,
  int sizeZ
  );

extern "C" void hessian_orientation
( float* d_Buffer,
  float* d_Output_theta,
  float* d_Output_phi,
  float* d_Input,
  float sigma,
  float *d_gxx,
  float *d_gxy,
  float *d_gxz,
  float *d_gyy,
  float *d_gyz,
  float *d_gzz,
  int sizeX,
  int sizeY,
  int sizeZ
  );

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
  );



#endif
