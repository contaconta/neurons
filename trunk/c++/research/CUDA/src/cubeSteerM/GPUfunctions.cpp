#include "GPUfunctions.h"


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
  float *d_gxx,
  float *d_gxy,
  float *d_gxz,
  float *d_gyy,
  float *d_gyz,
  float *d_gzz,
  int sizeX,
  int sizeY,
  int sizeZ
  )
{

  vector<float> kernel_0 = Mask::gaussian_mask(0, sigma, 1);
  vector<float> kernel_1 = Mask::gaussian_mask(1, sigma, 1);
  vector<float> kernel_2 = Mask::gaussian_mask(2, sigma, 1);

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

  hessianGPU(d_Buffer, d_gxx, d_gxy, d_gxy, d_gyy, d_gyz, d_gzz, sizeX, sizeY, sizeZ);
}


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
  )
{

  vector<float> kernel_0 = Mask::gaussian_mask(0, sigma, 1);
  vector<float> kernel_1 = Mask::gaussian_mask(1, sigma, 1);
  vector<float> kernel_2 = Mask::gaussian_mask(2, sigma, 1);

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

  // printf("H orientation: o: %i, t: %i p: %i\n",
         // d_Buffer, d_Output_theta, d_Output_phi);
  hessianGPU_orientation
    (d_Buffer, d_Output_theta, d_Output_phi,
     d_gxx, d_gxy, d_gxy, d_gyy, d_gyz, d_gzz, sizeX, sizeY, sizeZ);
}
