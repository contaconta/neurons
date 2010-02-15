#include <assert.h>
#include <cutil_inline.h>
#include <math_functions.h>
#include "GPUcommon.h"

////////////////////////////////////////////////////////////////////////////////
// ROTATIONAL FEATURES ROCK
////////////////////////////////////////////////////////////////////////////////
__constant__ float* d_derivs[21]; //Pointers to the derivatives

extern "C" void set_derivatives_pointer(float **h_derivs){
  cudaMemcpyToSymbol(d_derivs, h_derivs, 21 * sizeof(float*));
}


__device__
int positionOrder(int order){
  if(order == 2) return 0;
  if(order == 4) return 6;
  return 0;
}

__device__
int numDerivsOrder(int order){
  if(order == 2) return 6;
  if(order == 4) return 15;
  return 0;
}

__device__
int deriv2idx(int dx, int dy, int dz){
  if(dx+dy+dz == 2){
    if(dx == 2)                  return 0;
    if( (dx == 1) && (dy == 1) ) return 1;
    if( (dx == 1) && (dz == 1) ) return 2;
    if(dy == 2)                  return 3;
    if( (dy == 1) && (dz == 1) ) return 4;
    if(dz == 2)                  return 5;
  }
  else if (dx+dy+dz == 4){
    if((dx==4)&(dy==0)&(dz==0))  return 0;
    if((dx==3)&(dy==1)&(dz==0))  return 1;
    if((dx==3)&(dy==0)&(dz==1))  return 2;
    if((dx==2)&(dy==2)&(dz==0))  return 3;
    if((dx==2)&(dy==1)&(dz==1))  return 4;
    if((dx==2)&(dy==0)&(dz==2))  return 5;
    if((dx==1)&(dy==3)&(dz==0))  return 6;
    if((dx==0)&(dy==4)&(dz==0))  return 7;
    if((dx==0)&(dy==3)&(dz==1))  return 8;
    if((dx==1)&(dy==2)&(dz==1))  return 9;
    if((dx==0)&(dy==2)&(dz==2))  return 10;
    if((dx==1)&(dy==0)&(dz==3))  return 11;
    if((dx==0)&(dy==1)&(dz==3))  return 12;
    if((dx==0)&(dy==0)&(dz==4))  return 13;
    if((dx==1)&(dy==1)&(dz==2))  return 14;
  }
 return -1;
}

__device__
void idx2deriv(int idx, int ord, int& m, int& n, int& p){
  if (ord==2){
    if(idx==0){ m = 2; n = 0; p = 0; return;}
    if(idx==1){ m = 1; n = 1; p = 0; return;}
    if(idx==2){ m = 1; n = 0; p = 1; return;}
    if(idx==3){ m = 0; n = 2; p = 0; return;}
    if(idx==4){ m = 0; n = 1; p = 1; return;}
    if(idx==5){ m = 0; n = 0; p = 2; return;}
  }

  if (ord==4){
    if(idx==0 ){ m = 4; n = 0; p = 0; return;}
    if(idx==1 ){ m = 3; n = 1; p = 0; return;}
    if(idx==2 ){ m = 3; n = 0; p = 1; return;}
    if(idx==3 ){ m = 2; n = 2; p = 0; return;}
    if(idx==4 ){ m = 2; n = 1; p = 1; return;}
    if(idx==5 ){ m = 2; n = 0; p = 2; return;}
    if(idx==6 ){ m = 1; n = 3; p = 0; return;}
    if(idx==7 ){ m = 0; n = 4; p = 0; return;}
    if(idx==8 ){ m = 0; n = 3; p = 1; return;}
    if(idx==9 ){ m = 1; n = 2; p = 1; return;}
    if(idx==10){ m = 0; n = 2; p = 2; return;}
    if(idx==11){ m = 1; n = 0; p = 3; return;}
    if(idx==12){ m = 0; n = 1; p = 3; return;}
    if(idx==13){ m = 0; n = 0; p = 4; return;}
    if(idx==14){ m = 1; n = 1; p = 2; return;}
  }


}


__device__ int factorial_n(int n)
{
  if(n==0) return 1;
  if(n==1) return 1;
  if(n==2) return 2;
  if(n==3) return 6;
  if(n==4) return 24;
  if(n==5) return 120;
  if(n==6) return 720;
  if(n==7) return 5040;
  if(n==8) return 40320;
  if(n==9) return  362880;
  if(n==10) return 3628800;

  return 0;
}

__device__ void rotateTheta
(float* dest, float* orig, float theta)
{
  float tmp1, tmp2, tmp3, ct, st, sp, phi;
  int m, n, p, i, j, k, l, q, b_v, ord, idx;

  for(int i = 0; i < 21; i++) dest[i] = 0;

  //In Aguet's formula the phi is defined in the opposite sense of the clock (towards x negative),
  // from there the inversion of phi
  phi = 0;
  ct = cos(theta);
  st = sin(theta);
  sp = sin(-phi);

  //For each order and each component of the order
  for(ord = 2; ord <=4; ord+=2){
    b_v = positionOrder(ord);
    for(idx = 0; idx < numDerivsOrder(ord); idx++){
      idx2deriv(idx, ord, m, n, p);
      for(i=0;i<=m;i++)
        for(k=0;k<=n;k++)
          for(q=0;q<=p;q++)
            for(j=0;j<=i;j++)
              for(l=0;l<=k;l++){
                if(j+l+p-q != 0)
                  continue;
                tmp1 = factorial_n(m)*factorial_n(n)*factorial_n(p)*pow(-1.0,i-j+p-q);
                tmp2 = factorial_n(m-i)*factorial_n(i-j)*factorial_n(j)*factorial_n(n-k)*
                  factorial_n(k-l)*factorial_n(l)*factorial_n(p-q)*factorial_n(q);
                tmp3 = ((double)tmp1)/tmp2;
                tmp3 = tmp3*pow(ct,m-i+j+k-l);
                tmp3 = tmp3*pow(st,i-j+n-k+l);
                tmp3 = tmp3*pow(sp,j+l+p-q);    //only if j+l+p-q is equal to 0
                dest[b_v + deriv2idx(m-i+n-k+p-q, i-j+k-l, j+l+q)] +=
                  tmp3*orig[b_v + deriv2idx(m,n,p)];
              }
    }
  }
}

__device__ void rotatePhi
(float* dest, float* orig, float phi)
{
  for(int i = 0; i < 21; i++) dest[i] = 0;
  float tmp1, tmp2, tmp3, cp, sp;
  int m, n, p, i, j, k, l, q, b_v, ord, idx;

  //In Aguet's formula the phi is defined in the opposite sense of the clock (towards x negative),
  // from there the inversion of phi
  cp = cos(-phi);
  sp = sin(-phi);

  //For each order and each component of the order
  for(ord = 2; ord <=4; ord+=2){
    b_v = positionOrder(ord);
    for(idx = 0; idx < numDerivsOrder(ord); idx++){
      idx2deriv(idx, ord, m, n, p);
      for(i=0;i<=m;i++)
        for(k=0;k<=n;k++)
          for(q=0;q<=p;q++)
            for(j=0;j<=i;j++)
              for(l=0;l<=k;l++){
                if(i-j+n-k+l != 0)
                  continue;
                tmp1 = factorial_n(m)*factorial_n(n)*factorial_n(p)*pow(-1.0,i-j+p-q);
                tmp2 = factorial_n(m-i)*factorial_n(i-j)*factorial_n(j)*factorial_n(n-k)*
                  factorial_n(k-l)*factorial_n(l)*factorial_n(p-q)*factorial_n(q);
                tmp3 = ((double)tmp1)/tmp2;
                tmp3 = tmp3*pow(cp,m-i+n-k+q);
                tmp3 = tmp3*pow(sp,j+l+p-q);
                dest[b_v + deriv2idx(m-i+n-k+p-q, i-j+k-l, j+l+q)] +=
                  tmp3*orig[b_v + deriv2idx(m,n,p)];
              }
    }
  }
}



__device__ void rotateFeatureVectorInverse
(float* dest, float* orig, float* tmp, float theta, float phi)
{
  rotateTheta(tmp, orig, -theta);
  rotatePhi(dest, tmp, -phi);
}

// __device__ float SVMResponse
// (float* vector,
 // float** support_vectors,
 // float* alphas,
 // float sk)
// {

  // return 0.0;
// }


/*********** KERNEL ***********************/
__global__ void rtfSVMKernel
(float*  d_output,
 float*  d_theta,
 float*  d_phi,
 float*  d_sv,
 float*  d_alphas,
 float   sk,
 int     n_sv_i,
 int     n_sv_e,
 int     imageW,
 int     imageH,
 int     imageD
)
{
  int n_blocks_per_width = imageW/blockDim.x;
  int z = (int)ceilf(blockIdx.x/n_blocks_per_width);
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  int x = (blockIdx.x - z*n_blocks_per_width)*blockDim.x + threadIdx.x;
  int pos = z*imageW*imageH + y*imageW + x;

  //preparation of the feature vectors
  float feature[21];
  float feature_tmp[21];
  float feature_rot[21];
  for(int nf = 0; nf < 21; nf++){
    feature[nf] = d_derivs[nf][pos];
    feature_rot[nf] = 0;
    feature_tmp[nf] = 0;
  }
  //***************************** rotation         ***************************//
  //The rotation is hard-coded in here
  float theta = -d_theta[pos];
  float phi   = -d_phi[pos];

  // inverse rotation in theta //
  float tmp1, tmp2, tmp3, ct, st, sp, cp;
  int m, n, p, i, j, k, l, q, b_v, ord, idx;

  for(int i = 0; i < 21; i++) feature_tmp[i] = 0;

  //In Aguet's formula the phi is defined in the opposite sense of the clock (towards x negative),
  // from there the inversion of phi
  ct = cos(theta);
  st = sin(theta);
  cp = cos(-phi);
  sp = sin(-phi);

  //For each order and each component of the order
  for(ord = 2; ord <=4; ord+=2){
    b_v = positionOrder(ord);
    for(idx = 0; idx < numDerivsOrder(ord); idx++){
      idx2deriv(idx, ord, m, n, p);
      for(i=0;i<=m;i++)
        for(k=0;k<=n;k++)
          for(q=0;q<=p;q++)
            for(j=0;j<=i;j++)
              for(l=0;l<=k;l++){
                if(j+l+p-q != 0)
                  continue;
                tmp1 = factorial_n(m)*factorial_n(n)*factorial_n(p)*pow(-1.0,i-j+p-q);
                tmp2 = factorial_n(m-i)*factorial_n(i-j)*factorial_n(j)*factorial_n(n-k)*
                  factorial_n(k-l)*factorial_n(l)*factorial_n(p-q)*factorial_n(q);
                tmp3 = ((double)tmp1)/tmp2;
                tmp3 = tmp3*pow(ct,m-i+j+k-l);
                tmp3 = tmp3*pow(st,i-j+n-k+l);
                tmp3 = tmp3*pow(sp,j+l+p-q);    //only if j+l+p-q is equal to 0
                feature_tmp[b_v + deriv2idx(m-i+n-k+p-q, i-j+k-l, j+l+q)] +=
                  tmp3*feature[b_v + deriv2idx(m,n,p)];
              }
    }
  }

  // inverse rotation in phi //
  for(int i = 0; i < 21; i++) feature_rot[i] = 0;
  //In Aguet's formula the phi is defined in the opposite sense of the clock (towards x negative),
  // from there the inversion of phi

  //For each order and each component of the order
  for(ord = 2; ord <=4; ord+=2){
    b_v = positionOrder(ord);
    for(idx = 0; idx < numDerivsOrder(ord); idx++){
      idx2deriv(idx, ord, m, n, p);
      for(i=0;i<=m;i++)
        for(k=0;k<=n;k++)
          for(q=0;q<=p;q++)
            for(j=0;j<=i;j++)
              for(l=0;l<=k;l++){
                if(i-j+n-k+l != 0)
                  continue;
                tmp1 = factorial_n(m)*factorial_n(n)*factorial_n(p)*pow(-1.0,i-j+p-q);
                tmp2 = factorial_n(m-i)*factorial_n(i-j)*factorial_n(j)*factorial_n(n-k)*
                  factorial_n(k-l)*factorial_n(l)*factorial_n(p-q)*factorial_n(q);
                tmp3 = ((double)tmp1)/tmp2;
                tmp3 = tmp3*pow(cp,m-i+n-k+q);
                tmp3 = tmp3*pow(sp,j+l+p-q);
                feature_rot[b_v + deriv2idx(m-i+n-k+p-q, i-j+k-l, j+l+q)] +=
                  tmp3*feature_tmp[b_v + deriv2idx(m,n,p)];
              }
    }
  }

  //Store the rotated features where the derivatives were
  for(int i = 0; i < 21; i++)
    d_derivs[i][pos] = feature_rot[i];
  // d_output[pos] = d_derivs[0][pos];
  // rotateFeatureVectorInverse(feature_rot, feature, feature_tmp,  d_theta[i], d_phi[i]);
}


__global__ void SVMResponse
(float*  d_output,
 float*  d_sv,
 float*  d_alphas,
 float   sk,
 int     n_sv,
 int     imageW,
 int     imageH
)
{
  int n_blocks_per_width = imageW/blockDim.x;
  int z = (int)ceilf(blockIdx.x/n_blocks_per_width);
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  int x = (blockIdx.x - z*n_blocks_per_width)*blockDim.x + threadIdx.x;
  int pos = z*imageW*imageH + y*imageW + x;

 float res=0;
 float expn = 0;
 // for(int isv=0;isv<n_sv;isv++){
 for(int isv=0; isv<100; isv++){
   expn = 0;
   for(int nd = 0; nd < 21; nd++)
     expn -= (d_derivs[nd][pos]-d_sv[isv*21 + nd])*(d_derivs[nd][pos]-d_sv[isv*21+nd]);
   res += d_alphas[isv]*exp(expn/(sk*sk));
 }

 d_output[pos] = res;

}


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
)
{
  dim3 gird (imageD*imageW/SVM_BLOCKDIM_X,imageH/SVM_BLOCKDIM_Y);
  dim3 block(SVM_BLOCKDIM_X,SVM_BLOCKDIM_Y);

  int nsv_step = 50;
  int nsv_i; int nsv_e;
  int isvblock = 0;
  // for(int isvblock = 0; isvblock < n_sv/nsv_step; isvblock++){
    // nsv_i = nsv_step*isvblock;
    // nsv_e = min(nsv_step*(isvblock+1), n_sv);
    // printf("   computing between sv: %i, %i\n", nsv_i, nsv_e);
  rtfSVMKernel<<<gird, block>>>
    (d_output,
     d_theta,
     d_phi,
     d_sv,
     d_alphas,
     sk,
     0,
     n_sv,
     imageW,
     imageH,
     imageD
     );
  cudaError err = cudaThreadSynchronize();
  if(cudaSuccess != err){
    printf("Error computing the SVM Inside - rtf: %s\n", cudaGetErrorString(err));
  }
  // }
  cutilCheckMsg("rtf3D() execution failed\n");

  SVMResponse<<<gird, block>>>
    (d_output,
     d_sv,
     d_alphas,
     sk,
     n_sv,
     imageW,
     imageH
     );
  err = cudaThreadSynchronize();
  if(cudaSuccess != err){
    printf("Error computing the SVM Inside - svm: %s\n", cudaGetErrorString(err));
  }



}

