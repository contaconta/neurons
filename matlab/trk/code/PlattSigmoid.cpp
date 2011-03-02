#include "mex.h"
#include <vector>
#include <math.h>
#include <omp.h>


/** Function Platt Sigmoid
 *
 * Fits a sigmoid to the training data to produce probabilistic outputs. 
 * Using the model-trust minimization algorithm.
 *
 * s(x) = 1 / (1 + exp(Ax + B))
 *
 * See: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.41.1639
 *
 * Input: vector of classifier values
 *        vector of classifier types 
 *               (assumed to be 0 for negative, 1 for positive)
 *        
 * Output: A - first parameter of the sigmoid
 *         B - second parameter of the sigmoid 
 *
 *
 * Compile with the following command to enable openmp:
 *  mex PlattSigmoid.cpp CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"
 *
 *
 * autor: German Gonzalez
 */

using namespace std;

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{

    // Parses the arguments
    
  if(nrhs!=2) {
    mexErrMsgTxt("Two inputs required.");
  }
  
  double *out,*target, *Ar, *Br;
  mwSize nrows,ncols, nrows1, ncols1, nrows2, ncols2;
  
    /* The input must be a noncomplex scalar double.*/
  nrows1 = mxGetM(prhs[0]);
  ncols1 = mxGetN(prhs[0]);
  if( !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) ) {
    mexErrMsgTxt("Input 1 must be a noncomplex scalar double.");
  }
  
  nrows2 = mxGetM(prhs[1]);
  ncols2 = mxGetN(prhs[1]);
  if( !mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) ) {
    mexErrMsgTxt("Input 2 must be a noncomplex scalar double.");
  }
  
  if((nrows1!=nrows2)||(ncols1!=ncols2))
    mexErrMsgTxt("The samples and the labels should have the same size");
  
  nrows = nrows1;
  ncols = ncols1;
  
  mexPrintf("Arguments correct\n");
  
  /* Create matrix for the return argument. */
  plhs[0] = mxCreateDoubleMatrix(1,1, mxREAL); //A
  plhs[1] = mxCreateDoubleMatrix(1,1, mxREAL); //B
  
  
  /* Assign pointers to each input and output. */
  out = mxGetPr(prhs[0]);
  target = mxGetPr(prhs[1]);
  Ar     = mxGetPr(plhs[0]);
  Br     = mxGetPr(plhs[1]);

  /** Now we start the algorithm*/
  long length = nrows*ncols;
  long prior0 = 0;
  long prior1 = 0;
  
  double max_val = -1000;
  double min_val = 1000;
  for(long i = 0; i < length; i++){
      if(target[i]>0.5)
          prior0++;
      else prior1++;
      if(out[i] > max_val)
            max_val = out[i];
     if(out[i] < min_val)
            min_val = out[i];

  }
//   mexPrintf("The maximum and minmun value of the data are [%e, %e]\n",
//             max_val, min_val);
//   
  mexPrintf("The number of positive sampels is %i and negatives %i\n",
            prior0, prior1);
  
  
  int nThreads = 1;
  nThreads = omp_get_max_threads();
  
  
  double A = 0;
  double B = log(double(prior0+1)/double(prior1+1));
  double hiTarget = double(prior1+1) / double(prior1+2);
  double loTarget = double(1.0)/ double(prior0+2);
  double lambda   = 0.001;
  double olderr   = 1e300;
  
  double tmp = double(prior1+1)/double(prior0+prior1+2);
  vector<double> pp(length);
  for(long i = 0; i < length; i++)
    pp[i] = tmp;
  
  int count = 0;
  
  /** Iteration loop*/
  for(int it = 0; it < 100; it++){

    mexPrintf("it=%i, A=%f, B=%f\n", it, A, B);
    double a,b,c,d,e;
    vector< double > aa(nThreads);
    vector< double > bb(nThreads);
    vector< double > cc(nThreads);
    vector< double > dd(nThreads);
    vector< double > ee(nThreads);
    for(int i = 0; i < nThreads; i++){
        aa[i] = 0; bb[i] = 0; cc[i] = 0;
        dd[i] = 0; ee[i] = 0;
    }
    
    a = 0; b = 0; c = 0; d = 0; e = 0;
    
    // First compute the hessian and the gradient with respect to A&B
#pragma omp parallel for
    for(long i = 0; i < length; i++){
        int tn = omp_get_thread_num();
        double t, d1, d2;
        if(target[i] > 0.5) t = hiTarget;
        else                t = loTarget ;
        
        d1 = pp[i] - t;
        d2 = pp[i]*(1-pp[i]);
        aa[tn] += out[i]*out[i]*d2;
        bb[tn] += d2;
        cc[tn] += out[i]*d2;
        dd[tn] += out[i]*d1;
        ee[tn] += d1;
    }
    for(int i = 0; i < nThreads; i++){
        a+=aa[i]; b+=bb[i]; c+=cc[i]; d+=dd[i]; e+=ee[i];
    }
    
    double oldA = A;
    double oldB = B;
    double err  = 0;
    
//     mexPrintf("it=%i, a=%e, b=%e, c=%e d=%e, e=%e, oldA=%e, oldB=%e\n",
//                 it, a, b, c, d, e, oldA, oldB);
//     mexPrintf("highTarget=%e, lowTarget=%e, oldA=%e, oldB=%e\n",
//                hiTarget, loTarget, oldA, oldB);

    
    // If the gradient is too small, stop
    if( (abs(d) < 1e-9) && (abs(e) < 1e-9)){
//         mexPrintf("The gradient is too small: d=%e, e=%e\n", d, e);
        break;    
    }
    
    // Loop until goodness of fit increases
    while(1){
      double det = (a+lambda)*(b+lambda)-c*c;
      //if the determinat of the hessian is 0, increase the stabilizer
      if( det == 0){ 
            lambda *= 10;
            continue;
      }
      A = oldA + ((b+lambda)*d - c*e)/det;
      B = oldB + ((a+lambda)*e - c*d)/det;
      
      // Now compute the goodness of the fit
      err = 0;
      vector< double > errors(nThreads);
      for(int i = 0; i < nThreads; i++)
          errors[i] = 0;

#pragma omp parallel for
      for (long i = 0; i < length; i++){
         int tn = omp_get_thread_num();
         double t;
         if(target[i] > 0.5) t = hiTarget;
         else                t = loTarget ; 
         
         double p, lp, lnp;         
         p = 1/(1+exp(out[i]*(A)+B));
         pp[i] = p;
         lp = log(p);
         lnp = log(1-p);
         if(isnan(lp) || isinf(lp) || (lp < -200) )
             lp = -200;
         if(isnan(lnp) || isinf(lnp) || (lnp < -200) )
             lnp = -200;
         errors[tn] -= t*lp + (1-t)*lnp;
      }
      for(int i = 0; i < nThreads; i++)
           err += errors[i];

      if(err < olderr*(1+1e-7)){
            lambda *= 0.1;
            break;
      }
      
      lambda *= 10;
      if (lambda >= 1e6){
          mexErrMsgTxt("Something is broken!");
          break;          
      }
    } // while loop
    
    double diff = err - olderr;
    double scale = 0.5*(err + olderr + 1);
    if( diff > -1e-3 * scale && diff < 1e-7*scale)
        count ++;
    else
        count = 0;
    olderr = err;
    if (count == 3)
        break;
  }// for i = 1:1:100
  
  Ar[0] = A;
  Br[0] = B;
    
    
      
}
