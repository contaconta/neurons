#include "mex.h"
#include "math.h"

/* Compute the normalized cross-correlatiom
 * This function uses Bessel's correction
 * (use of N − 1 instead of N when computing the variance
 * and the normalized cross-correlation)
 */
void mexFunction(int nlhs,       mxArray *plhs[],
		 int nrhs, const mxArray *prhs[])
{
    /* Declare variables */ 
    mwSize j;
    double n;
    mwSize number_of_dims;
    double *prI;
    double *prF;
    mwSize *dim_array;     
    
    /* Check for proper number of input and output arguments */    
    if (nrhs != 2) {
	mexErrMsgTxt("Two input arguments required.");
    } 
    if (nlhs > 1){
	mexErrMsgTxt("Too many output arguments.");
    }
    
    /* Check data type of input argument */
    if (!(mxIsDouble(prhs[0]))) {
      mexErrMsgTxt("First input array must be of type double.");
    }
    if (!(mxIsDouble(prhs[1]))) {
      mexErrMsgTxt("Second input array must be of type double.");
    }
   
    /* Get the data */
    prI=(double *)mxGetPr(prhs[0]);
    prF=(double *)mxGetPr(prhs[1]);
    n=(double)mxGetNumberOfElements(prhs[0]);

    /* Compute means */
    double avgI = 0;
    double avgF = 0;
    for(j=0;j<n;j++){
      avgI += prI[j];
      avgF += prF[j];
    }
    avgI /= n;
    avgF /= n;
	
    /* Compute stds */
    double temp;
    double stdI = 0;
    double stdF = 0;
    for(j=0;j<n;j++) {
      temp = prI[j] - avgI;
      stdI += temp*temp;

      temp = prF[j] - avgF;
      stdF += temp*temp;
    }
    stdI = sqrt(stdI/(n-1));
    stdF = sqrt(stdF/(n-1));

    /* Compute the normalized cross-correlation */
    double N = 0;
    for(j=0;j<n;j++){
      N += (prI[j] - avgI)*(prF[j]-avgF) / (stdI*stdF);
    }
    N = N/(n-1);
    plhs[0]=mxCreateDoubleScalar(N);
}
