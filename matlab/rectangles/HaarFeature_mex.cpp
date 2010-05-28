#include <math.h>
#include <algorithm>
#include "mex.h"

#define NDIMS 2

mxArray* rects = NULL;
mxArray* cols = NULL;
double* pol = NULL;
double* pr = NULL;
double* pc = NULL;
float* D = NULL;
int nb_learners;
int nb_examples;
double fval = NULL;

//double* F = NULL;
float* F = NULL;
int rect_dim;



void mexFunction(int nb_outputs,  mxArray* outputs[], 
				int nb_inputs, const mxArray* inputs[] ) 
{
    
    //=====================================================================
    if(nb_inputs != 3)
        mexErrMsgTxt("3 input argument required, usage : blablablabalabla");
    
    //=====================================================================
    // read the cell inputs
    //=====================================================================
    rects = (mxArray*) inputs[1];
    cols  = (mxArray*) inputs[2];
    
    //=====================================================================
    // read the array inputs
    //=====================================================================      
    pol     = (double*) mxGetPr(inputs[1]);
    D       = (float*) mxGetPr(inputs[0]);
    //=====================================================================
    
    // get the # of learners
    nb_learners = mxGetDimensions(inputs[1])[1];
    nb_examples = mxGetDimensions(inputs[0])[0];
    //mexPrintf("number of learners = %d\n", nb_learners);
    //mexPrintf("number of examples = %d\n", nb_examples);
    
    // allocate the output
    //outputs[0]=mxCreateDoubleMatrix(nb_examples, nb_learners, mxREAL);
    const mwSize dims[]={nb_examples,nb_learners};
    outputs[0]=mxCreateNumericArray(NDIMS, dims, mxSINGLE_CLASS, mxREAL);
    F = (float*) mxGetPr(outputs[0]);
    
    // loop through each example, compute the classifier response
    for(int i = 0; i < nb_examples; i++){
    //for(int i = 0; i < 10; i++){
                
        // loop through the weak learners
        for(int k = 0; k < nb_learners; k++){
        //for(int k = 0; k < 10; k++){
            mxArray* RectCell = mxGetCell(rects, k);
            mxArray* ColsA    = mxGetCell(cols, k);
            
            double response = 0;
            
            pc=(double *)mxGetPr(ColsA);
            int nb_rects = mxGetDimensions(RectCell)[1];
            //mexPrintf("%d\n", nb_rects);
            
            for(int j = 0; j < nb_rects; j++){
                mxArray* RectA = mxGetCell(RectCell,j);

                //int rect_dim = mxGetDimensions(RectA)[1];
                pr=(double *)mxGetPr(RectA);

                // print the data we want to access
                //mexPrintf("[%3.0f %3.0f %3.0f %3.0f]", D[nb_examples * ((int)pr[0]-1) + i], D[nb_examples*((int)pr[1]-1) + i],D[nb_examples*((int)pr[2]-1)+i], D[nb_examples*((int)pr[3]-1)+i]);
                //mexPrintf("[%d %d %d %d]", nb_examples*i + (int)pr[0], nb_examples*i + (int)pr[1],nb_examples*i + (int)pr[2], nb_examples*i + (int)pr[3]);
                
                fval = pc[j]*D[nb_examples * ((int)pr[0]-1) + i] - pc[j]*D[nb_examples*((int)pr[1]-1) + i]  - pc[j]*D[nb_examples*((int)pr[2]-1)+i] + pc[j]*D[nb_examples*((int)pr[3]-1)+i];
                //fval = 1.0;
                //mexPrintf("col %3.0f => %3.0f  ", pc[j], fval);
                
                //mexPrintf("[%3.0f %3.0f %3.0f %3.0f] ", pr[0],pr[1],pr[2],pr[3]);        
                //mexPrintf("[%3.0f] ", pc[j]);
                
                response = response + fval;
            }
            
            //mexPrintf("response = %3.0f  thresh = %3.0f  pol = %3.0f\n", response, thresh[k], pol[k]);
            //mexPrintf("%3.0f ", response);
            //mexPrintf("%d,%d ", i,k);
            
            // store the weak responses for each example
            F[ nb_examples*k + i] = (float) response;
            
        }
        //mexPrintf("\n");
        //int D_r = mxGetDimensions(inputs[5])[0];
        //int D_c = mxGetDimensions(inputs[5])[1];
        //mexPrintf("%d %d\n", D_r, D_c);
        //mexPrintf("%d %3.0f \n", nb_examples *110 + i ,  D[nb_examples*110 + i]);  // gets column 110
    
        
    }
};