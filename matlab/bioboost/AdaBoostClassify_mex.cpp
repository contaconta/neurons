#include <math.h>
#include <algorithm>
#include "mex.h"
#include <iostream>



mxArray* inds = NULL;
double* thresh = NULL;
double* pol = NULL;
double* alpha = NULL;
double* pi = NULL;
float* D = NULL;
int nb_learners;
int nb_examples;

double* o_ind = NULL;
int rect_dim;



void mexFunction(int nb_outputs,  mxArray* outputs[], 
				int nb_inputs, const mxArray* inputs[] ) 
{
    
    //=====================================================================
    if(nb_inputs != 5)
        mexErrMsgTxt("5 input argument required, usage : blablablabalabla");
    
    //=====================================================================
    // read the cell inputs
    //=====================================================================
    inds = (mxArray*) inputs[0];

    //=====================================================================
    // read the array inputs
    //=====================================================================      
    thresh  = (double*) mxGetPr(inputs[1]);
    pol     = (double*) mxGetPr(inputs[2]);
    alpha   = (double*) mxGetPr(inputs[3]);
    D       = (float*) mxGetPr(inputs[4]);
    //=====================================================================
    
    // get the # of learners
    nb_learners = mxGetDimensions(inputs[0])[1];
    nb_examples = mxGetDimensions(inputs[4])[0];
    //std::cout << "hi0" << std::flush;
    //mexPrintf("number of learners = %d\n", nb_learners);
    //mexPrintf("number of examples = %d\n", nb_examples);
    
    // allocate the output
    outputs[0]=mxCreateDoubleMatrix(nb_examples, 1, mxREAL);
    o_ind = mxGetPr(outputs[0]);
    
    // loop through each example, compute the classifier response
    for(int i = 0; i < nb_examples; i++){
        
        double strong_response = 0;
        
        //if (i==1)
        //    mexPrintf("learner: ");
        
        // loop through the weak learners
        for(int k = 0; k < nb_learners; k++){
            mxArray* IndArray = mxGetCell(inds, k);

            double response = 0;
            double f1 = 0;
            double f0 = 0;
            
            pi=(double *)mxGetPr(IndArray);
            
            
           
            response = (double) D[nb_examples * ((int)pi[0]-1) + i];
                      
            //if (i==1)
            //    mexPrintf("%d ", k);

            int weak_classification = 0;
            if (pol[k] * response < pol[k] * thresh[k]){
                weak_classification = 1;
            }
            else{
                weak_classification = -1;
            }

            // compute the strong response
            strong_response = strong_response + alpha[k]*weak_classification;
            
        }
        o_ind[i] = strong_response;
    
        //if (i==1)
        //    mexPrintf("\nstrong response = %3.0f\n",strong_response);
    }
};