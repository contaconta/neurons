#include <math.h>
#include <algorithm>
#include "mex.h"

// [err, pol] = ChooseThreshold_mex(F, W, L, TPOS, TNEG)


double* F = NULL;
double* W = NULL;
double* L = NULL;
//double* Lneg = NULL;
double TPOS = 0;
double TNEG = 0;
double N_examples = 0;

double SPOS = 0;
double SNEG = 0;
double fp_err_pos_pol = 0;
double fp_err_neg_pol = 0;
double fn_err_pos_pol = 0;
double fn_err_neg_pol = 0;
double pos_pol_err = 0;
double neg_pol_err = 0;

double* err = NULL;
double* pol = NULL;



void mexFunction(int nb_outputs,  mxArray* outputs[], 
				int nb_inputs, const mxArray* inputs[] ) 
{
    

if(nb_inputs != 5)
    mexErrMsgTxt("5 input arguments expected.");

    F = (double*) mxGetPr(inputs[0]);
    W = (double*) mxGetPr(inputs[1]);
    L = (double*) mxGetPr(inputs[2]);
    //Lneg = (double*) mxGetPr(inputs[3]);
    TPOS = (double) mxGetScalar(inputs[3]);
    TNEG = (double) mxGetScalar(inputs[3]);

    N_examples = mxGetDimensions(inputs[0])[0];

    SPOS = 0;
    SNEG = 0;
    fp_err_pos_pol = 0;
    fp_err_neg_pol = 0;
    fn_err_pos_pol = 0;
    fn_err_neg_pol = 0;
    pos_pol_err = 0;
    neg_pol_err = 0;
    
    // create the outputs
    outputs[0] = mxCreateDoubleMatrix(N_examples,1,mxREAL);
    outputs[1] = mxCreateDoubleMatrix(N_examples,1,mxREAL);
    //outputs[1] = mxCreateDoubleMatrix(N_examples,1,mxREAL);
    //T = mxGetPr(outputs[0]);
    err = mxGetPr(outputs[0]);
    pol = mxGetPr(outputs[1]);
    
    for (int n = 0; n < N_examples; n++){
        

        if((n > 0) && (F[n] == F[n-1])){
            err[n] = err[n-1];
            pol[n] = pol[n-1];
        }
        else{
            // positive polarity => + class < THRESH <= - class
            // negative polarity => - class < THRESH <= + class

            fp_err_pos_pol = TPOS - SPOS;
            fn_err_pos_pol = SNEG;
            
            pos_pol_err = fp_err_pos_pol + fn_err_pos_pol;
            
            
            if (pos_pol_err <= .5){
                err[n] = pos_pol_err;
                pol[n] = 1;
            }
            else{
                err[n] = 1-pos_pol_err;
                pol[n] = -1;
            } 
        }
        
         // the SPOS and SNEG count get updated after the threshold passes them
        if (L[n] == 1){
            SPOS = SPOS + W[n];
        }
        else{
            SNEG = SNEG + W[n];
        }
        
        //if(n < 10)
        //    mexPrintf("[SPOS %5f  SNEG %5f TPOS %3f TNEG %3f err %5f pol %1f]\n", SPOS, SNEG, TPOS, TNEG, err[n], pol[n]); 
    }
}