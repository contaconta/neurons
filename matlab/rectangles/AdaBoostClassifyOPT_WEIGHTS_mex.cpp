#include <math.h>
#include <algorithm>
#include "mex.h"
#include <iostream>



mxArray* rects = NULL;
mxArray* weights = NULL;
mxArray* areas = NULL;
double* thresh = NULL;
double* pol = NULL;
double* alpha = NULL;
double* pr = NULL;
double* pc = NULL;
double* pa = NULL;
float* D = NULL;
int nb_learners;
int nb_examples;
double fval = 0;

double* o_ind = NULL;
int rect_dim;

//AdaBoostClassifyDynamicA_mex(f_rects, f_cols, f_areas,thresh, alpha, pol, DATA);

void mexFunction(int nb_outputs,  mxArray* outputs[], 
				int nb_inputs, const mxArray* inputs[] ) 
{
    //std::cout << "hi0" << std::flush;
    //mexPrintf("hi0");
    //=====================================================================
    if(nb_inputs != 7)
        mexErrMsgTxt("7 input argument required, usage : blablablabalabla");
    
    //=====================================================================
    // read the cell inputs
    //=====================================================================
    rects = (mxArray*) inputs[0];
    weights  = (mxArray*) inputs[1];
    areas = (mxArray*) inputs[2];
    
    //=====================================================================
    // read the array inputs
    //=====================================================================      
    thresh  = (double*) mxGetPr(inputs[3]);
    pol     = (double*) mxGetPr(inputs[4]);
    alpha   = (double*) mxGetPr(inputs[5]);
    D       = (float*) mxGetPr(inputs[6]);
    //=====================================================================
    
	//std::cout << "hi1" << std::flush;
    // get the # of learners
    nb_learners = mxGetDimensions(inputs[0])[1];
    nb_examples = mxGetDimensions(inputs[6])[0];
    //mexPrintf("number of learners = %d\n", nb_learners);
    //mexPrintf("number of examples = %d\n", nb_examples);
    //std::cout << "hi1.1" << std::flush;
    // allocate the output
    outputs[0]=mxCreateDoubleMatrix(nb_examples, 1, mxREAL);
    o_ind = mxGetPr(outputs[0]);
    //std::cout << "hi1.2" << std::flush;
    // loop through each example, compute the classifier response
    for(int i = 0; i < nb_examples; i++){
    //for(int i = 0; i < 4; i++){
        
        double strong_response = 0;
        
        // loop through the weak learners
        for(int k = 0; k < nb_learners; k++){
        //for(int k = 0; k < 4; k++){

            mxArray* RectCell = mxGetCell(rects, k);
            mxArray* ColsA    = mxGetCell(weights, k);
            mxArray* AreaArray= mxGetCell(areas,k);
            //std::cout << "hi1.3" << std::flush;
            double response = 0;
            double f1 = 0;
            double f0 = 0;
            
            pc=(double *)mxGetPr(ColsA);
//std::cout << "hi1.3.1" << std::flush;
            pa=(double *)mxGetPr(AreaArray);
//std::cout << "hi1.3.2" << std::flush;
            int nb_rects = mxGetDimensions(RectCell)[1];
            //mexPrintf("%d\n", nb_rects);
            //std::cout << "hi1.4" << std::flush;
            //double area1 = pa[0];
            //double area0 = pa[1];
            
            for(int j = 0; j < nb_rects; j++){
                mxArray* RectA = mxGetCell(RectCell,j);
			//std::cout << "hi1.5" << std::flush;
                //int rect_dim = mxGetDimensions(RectA)[1];
                pr=(double *)mxGetPr(RectA);

                // print the data we want to access
                //mexPrintf("[%3.0f %3.0f %3.0f %3.0f]", D[nb_examples * ((int)pr[0]-1) + i], D[nb_examples*((int)pr[1]-1) + i],D[nb_examples*((int)pr[2]-1)+i], D[nb_examples*((int)pr[3]-1)+i]);
                //mexPrintf("[%d %d %d %d]", nb_examples*i + (int)pr[0], nb_examples*i + (int)pr[1],nb_examples*i + (int)pr[2], nb_examples*i + (int)pr[3]);
                //std::cout << "hi2" << std::flush;
                
                fval = pc[j]*D[nb_examples * ((int)pr[0]-1) + i] - pc[j]*D[nb_examples*((int)pr[1]-1) + i]  - pc[j]*D[nb_examples*((int)pr[2]-1)+i] + pc[j]*D[nb_examples*((int)pr[3]-1)+i];
               //std::cout << "hi3" << std::flush;
                fval = fval / pa[j];
                //std::cout << "hi4" << std::flush;
                
                //mexPrintf("col %3.0f => %3.0f  ", pc[j], fval);
                
                
                //mexPrintf("[%3.0f %3.0f %3.0f %3.0f] ", pr[0],pr[1],pr[2],pr[3]);        
                //mexPrintf("[%3.0f] ", pc[j]);

                // col * [1 + 4 - 2 -3]
                response = response + fval;
                //std::cout << "hi5" << std::flush;
            }
            
            // get the areas

            
            int weak_classification = 0;
            if (pol[k] == 1){
                if (response < thresh[k]){
                    weak_classification = 1;
                }else{
                    weak_classification = -1;
                }
            }else{
                if (response < thresh[k]){
                    weak_classification = -1;
                }else{
                    weak_classification = 1;
                }
            }
            
            
            // compute the strong response
            strong_response = strong_response + pol[k]*alpha[k]*weak_classification;
            
        }
        //int D_r = mxGetDimensions(inputs[6])[0];
        //int D_c = mxGetDimensions(inputs[6])[1];
        //mexPrintf("%d %d\n", D_r, D_c);
        //mexPrintf("%d %3.0f \n", nb_examples *110 + i ,  D[nb_examples*110 + i]);  // gets column 110
        //mexPrintf("\n");
        o_ind[i] = strong_response;
        //o_ind[i] = 10;
    
    }
};
