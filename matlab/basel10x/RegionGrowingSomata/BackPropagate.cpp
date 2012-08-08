#include "fm.h"

#define kDead -1
#define kOpen -2
#define kFar -3
#define kBorder -4


/* Global variables */
int nx;			// real size on X
int ny;			// real size on Y
int Nx, Ny; // size for computing
int size;

double* POINTS  = NULL;
int* END_POINTS = NULL;
double*  UU      = NULL;
double*  U       = NULL;
bool* Filaments = NULL;

unsigned int number_of_points = 0;

unsigned int connectivity_large = 0;
int* NeighborhoodLarge = NULL;


//================================================================
void InitializeNeighborhoods()
//================================================================
{
    connectivity_large = 8;
    NeighborhoodLarge = (int*) mxCalloc(connectivity_large, sizeof(int));
    if(NeighborhoodLarge == NULL)
        mexErrMsgTxt("Bad memory allocation NeighborhoodLarge");
	NeighborhoodLarge[ 0]= -1-Nx;
	NeighborhoodLarge[ 1]= -1+Nx;
	NeighborhoodLarge[ 2]= -1   ;
	NeighborhoodLarge[ 3]= +1-Nx;
    NeighborhoodLarge[ 4]= +1+Nx;
    NeighborhoodLarge[ 5]= +1   ;
    NeighborhoodLarge[ 6]=   -Nx;
    NeighborhoodLarge[ 7]=   +Nx;
};

//================================================================
void InitializeArrays()
//================================================================
{
	int x, y, point;
	//copy the weight list and initialize
	U = (double*) mxCalloc(size, sizeof(double));
	if(U == NULL)
    {
        mexErrMsgTxt("Bad memory allocation U");
    }
    //------------------------------------------------------------
    for(point = 0; point < size; point++)
    {
		U[point] = INFINITE;
        Filaments[point] = false;
    }
    //------------------------------------------------------------
    for(x = 0; x < nx; x++)
    {
		for(y = 0; y < ny; y++)
        {
			point = (x+1) + (y+1)*Nx;
			U[point] = UU[x + y*nx];
		}
	}
    //------------------------------------------------------------
    END_POINTS = (int*) mxCalloc(number_of_points, sizeof(int));
    if(END_POINTS == NULL)
    {
        mexErrMsgTxt("Bad memory allocation END_POINTS");
    }
    for( int s=0; s<number_of_points; s++ )
    {
		x = (int) round(POINTS[2*s]);
		y = (int) round(POINTS[1+2*s]);
		END_POINTS[s] = x + y*Nx;
    }
};

//================================================================
void BackPropagate()
//================================================================
{
    double Umin;
    int k, j, point, npoint, npoint_Umin;
    bool is_smallerNei_found = false;
    
    for(k = 0; k < number_of_points; k++){
        point = END_POINTS[k];
        Filaments[point] = true;
        while(U[point] > 0){
            Umin = U[point];
            npoint_Umin = point;
            is_smallerNei_found = false;
            for (j = 0; j < connectivity_large; j++){
                npoint=point+NeighborhoodLarge[j];
                if ( U[npoint] < Umin )
                {
                    Umin = U[npoint];
                    npoint_Umin = npoint;
                    is_smallerNei_found = true;
                }
            }
            if(!is_smallerNei_found){
                mexPrintf("Point different of the source and no smaller neighbor, hell!!!! \n");
                break;
            }
            point = npoint_Umin;
            Filaments[point] = true;
        }
    }
};


//================================================================
void resize()
//================================================================
{
    int x, y, point, Point;
    for(y=0;y<ny;y++)
        for(x=0;x<nx;x++){
            point = x+y*nx;
            Point = (x+1)+(y+1)*Nx;
            Filaments[point] = U[Point];
        }
};

//================================================================
void mexFunction(	int nlhs, mxArray *plhs[], 
					int nrhs, const mxArray*prhs[] ) 
//================================================================                    
{
	//------------------------------------------------------------------
	/* retrive arguments */
	//------------------------------------------------------------------
	if( (nrhs!=2)  )
    {
		mexErrMsgTxt("2 arguments are required.");
    }
	//------------------------------------------------------------------
	// first argument : Set of points
    if( (mxGetClassID(prhs[0]) != mxDOUBLE_CLASS) || (mxGetNumberOfDimensions(prhs[0])!= 2) 
		||    mxGetDimensions(prhs[0])[0] != 2 ) 
	{
			mexErrMsgTxt("tforth input shoud be an array containing the [mean_i, std_i] intensities of the nuclei regions") ;
	}
    
    POINTS = mxGetPr(prhs[0]);
    number_of_points = mxGetDimensions(prhs[0])[1];
	//------------------------------------------------------------------
	// Second argument : Distance map for the back propagation
    if( (mxGetClassID(prhs[1]) != mxDOUBLE_CLASS) || (mxGetNumberOfDimensions(prhs[1])!= 2) ) 
    {
        mexErrMsgTxt("Distance map for the back propagation must be a 2D double array AND \n must be of class double") ;
    }
    
	U = (double*)mxGetPr(prhs[1]);
    nx = mxGetDimensions(prhs[1])[0];
	ny = mxGetDimensions(prhs[1])[1];
	Nx = nx+2; Ny = ny+2;
	size = Nx*Ny;
	//------------------------------------------------------------------
    //==================================================================
	// Outputs
	mwSize dims[2] = {Nx,Ny};
	// First output : minimal action map
	plhs[0] = mxCreateNumericArray(2, dims, mxLOGICAL_CLASS, mxREAL );
	Filaments = (bool*) mxGetPr(plhs[0]);
	//==================================================================
	InitializeNeighborhoods();
	//------------------------------------------------------------------
	InitializeArrays();
	//------------------------------------------------------------------
	BackPropagate();
	//==================================================================
	resize();
	dims[0] = Nx-2; dims[1] = Ny-2;
	mxSetDimensions(plhs[0], dims, 2);
	//==================================================================
    mxFree(U);
    mxFree(NeighborhoodLarge);
    mxFree(END_POINTS);
    return;
}