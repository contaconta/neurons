#include "fm.h"

int Nx, Ny, NxNy;
short halfWindow;
short Window, Window2;
double sigma  = 1.0;
double sigma2 = 1.0;
double* kernelGxx = NULL;
double* kernelGyy = NULL;
double* kernelGxy = NULL;
double* Image = NULL;
double* Hessian = NULL;
double* H = NULL;
double hx, hy, hx2, hy2, hxhy;
double* Radius = NULL;
int N_Radius;

//-------------------------------------------------------------------------
double Gxx(double x, double y)
//-------------------------------------------------------------------------
{
    return (1/hx2) * ((x*x-sigma2) / (sigma2*sigma2)) * exp(-(x*x+y*y)/(2*sigma2));
};

//-------------------------------------------------------------------------
double Gyy(double x, double y)
//-------------------------------------------------------------------------
{
    return (1/hy2) * ((y*y-sigma2) / (sigma2*sigma2)) * exp(-(x*x+y*y)/(2*sigma2));
};

//-------------------------------------------------------------------------
double Gxy(double x, double y)
//-------------------------------------------------------------------------
{
    return (1/hxhy) * ((x*y) / (sigma2*sigma2)) * exp(-(x*x+y*y)/(2*sigma2));
};

//-------------------------------------------------------------------------
void InitializeArrays()
//-------------------------------------------------------------------------
{
    int i, j, pos;
    double Normalize = 0.0;
    Window = 2*halfWindow + 1; Window2 = Window*Window;
    H         = new double[3*NxNy];
    kernelGxx = new double[Window2];
    kernelGyy = new double[Window2];
    kernelGxy = new double[Window2];
    for(i = -halfWindow; i <= halfWindow; i++){
        for(j = -halfWindow; j <= halfWindow; j++){
            pos = i+halfWindow + (j+halfWindow)*Window;
            kernelGxx[pos] = Gxx((double)i*hx,(double)j*hy);
            kernelGyy[pos] = Gyy((double)i*hx,(double)j*hy);
            kernelGxy[pos] = Gxy((double)i*hx,(double)j*hy);
            Normalize += kernelGxx[pos];
        }
    }
    //---------------------------------------------------------------------
	if(Normalize != 0){
        for(i = -halfWindow; i <= halfWindow; i++){
            for(j = -halfWindow; j <= halfWindow; j++){
                pos = i+halfWindow + (j+halfWindow)*Window;
                kernelGxx[pos] -= (Normalize/(double)Window2);
                kernelGyy[pos] -= (Normalize/(double)Window2);
            }
        }
    }
};

//-------------------------------------------------------------------------
void Compute_Hessian_Matrix()
//-------------------------------------------------------------------------
{
    int x, y, dx, dy, xk, yk, point, Point, pos, i;
    double vk, dist;
    double fxx, fyy, fxy;
    for(x=0; x < Nx; x++)
    for(y=0; y < Ny; y++){
        fxx = fyy = fxy = 0.0;
        for(dx = -halfWindow; dx <= halfWindow; dx++){
        	for(dy = -halfWindow; dy <= halfWindow; dy++){
                xk = x - dx;
                yk = y - dy;
                if(xk <     0 ) xk = 0;
                if(xk > (Nx-1)) xk = Nx-1;
                if(yk <     0 ) yk = 0;
                if(yk > (Ny-1)) yk = Ny-1;
                point = xk + yk*Nx;
                pos = dx+halfWindow + (dy+halfWindow)*Window;
                vk = Image[point];
                fxx += kernelGxx[pos] * vk;
                fyy += kernelGyy[pos] * vk;
                fxy += kernelGxy[pos] * vk;
            }
        }
        point = x + y*Nx;
        H[point         ] = fxx;
        H[point +   NxNy] = fyy;
        H[point + 2*NxNy] = fxy;
    }
    for(i = 0; i < N_Radius; i++){
        for(x=0; x < Nx; x++){
            for(y=0; y < Ny; y++){
                Point = x +y*Nx;
                Hessian[Point          + i*3*NxNy] = 0.0;
                Hessian[Point +   NxNy + i*3*NxNy] = 0.0;
                Hessian[Point + 2*NxNy + i*3*NxNy] = 0.0;
                int Rmax = floor(Radius[i]) + 1;
                for(dx = -Rmax; dx <= Rmax; dx++){
                    for(dy = -Rmax; dy <= Rmax; dy++){
                        xk = x - dx;
                        yk = y - dy;
                        if(xk <     0 ) xk = 0;
                        if(xk > (Nx-1)) xk = Nx-1;
                        if(yk <     0 ) yk = 0;
                        if(yk > (Ny-1)) yk = Ny-1;
                        point = xk + yk*Nx;
                        dist = sqrt(dx*dx+dy*dy);
                        if(dist <= Radius[i]){
                            Hessian[Point          + i*3*NxNy] += H[point         ];
                            Hessian[Point +   NxNy + i*3*NxNy] += H[point +   NxNy];
                            Hessian[Point + 2*NxNy + i*3*NxNy] += H[point + 2*NxNy];
                        }
                    }
                }
                Hessian[Point          + i*3*NxNy] /= Radius[i];
                Hessian[Point +   NxNy + i*3*NxNy] /= Radius[i];
                Hessian[Point + 2*NxNy + i*3*NxNy] /= Radius[i];
            }
        }
    }
};


//-------------------------------------------------------------------------
void mexFunction(	int nlhs, mxArray *plhs[], 
					int nrhs, const mxArray*prhs[] ) 
//-------------------------------------------------------------------------
{
    //------------------------------------------------------------------
	/* retrive arguments */
	//------------------------------------------------------------------
	if( nrhs!=3 )
		mexErrMsgTxt("3 arguments are required.");
	if( mxGetNumberOfDimensions(prhs[1])!= 2 )
		mexErrMsgTxt("Image must be a 2D double array.");
	//------------------------------------------------------------------
	// first argument : spacing and dimensions
	const int* dim_h = mxGetDimensions(prhs[0]);
    if ( (dim_h[0]!=2) || (dim_h[1]!=1) )
	  mexErrMsgTxt("Library error: h must be a 2x1 array list.");
	hx = mxGetPr(prhs[0])[0]; hy = mxGetPr(prhs[0])[1];
	hx2 = hx*hx; hy2 = hy*hy;
    hxhy = hx*hy;
	Nx = mxGetDimensions(prhs[1])[0];
	Ny = mxGetDimensions(prhs[1])[1];
    NxNy = Nx*Ny;
	//------------------------------------------------------------------
	// Second argument : Image
	Image = (double*) mxGetPr(prhs[1]);
	//------------------------------------------------------------------
	// Third argument : Set of scales
	if(mxGetDimensions(prhs[2])[0] != 1)
        mexErrMsgTxt("The input radii list must be a 1xN array list.");
    N_Radius = mxGetDimensions(prhs[2])[1];
    Radius = (double*) mxGetPr(prhs[2]);
	//------------------------------------------------------------------
	// Fourth argument : the tolerance gives the window size
	halfWindow = 3;//floor(sigma*sqrt(-2*log(epsilon))) + 1;
    //==================================================================
	// Outputs
	int dims[4] = {Nx,Ny,3, N_Radius};
	//------------------------------------------------------------------
	// First output : Matrice Hessienne
	plhs[0] = mxCreateNumericArray(4, dims, mxDOUBLE_CLASS, mxREAL );
	Hessian = (double*) mxGetPr(plhs[0]);
	//==================================================================
	InitializeArrays();
	//------------------------------------------------------------------
	Compute_Hessian_Matrix();
    //==================================================================
    DELETEARRAY(kernelGxx); DELETEARRAY(kernelGyy); DELETEARRAY(kernelGxy);
    DELETEARRAY(H);
	return;
};