//**********************************************************
//Copyright 2012 Fethallah Benmansour
//
//Licensed under the Apache License, Version 2.0 (the "License"); 
//you may not use this file except in compliance with the License. 
//You may obtain a copy of the License at
//
//http://www.apache.org/licenses/LICENSE-2.0 
//
//Unless required by applicable law or agreed to in writing, software 
//distributed under the License is distributed on an "AS IS" BASIS, 
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
//See the License for the specific language governing permissions and 
//limitations under the License.
//**********************************************************

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
double hx, hy;// spacing
double hx2, hy2;
double hx2hy2;
double hx2_plus_hy2;
double* U = NULL;// action map
double* L = NULL; // distance map
short* S = NULL; // states
short* V = NULL; // voronoi
double* W = NULL; // potential
double* WW = NULL;
double* Nuclei = NULL;

double* mean_std = NULL;
unsigned int number_of_regions = 0;
double multFactor = 0;
double meanGlobalInt = 0;
double stdGlobalInt = 0;

int   connectivity_small;
int*    NeighborhoodSmall = NULL;

// min-heap
int*	  Tree;
int*	  Trial;
int*	  PtrToTrial;

//================================================================
// MIN-HEAP
//================================================================

//================================================================
int Tree_GetFather(int position)
//================================================================
{
    if(position)
    {
        if (ceil((double)position/2)==((double)position/2))
            return (position/2 -1);
        else
            return((position-1)/2);
    }
    else
        return -1;
};

//================================================================
// Tree_PushIn
/*
 * COMMENTS :
 * The tree is updated, since the value at the bottom is no longer
 * necessarily greater than the value of its father. Starting from
 * the bottom of the tree (the node we have just pushed in), the
 * value of each node is compared with the value of its father. If
 * the value of the father is greater, the two nodes are permuted.
 * The process stops when the top of the tree has been reached.
 */
//================================================================
int Tree_PushIn(int NewPosition)
{
    *(++PtrToTrial) = NewPosition;
    int position = (int)(PtrToTrial - Trial);
    Tree[NewPosition] = position;
    int s_idx = Trial[position];
    int f_idx = Trial[Tree_GetFather(position)];
    while((position!=0)&&(U[s_idx]<U[f_idx]))
    {
        int buffer = Trial[position];
        Tree[Trial[position]] = Tree_GetFather(position);
        Trial[position] = Trial[Tree_GetFather(position)];
        Tree[Trial[Tree_GetFather(position)]] = position;
        Trial[Tree_GetFather(position)] = buffer;
        position = Tree_GetFather(position);
        s_idx = Trial[position];
        f_idx = Trial[Tree_GetFather(position)];
    }
    return (PtrToTrial - Trial +1);
};

//================================================================
bool Tree_IsEmpty()
//================================================================
{
    return ((PtrToTrial - Trial + 1) == 0);
};

//================================================================
int Tree_GetRightSon(int position)
//================================================================
{
    if ((2*position+2) < (int)(PtrToTrial - Trial +1))
        return 2*position+2 ;
    else
        return -1;
};

//================================================================
int Tree_GetLeftSon(int position)
//================================================================
{
    if ((2*position+1) < (int)(PtrToTrial - Trial +1))
        return 2*position+1 ;
    else
        return -1;
};

//================================================================
// Tree_UpdateDescent
/*
 * COMMENTS :
 * The tree is updated in order to extract the head by marching down
 * the tree. Starting from the head of the tree, the value of a
 * node is compared with the values of its two sons and replaced by
 * the smallest one. This process is iterated from the son with the
 * smallest value, until a leaf has been reached.
 */
//================================================================
void Tree_UpdateDescent()
{
    int position = 0;
    bool stop = false;
    while((position >= 0)&&(!stop))
    {
        if((Tree_GetRightSon(position)>0)&&(Tree_GetLeftSon(position)>0))
        {
            int ls_idx = Trial[Tree_GetLeftSon(position)];
            int rs_idx = Trial[Tree_GetRightSon(position)];
            if( U[ls_idx] <= U[rs_idx] )
            {
                Trial[position] = Trial[Tree_GetLeftSon(position)];
                Tree[Trial[position]] = position;
                position = Tree_GetLeftSon(position);
            }
            else
            {
                Trial[position] = Trial[Tree_GetRightSon(position)];
                Tree[Trial[position]] = (position);
                position = Tree_GetRightSon(position);
            }
        }
        else
            if(Tree_GetLeftSon(position)>0)
            {
            Trial[position] = Trial[Tree_GetLeftSon(position)];
            Tree[Trial[position]] = (position);
            position = Tree_GetLeftSon(position);
            }
            else
                stop = true;
    }
    if(position != (PtrToTrial - Trial))
    {
        Tree[*PtrToTrial] = position;
        Trial[position]=*PtrToTrial;
        int s_idx = Trial[position];
        int f_idx = Trial[Tree_GetFather(position)];
        while((position!=0)&&(U[s_idx]<U[f_idx]))
        {
            int buffer = Trial[position];
            Tree[Trial[position]] = Tree_GetFather(position);
            Trial[position] = Trial[Tree_GetFather(position)];
            Tree[Trial[Tree_GetFather(position)]] = position;
            Trial[Tree_GetFather(position)] = buffer;
            position = Tree_GetFather(position);
            s_idx = Trial[position];
            f_idx = Trial[Tree_GetFather(position)];
        }
    }
};

//================================================================
int Tree_PopHead()
//================================================================
{
    if(PtrToTrial - Trial + 1)
    {
        int first = *Trial;
        Tree[first] = -1;
        Tree_UpdateDescent();
        PtrToTrial--;
        return first;
    }
    else
        return NULL;
};

//================================================================
void Tree_UpdateChange(int position)
//================================================================
{
    int s_idx = Trial[position];
    int f_idx = Trial[Tree_GetFather(position)];
    while((position!=0)&&(U[s_idx]<U[f_idx]))
    {
        int buffer = Trial[position];
        Tree[Trial[position]] = Tree_GetFather(position);
        Trial[position] = Trial[Tree_GetFather(position)];
        Tree[Trial[Tree_GetFather(position)]] = position;
        Trial[Tree_GetFather(position)] = buffer;
        position = Tree_GetFather(position);
        s_idx = Trial[position];
        f_idx = Trial[Tree_GetFather(position)];
    }
};

//================================================================
void Tree_PullFromTree(int point)
//================================================================
{
    double Uv = U[point];
    U[point] = 0;
    Tree_UpdateChange(Tree[point]);
    U[Tree_PopHead()]=Uv;
};

//================================================================
int Tree_GetSize()
//================================================================
{
    return PtrToTrial - Trial + 1;
};

//================================================================
//================================================================
//================================================================

//================================================================
void InitializeNeighborhoods()
//================================================================
{
    connectivity_small = 4;
    NeighborhoodSmall = (int*) mxCalloc(connectivity_small+1, sizeof(int));
    if(NeighborhoodSmall == NULL)
    {
        mexErrMsgTxt("Bad memory allocation NeighborhoodSmall");
    }
    NeighborhoodSmall[0] = -1;
    NeighborhoodSmall[1] = -Nx;
    NeighborhoodSmall[2] =  1;
    NeighborhoodSmall[3] =  Nx;
    NeighborhoodSmall[4] = -1;
};

//================================================================
void InitializeArrays()
//================================================================
{
    int x, y, point;
    //copy the weight list and initialize arrays
    W = (double*) mxCalloc(size, sizeof(double));
    if(W == NULL)
    {
        mexErrMsgTxt("Bad memory allocation Weights");
    }
    
    S = (short*) mxCalloc(size, sizeof(short));
    if(S == NULL)
    {
        mexErrMsgTxt("Bad memory allocation Statuses");
    }
    
    Tree = (int*) mxCalloc(size, sizeof(int));
    if(Tree == NULL)
    {
        mexErrMsgTxt("Bad memory allocation Tree");
    }
    
    Trial = (int*) mxCalloc(size, sizeof(int));
    if(Trial == NULL)
    {
        mexErrMsgTxt("Bad memory allocation Trial");
    }
    //------------------------------------------------------------
    for(x = 0; x < nx; x++)
    {
        for(y = 0; y < ny; y++)
        {
            point = (x+1) + (y+1)*Nx;
            W[point] = WW[x + y*nx];
            V[point] = kDead; S[point] = kFar;
        }
    }
    for(x = 0; x < size; x++)
    {
        U[x] = INFINITE;
        L[x] = INFINITE;
        Tree[x]=-1;
    }
    //------------------------------------------------------------
    PtrToTrial = Trial - 1;
    //------------------------------------------------------------
    // Initialize Borders
    for(x = 0; x < Nx; x++){
        y = 0;
        point = x + y*Nx;
        V[point] = kBorder; S[point] = kBorder;
        y = Ny-1;
        point = x + y*Nx;
        V[point] = kBorder; S[point] = kBorder;
    }
    for(y = 0; y < Ny; y++){
        x = 0;
        point = x + y*Nx;
        V[point] = kBorder; S[point] = kBorder;
        x = Nx-1;
        point = x + y*Nx;
        V[point] = kBorder; S[point] = kBorder;
    }
};

//================================================================
void InitializeOpenHeap()
//================================================================
{
    unsigned int point, Point;
    for(unsigned int x  = 0; x < nx; x++)
    {
        for(unsigned int y  = 0; y < ny; y++)
        {
            point = x + y*nx;
            Point = x+1 + (y+1)*Nx;
            V[Point] = Nuclei[point];
            if( V[Point] > 0 )
            {
                U[Point] = 0.0;
                L[Point] = 0.0;
                S[Point] = kOpen;
                Tree_PushIn(Point);
            }
        }
    }
    
};

//================================================================
double SethianQuadrant(double Pc,double Ux,double Uy)
//================================================================
{
    double Ua,Ub,qa,qb,Delta;
    
    double result;
    if (Ux<Uy)
    {
        Ua = Ux;  qa = hx2;
        Ub = Uy;  qb = hy2;
    }
    else
    {
        Ua = Uy;  qa = hy2;
        Ub = Ux;  qb = hx2;
    }
    result = INFINITE;
    if ((sqrt(qa)*Pc)>(Ub-Ua))
    {
        Delta = (qa*qb)*((qa+qb)*Pc*Pc-(Ua-Ub)*(Ua-Ub));
        if (Delta>=0)
        {
            result = ((qb*Ua+qa*Ub)+sqrt(Delta))/ hx2_plus_hy2;
        }
        
    }
    else
    {
        result = Ua+sqrt(qa)*Pc;
    }
    return result;
};

//================================================================
bool SethianUpdate(int point)
/*
 * COMMENTS :
 */
//================================================================
{
    int	  npoint;
    std::vector<double>  neighborU(connectivity_small+1, INFINITE);
    std::vector<double>  neighborL(connectivity_small+1, INFINITE);
    double	  Pc = W[point];
    double	  Ur = U[point];
    short	    Vr = V[point];
    double	  Lr = L[point];
    bool is_updated = false;
    
    double weightBackground = 1.0 / (1e7* exp((Pc-meanGlobalInt)*(Pc-meanGlobalInt) / (2.0*stdGlobalInt*stdGlobalInt) ) +1);
    
    //--------------------------------------------------------------
    // Get the U & L values for each neighbor.
    std::vector<unsigned int> listOfRegionIdx;
    listOfRegionIdx.clear();
    bool chockPoint = false;
    for (unsigned int i = 0; i< connectivity_small+1; i++)
    {
        npoint=point+NeighborhoodSmall[i];
        if (S[npoint]==kDead)
        {
            neighborU[i]=U[npoint];
            neighborL[i]=L[npoint];
            listOfRegionIdx.push_back((unsigned int ) V[npoint]);
        }
        else
        {
            neighborU[i]=INFINITE;
            neighborL[i]=INFINITE;
        }
    }
    
    if(listOfRegionIdx.size() > 0)
    {
        for(unsigned int i = 1; i < listOfRegionIdx.size(); i++)
        {
            if(listOfRegionIdx[i] != listOfRegionIdx[0])
            {
                chockPoint = true;
                break;
            }
        }
    }
    
    if( !chockPoint && listOfRegionIdx.size() > 0 )
    {
        unsigned int regionIdx = listOfRegionIdx[0]-1;
        Vr = listOfRegionIdx[0];
        double meanInt = mean_std[2*regionIdx];
        double stdInt  = mean_std[2*regionIdx+1];
        double weight = 1.0 / (1e7* exp(-(Pc-meanInt)*(Pc-meanInt) / (2.0*multFactor*multFactor*stdInt*stdInt) ) +1);
        weight = (weight + weightBackground) /2.0;
        for(unsigned int i = 0; i < connectivity_small; i++)
        {
            
            double Utmp = SethianQuadrant(weight,neighborU[i],neighborU[i+1]);
            if(Utmp < Ur)
            {
                Ur = Utmp;
                Lr = SethianQuadrant(1,neighborL[i],neighborL[i+1]);
                is_updated = true;
            }
        }
    }
    else if(chockPoint)
    {
        sort(listOfRegionIdx.begin(), listOfRegionIdx.begin()+listOfRegionIdx.size());
        std::vector<unsigned int>::iterator it;
        it = unique (listOfRegionIdx.begin(), listOfRegionIdx.end()); // 10 20 30 20 10 ?  ?  ?  ?
        listOfRegionIdx.resize( it - listOfRegionIdx.begin() );
        
        std::vector<double> valuesPerRegion;
        valuesPerRegion.clear();
        std::vector<double> lengthPerRegion;
        lengthPerRegion.clear();
        
        for(unsigned int j = 0; j < listOfRegionIdx.size(); j++)
        {
            for (unsigned int i = 0; i< connectivity_small+1; i++)
            {
                npoint=point+NeighborhoodSmall[i];
                if (S[npoint]==kDead && V[npoint] == listOfRegionIdx[j])
                {
                    neighborU[i]=U[npoint];
                    neighborL[i]=L[npoint];
                }
                else
                {
                    neighborU[i]=INFINITE;
                    neighborL[i]=INFINITE;
                }
            }
            unsigned int regionIdx = listOfRegionIdx[j];
            double meanInt = mean_std[2*regionIdx];
            double stdInt  = mean_std[2*regionIdx+1];
            double weight = 1.0 / (1e7* exp(-(Pc-meanInt)*(Pc-meanInt) / (2.0*multFactor*multFactor*stdInt*stdInt) ) +1);
            weight = (weight+ weightBackground) /2.0;
            for(unsigned int i = 0; i < connectivity_small; i++)
            {
                
                double Utmp = SethianQuadrant(weight,neighborU[i],neighborU[i+1]);
                if(Utmp < Ur)
                {
                    Ur = Utmp;
                    Lr = SethianQuadrant(1,neighborL[i],neighborL[i+1]);
                    is_updated = true;
                }
            }
            
            valuesPerRegion.push_back(Ur);
            lengthPerRegion.push_back(Lr);
        }
        Ur = valuesPerRegion[0];
        Lr = lengthPerRegion[0];
        Vr = listOfRegionIdx[0];
        for(unsigned int j = 1; j < listOfRegionIdx.size(); j++)
        {
            if(valuesPerRegion[j] < Ur)
            {
                Ur = valuesPerRegion[j];
                Lr = lengthPerRegion[j];
                Vr = listOfRegionIdx[j];
            }
        }
    }
    //--------------------------------------------------------------
    if (is_updated){
        U[point]=Ur;
        L[point]=Lr;
        V[point]=Vr;
    }
    //--------------------------------------------------------------
    neighborU.clear(); neighborL.clear();
    //--------------------------------------------------------------
    return is_updated;
};

//================================================================
void RunPropagation()
//================================================================
{
    int point,npoint,k;
    bool is_updated = false;
    //--------------------------------------------------------------
    while ( Tree_GetSize()>0 )
    {
        point = Tree_PopHead();
        if(S[point]!=kOpen)
            mexErrMsgTxt("err");
        S[point]=kDead;
        //--------------------------------------------------------------
        for (k=0;k<connectivity_small;k++){
            npoint = point+NeighborhoodSmall[k];
            //--------------------------------------------------------------
            if (S[npoint]==kOpen){
                is_updated = SethianUpdate(npoint);
                if(is_updated)
                    Tree_UpdateChange(Tree[npoint]);
            }
            //--------------------------------------------------------------
            else if (S[npoint]==kFar){
                S[npoint] = kOpen;
                SethianUpdate(npoint);
                Tree_PushIn(npoint);
            }
            //--------------------------------------------------------------
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
        U[point] = U[Point];
        L[point] = L[Point];
        V[point] = V[Point];
        }
};