#include "mex.h"
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/config.hpp>
#include <iostream>
#include <fstream>
#include <vector>

using namespace boost;

typedef    property < vertex_index_t, long ,
                      property< vertex_name_t, long> > VertexProperty;
typedef property < edge_weight_t, double > EdgeProperty;


typedef adjacency_list < listS, vecS, undirectedS,
                         VertexProperty,
                         EdgeProperty
                         > GraphType;

typedef graph_traits < GraphType >::vertex_descriptor VertexType;
typedef graph_traits < GraphType >::edge_descriptor   EdgeType;
typedef std::pair<long, long>                         Edge;


void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
  double *Img,*Mask, *Output;
  mwSize nrows,ncols, nrows1, ncols1, nrows2, ncols2;
  
  /* Check for proper number of arguments. */
  if(nrhs!=2) {
    mexErrMsgTxt("Two inputs required.");
  } else if(nlhs>2) {
    mexErrMsgTxt("Too many output arguments.");
  }
  
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
    mexErrMsgTxt("The images have different size");
  
  nrows = nrows1;
  ncols = ncols1;
  
  mexPrintf("Arguments correct\n");
  
  /* Create matrix for the return argument. */
  plhs[0] = mxCreateDoubleMatrix(nrows,ncols, mxREAL);
  
  /* Assign pointers to each input and output. */
  Img    = mxGetPr(prhs[0]);
  Mask   = mxGetPr(prhs[1]);
  Output = mxGetPr(plhs[0]);

  int y = 127;
  int x = 369;
  mexPrintf("the value at [%i,%i] is %f and %f\n", x, y,
            Img[y*ncols+x], Mask[y*ncols+x]);

  
  mexPrintf("Creating the graph\n");
  
  long num_nodes = (nrows)*(ncols);
  long num_arcs  = (nrows-1)*(ncols-1)*2; // Change afterwards to 8-connectivity

  mexPrintf("  Number of nodes: %i, Number of Arcs: %i\n", num_nodes, num_arcs);

  std::vector< Edge > edge_array(num_arcs);
  std::vector< double > weights(num_arcs);
  
  int rootX = 0;
  int rootY = 0;
  
  long edgeCounter = 0;
  long ydx0, ydx1;
  double val;
  for(int y = 0; y < nrows-1; y++){
    ydx0 = y*ncols;
    ydx1 = (y+1)*ncols;
    // printf("processing row %i\n", y);
    for(int x =0 ; x < ncols-1; x++){
  
      if( (Mask[ydx0+x] == 1) && (Mask[ydx0+x+1] == 1) ) continue;

      //     To the right              
      edge_array[edgeCounter] = Edge( ydx0 + x, ydx0 + x + 1);
      if( (Mask[ydx0+x] == 2) && (Mask[ydx0+x+1] == 2) )
         val = 1;
      else
         val = (Img[ydx0+x] + Img[ydx0+x+1])/2;
      if(val > 0.9999999999) val = 0.9999999999;
      if(val < 0.0000000001) val = 0.0000000001;
      weights[edgeCounter]     = (double)(-log( val));
      
      edgeCounter++;

      if( (Mask[ydx0+x] ==  1) && (Mask[ydx1+x] == 1) )
      //     To the bottom
      edge_array[edgeCounter] = Edge(ydx0+x, ydx1+x);
      if( (Mask[ydx0+x] ==  2) && (Mask[ydx1+x] == 2) )
         val = 1;
      else
      val = (Img[ydx0+x] + Img[ydx1+x])/2;
      if(val > 0.9999999999) val = 0.9999999999;
      if(val < 0.0000000001) val = 0.0000000001;
      weights[edgeCounter]     = (double)(-log( val));

      edgeCounter++;
      
      if(Mask[ydx0+x] > 0){
        rootX = x;
        rootY = y;
      }

    }
  }
  
   
   mexPrintf("Done creating the graph\n");

   // Check for negative edge weights
  for(unsigned int i = 0; i < num_arcs; i++)
    if(weights[i] <= 0)
      printf("Edge[%i] = %f\n", i, weights[i]);

  GraphType g(&edge_array[0], &edge_array[0] + num_arcs, &weights[0], num_nodes);

     
    property_map<GraphType, edge_weight_t>::type weightmap = get(edge_weight, g);
  std::vector<VertexType>    p(num_vertices(g));
  std::vector<double>        d(num_vertices(g));



  
  VertexType s = vertex(rootY*ncols + rootX, g);

  dijkstra_shortest_paths(g, s, predecessor_map(&p[0]).distance_map(&d[0]));


  long rootIndex = rootY*ncols + rootX;
  std::cout << "Distance to root: " << d[rootIndex] << " " << rootIndex << std::endl;
  std::cout << " distance to the right " << d[rootIndex+1] << " parent: " << p[rootIndex+1] << std::endl;
  std::cout << " distance to the left " << d[rootIndex-1] << " parent: " << p[rootIndex-1] <<std::endl;

  for(int y = 0; y < nrows; y++){
    for(int x = 0; x < ncols; x++){
        Output[y*ncols + x] = exp(-d[y*ncols + x]);
    }
  }
  
  
  
}
