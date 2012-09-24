/************************************************************************/
/* (c) 2009-2011 Ecole Polytechnique Federale de Lausanne               */
/* All rights reserved.                                                 */
/*                                                                      */
/* EPFL grants a non-exclusive and non-transferable license for non     */
/* commercial use of the Software for education and research purposes   */
/* only. Any other use of the Software is expressly excluded.           */
/*                                                                      */
/* Redistribution of the Software in source and binary forms, with or   */
/* without modification, is not permitted.                              */
/*                                                                      */
/* Written by Engin Turetken.                                           */
/*                                                                      */
/* http://cvlab.epfl.ch/research/body/surv                              */
/* Contact <pom@epfl.ch> for comments & bug reports.                    */
/************************************************************************/

#ifndef K_SHORTHEST_PATH_GRAPH_H
#define K_SHORTHEST_PATH_GRAPH_H

#include <vector>
#include <boost/graph/adjacency_list.hpp>

#include "mex.h"
#include "matrix.h"
#include "global.h"



class KShorthestPathGraph
{
		
 public:

  /**
   *
   * @param pfData 1D array of node values. The orginal data is
   * assumed to be 3D. First two dimensions correspond to width and
   * height respectively. Third dimension corresponds to depth / time.
   * @param 1D data array indices (i.e., scan line order) of nodes,
   * which are to be connected to source and destination nodes, in
   * every depth channel.
   */		
  KShorthestPathGraph( 
                      float* pfData,
                      int nDataWidth,
                      int nDataHeight,
                      int nDataDepth,
                      int nNodeNeighborhoodSize,
                      std::vector<int> &pnSrcAndDstNeighIndices);
  
  KShorthestPathGraph(const mxArray* Cells,
                      const mxArray* CellsList,
                      int temporal_windows_size,
                      double spatial_windows_size,
                      double *imagesize,
                      double distanceToBoundary);
  
  
	
  virtual ~KShorthestPathGraph();
	
	
  typedef boost::adjacency_list<	boost::vecS, 
    boost::vecS, 
    boost::directedS, 
    boost::no_property, 
    boost::property < boost::edge_weight_t, 
    float > > BaseGraphType;	
	
  inline BaseGraphType& GetBaseGraph()
  {
    return (*m_pG);
  }
	
  inline int GetSrcNodeIndx()
  {
    return m_nSrcNodeIndx;
  }
	
  inline int GetDstNodeIndx()
  {
    return m_nDstNodeIndx;
  }
	
  inline int GetNoOfVertices()
  {
    return boost::num_vertices(*m_pG);
  }

 private:
  BaseGraphType* m_pG;
  int m_nSrcNodeIndx;
  int m_nDstNodeIndx;		
};

#endif
