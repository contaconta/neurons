/* Class to load any Cloud.*/

#ifndef GRAPH_P_H
#define GRAPH_P_H

#include "neseg.h"

#include "Point.h"
#include "Point2D.h"
#include "Point3D.h"
#include "Point3Dw.h"
#include "utils.h"
#include "Object.h"
#include "VisibleE.h"
#include "EdgeSet.h"
#include "Edge.h"
#include "Cube_P.h"

class Graph_P : public VisibleE
{
 public:

  Cloud_P* cloud;

  // EdgeSet<Point, Edge< Point > >* esetp;

  Graph_P() : VisibleE(){}

  virtual void prim()=0;

  virtual void changeCloud(Cloud_P* cl)=0;

//   virtual vector< vector< double > > sampleLatticeArroundEdges
//   (Cube_P* cube, int nx, int ny, int nz, double dy, double dz);

};




#endif
