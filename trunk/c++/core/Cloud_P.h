/* Class to load any Cloud.*/

#ifndef CLOUD_P_H
#define CLOUD_P_H

#include "Point.h"
#include "Point2D.h"
#include "Point2Do.h"
#include "Point2Dot.h"
#include "Point2Dotw.h"
#include "Point3D.h"
#include "Point3Dt.h"
#include "Point3Do.h"
#include "Point3Dot.h"
#include "Point3Dotw.h"
#include "utils.h"
#include <string>
#include "Object.h"
#include "VisibleE.h"
#include <math.h>
// #include "CloudFactory.h"

class Cloud_P : public VisibleE
{
 public:

  vector< Point* > points;

  Cloud_P( ) : VisibleE(){}

  virtual string className(){
    return "Cloud_P";
  }

  virtual vector<double> spread() = 0;

  /** Splits the cloud in two clouds, one with the odd points and the other one with the
      even ones*/
  virtual void split(Cloud_P* &cl1, Cloud_P* &cl2) = 0;

  /** If the cloud includes some weight in the points, then elliminates the points out of the range.
   If the cloud's points do not have type, then do nothing to them.*/
  virtual void cleanPointsAccordingToWeight(double minWeight, double maxWeight) = 0;

  int findPointClosestTo(float xS, float yS, float zS)
  {
    float dMaxPSoma = FLT_MAX;
    int   pointSoma = 0;
    for(int i = 0; i < points.size(); i++){
      float d = sqrt
        ( (points[i]->coords[0]-xS)*(points[i]->coords[0]-xS) +
          (points[i]->coords[1]-yS)*(points[i]->coords[1]-yS) +
          (points[i]->coords[2]-zS)*(points[i]->coords[2]-zS) );
      if( d < dMaxPSoma){
        dMaxPSoma = d;
        pointSoma = i;
      }
    }
    return pointSoma;
  }

  vector< int >  findPointsCloserToThan(float xS, float yS, float zS, float R)
  {
    vector< int > pointsClose;
    for(int i = 0; i < points.size(); i++){
      float d = sqrt
        ( (points[i]->coords[0]-xS)*(points[i]->coords[0]-xS) +
          (points[i]->coords[1]-yS)*(points[i]->coords[1]-yS) +
          (points[i]->coords[2]-zS)*(points[i]->coords[2]-zS) );
      if( d < R)
        pointsClose.push_back(i);
    }
    return pointsClose;
  }


  // virtual void addPoint(float x, float y, float z) = 0;

};



#endif
