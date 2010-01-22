
/////////////////////////////////////////////////////////////////////////
// This program is free software; you can redistribute it and/or       //
// modify it under the terms of the GNU General Public License         //
// version 2 as published by the Free Software Foundation.             //
//                                                                     //
// This program is distributed in the hope that it will be useful, but //
// WITHOUT ANY WARRANTY; without even the implied warranty of          //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU   //
// General Public License for more details.                            //
//                                                                     //
// Written and (C) by German Gonzalez                                  //
// Contact <german.gonzalez@epfl.ch> for comments & bug reports        //
/////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "SteerableFeatures2DRadius.h"
#include "Cloud.h"

using namespace std;

vector<RadialFunction*>  polynomialBasis(float radius, int order)
{
  vector<RadialFunction*> toRet;
  toRet.push_back(new Po0(radius));
  if(order >= 1)
    toRet.push_back(new Po1(radius));
  if(order >= 2)
    toRet.push_back(new Po2(radius));
  if(order >= 3)
    toRet.push_back(new Po3(radius));
  if(order >= 4)
    toRet.push_back(new Po4(radius));
  if(order >= 5)
    toRet.push_back(new Po5(radius));
  if(order >= 6)
    toRet.push_back(new Po6(radius));
  if(order >= 7)
    toRet.push_back(new Po7(radius));

  return toRet;
}



int main(int argc, char **argv) {

  if(argc != 8){
    printf("Usage getCoordinatesRadius image radius cloud.cl outputName.txt order includeOddOrders polynomialOrder\n");
    exit(0);
  }

  string imageName (argv[1]);
  float radius = atof(argv[2]);
  Cloud<Point2Dot>* cl = new Cloud<Point2Dot>(argv[3]);
  string outputName(argv[4]);
  int order = atoi(argv[5]);
  int includeOddOrders = atoi(argv[6]);
  int polynomialOrder  = atoi(argv[7]);

  printf("imageRadius %s, %f, npoints = %i includeOddOrders = %i\n", imageName.c_str(), radius, cl->points.size(), includeOddOrders);

  // rfunc.push_back(new Torus(2*radius, radius));

  SteerableFilter2DMR* stf = new SteerableFilter2DMR
    (imageName, order, 1.0, "result.jpg",
     polynomialBasis(radius, polynomialOrder), includeOddOrders, true, false);

  Image<float>* img = new Image<float>(imageName);

  vector< float > coords = stf->getDerivativeCoordinatesRotated(0,0,0);
  std::ofstream out(outputName.c_str());
  out << cl->points.size() << " " << coords.size() + 1 << std::endl;
  int x, y;
  for(int i = 0; i < cl->points.size(); i++){
    Point2Dot* pt = dynamic_cast<Point2Dot*>(cl->points[i]);
    img->micrometersToIndexes(pt->coords[0], pt->coords[1], x, y);
    coords = stf->getDerivativeCoordinatesRotated(x, y, pt->theta);
    for(int j = 0; j < coords.size(); j++)
      out << coords[j] << " ";
    out << pt->type << std::endl;
  }
  out.close();
}
