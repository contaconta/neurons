
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

  if(argc != 9){
    printf("Usage steerRadius image radius thetaImage fisher_vector outputImage order includeOddOrders polynomialOrder\n");
    exit(0);
  }

  printf("New\n");
  string imageName (argv[1]);
  float radius = atof(argv[2]);
  printf("steerRadius %s, %f\n", imageName.c_str(), radius);
  vector< double > fisher = readVectorDouble(argv[4]);
  string outputImage(argv[5]);
  int order = atoi(argv[6]);
  int includeOddOrders= atoi(argv[7]);
  int polynomialOrder= atoi(argv[8]);

  Image<float>* img   = new Image<float>(imageName.c_str());
  Image<float>* theta = new Image<float>(argv[3]);
  Image<float>* result = img->create_blank_image_float(outputImage);

  // vector< RadialFunction* > rfunc;
  // rfunc.push_back(new Po0(radius));
  // rfunc.push_back(new Po1(radius));
  // rfunc.push_back(new Po2(radius));
  // rfunc.push_back(new Po3(radius));

  SteerableFilter2DMR* stf = new SteerableFilter2DMR
    (imageName, order, 1.0, "result.jpg",
     polynomialBasis(radius, polynomialOrder), includeOddOrders, true, false);

  printf("Doing linear filtering: [");
#pragma omp parallel for
  for(int y = 10; y < img->height-10; y++){
    for(int x = 10; x < img->width-10; x++){

  // for(int y = 100; y < 300; y++){
      // for(int x = 0; x < img->width; x++){
      vector< float > imgData =
        stf->getDerivativeCoordinatesRotated(x,y,theta->at(x,y));
        // stf->getDerivativeCoordinatesRotated(x,y,0);
      float res = 0;
      for(int i = 0; i < imgData.size(); i++)
        res += fisher[i]*imgData[i];
      result->put(x,y,res);
    }
    printf("#"); fflush(stdout);
  }
  printf("]\n");
  result->save();

}
