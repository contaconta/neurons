
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
#include "Neuron.h"
#include "Image.h"
#include "Cloud.h"

using namespace std;

Neuron* n;

void addNeuronSegmentToCloud(NeuronSegment* segment,
                             Cloud<Point2Dotw>* cl, Image<float>* img)
{
  float theta, radius;
  vector< float > mcoords(3);
  vector< float > mcoords2(3);
  vector< float > mcnext(3);

  // Gets the limits in micrometers of the images
  vector< float > v_0m(3);
  vector< float > v_6m(3);
  vector< int > v_0(3);
  vector< int > v_6(3);
  v_0[0]=0;v_0[1]=0;v_0[2]=0;
  v_6[0]=img->width-1; v_6[1]=img->height-1; v_6[2]=0;
  img->indexesToMicrometers(v_0, v_0m);
  img->indexesToMicrometers(v_6, v_6m);

  // printf("Limits in micrometers: [%f, %f, %f]->[%f, %f, %f]\n",
         // v_0m[0], v_0m[1], v_0m[2], v_6m[0], v_6m[1], v_6m[2]); 

  for(int i = 0; i < segment->points.size()-1; i++){
    printf("   point %i: ", i);
    vector< float > neuronMicrometers = segment->points[i].coords;
    neuronMicrometers[0] += img->width/2;
    neuronMicrometers[1] += img->height/2;
    n->neuronToMicrometers(neuronMicrometers,
                           mcoords2);
    n->neuronToMicrometers(segment->points[i].coords,
                           mcoords);
    //If the point is within the image, accept it
    if ( (mcoords2[0] > v_0m[0]) && (mcoords2[0] < v_6m[0]) &&
         (mcoords2[1] < v_0m[1]) && (mcoords2[1] > v_6m[1]) ){
      //Computes thetea
      if(i!= segment->points.size()-1)
        n->neuronToMicrometers(segment->points[i+1].coords, mcnext);
      else if (i >= 1)
        n->neuronToMicrometers(segment->points[i-1].coords, mcnext);
      else continue;
      theta = atan2(+mcnext[1]- mcoords[1], mcnext[0]-mcoords[0]);

 //mcoords[0],mcoords[1],
      printf("[%f,%f]-%f", mcoords2[0], mcoords2[1], theta);
      cl->points.push_back(new Point2Dotw(mcoords2[0], mcoords2[1],
                                          // segment->points[i].coords[0],
                                          // segment->points[i].coords[1],
                                          theta, 1,
                                          segment->points[i].coords[3]));

    }
    printf("\n");
  }

  for(int i = 0; i < segment->childs.size(); i++)
    addNeuronSegmentToCloud(segment->childs[i], cl, img);

}

int main(int argc, char **argv) {

  if(argc!= 4){
    printf("Usage: neuronToCloudI neuron.asc imageMode.jpg cloud.cl\n");
    exit(0);
  }

  n = new Neuron(argv[1]);
  Image<float>* img = new Image<float>(argv[2],0);
  Cloud<Point2Dotw>* cl = new Cloud<Point2Dotw>();

  for(int i = 0; i < n->dendrites.size(); i++){
    printf("Parsing dendrite %i\n", i);
    addNeuronSegmentToCloud(n->dendrites[i], cl, img);
  }
  for(int i = 0; i < n->axon.size(); i++){
    printf("Parsing axon %i\n", i);
    addNeuronSegmentToCloud(n->axon[i], cl, img);
  }

  cl->saveToFile(argv[3]);

}
