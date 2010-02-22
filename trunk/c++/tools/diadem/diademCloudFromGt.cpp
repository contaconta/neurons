
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
#include "Cube.h"
#include "SWC.h"
#include <gsl/gsl_rng.h>

using namespace std;




int main(int argc, char **argv) {

  if(argc!=6){
    printf("Usage: diademCloudFromGr swc cube cube_theta cube_phi out.cl\n");
    exit(0);
  }

  const gsl_rng_type * T2;
  gsl_rng * r;
  gsl_rng_env_setup();
  T2 = gsl_rng_default;
  r = gsl_rng_alloc (T2);

  SWC*  swc = new SWC(argv[1]);
  CubeF* cb = new CubeF(argv[2]);
  CubeF* cb_theta = new CubeF(argv[3]);
  CubeF* cb_phi = new CubeF(argv[4]);
  Cloud<Point3Dot>* out = new Cloud<Point3Dot>();
  Graph<Point3Dw>* gr = swc->toGraphInMicrometers(cb);

  printf("Generating positive points .... \n");
  float theta, phi, radius;
  phi = 3.14159/2;
  vector< int > visited(gr->cloud->points.size());
  for(int i = 0; i < gr->cloud->points.size()-1; i++){
    // printf("%i\n", i);
    visited[i] = 1;
    Point* pt   = gr->cloud->points[i];
    //Look for the points that are close in the sense of two edges far appart
    for(int nE = 0; nE < gr->eset.edges.size(); nE++){
      Edge<Point3Dw>* ed = gr->eset.edges[nE];
      int nextIdx, nextIdxTmp;
      if ((ed->p0 == i) || (ed->p1 == i)){
        if (ed->p0 == i) nextIdxTmp = ed->p1;
        if (ed->p1 == i) nextIdxTmp = ed->p0;
        for(int nE2 = 0; nE2 < gr->eset.edges.size(); nE2++){
           Edge<Point3Dw>* ed2 = gr->eset.edges[nE2];
           if ((ed2->p0 == nextIdxTmp) || (ed2->p1 == nextIdxTmp)){
             if (ed2->p0 == nextIdxTmp) nextIdx = ed->p1;
             if (ed2->p1 == nextIdxTmp) nextIdx = ed->p0;
             if(visited[nextIdx]==1 || (nextIdx==i) ) continue;
             Point* next = gr->cloud->points[nextIdx];
             theta = atan2(next->coords[1] - pt->coords[1], next->coords[0]-pt->coords[0]);
             radius = sqrt( (next->coords[2]-pt->coords[2])*(next->coords[2]-pt->coords[2]) +
                            (next->coords[1]-pt->coords[1])*(next->coords[1]-pt->coords[1]) +
                            (next->coords[0]-pt->coords[0])*(next->coords[0]-pt->coords[0]) );
             if(radius==0) continue;
             phi = acos( (next->coords[2]-pt->coords[2])/radius);
             // printf("nE: %f %f\n", phi, radius);
             if(phi > M_PI) phi = phi -M_PI;
             out->points.push_back
               (new Point3Dot( pt->coords[0], pt->coords[1], pt->coords[2],
                               theta, phi, 1));
           }//ed2
        }//nE2
      }//ed
    }//nE
  }//point idx


  ///////////// NEGATIVE POINTS GENERATION      /////////
  printf("Creating the negative mask  points .... \n");
  CubeU* maskTrue = cb->create_blank_cube_uchar("mask_positive");
  CubeU* maskDouble = cb->create_blank_cube_uchar("mask_positive_x2");
  maskTrue->put_all(255);
  maskDouble->put_all(255);
  for(int nE = 0; nE < gr->eset.edges.size(); nE++){
    vector< int > idx1(3);
    vector< int > idx2(3);
    Point3Dw* pt1 = dynamic_cast<Point3Dw*>(gr->cloud->points[gr->eset.edges[nE]->p0]);
    Point3Dw* pt2 = dynamic_cast<Point3Dw*>(gr->cloud->points[gr->eset.edges[nE]->p1]);
    maskTrue->micrometersToIndexes(pt1->coords, idx1);
    maskTrue->micrometersToIndexes(pt2->coords, idx2);
    maskTrue->render_cylinder(idx1, idx2, (pt1->weight + pt2->weight)/2);
    maskDouble->render_cylinder(idx1, idx2, 4*(pt1->weight + pt2->weight)/2);
  }
  for(int i = 0; i < maskTrue->size(); i++)
    if(maskTrue->voxels_origin[i] == 0)
      maskDouble->voxels_origin[i] = 255;



  printf("Generating negative points .... \n");
  int nNegativePointsGenerated = 0;
  int nPositivePoints = out->points.size();
  int x, y, z;
  float mx, my, mz;
  //Points close to the tree
  while(nNegativePointsGenerated < nPositivePoints/2){
    x = (int)(gsl_rng_uniform(r)*cb->cubeWidth);
    y = (int)(gsl_rng_uniform(r)*cb->cubeHeight);
    z = (int)(gsl_rng_uniform(r)*cb->cubeDepth);
    if(maskDouble->at(x,y,z)==  255) continue;
    printf("nNegativePointsGenerated %i\n", nNegativePointsGenerated);
    cb->indexesToMicrometers3(x,y,z,mx,my,mz);
    out->points.push_back
      (new Point3Dot( mx, my, mz,
                      cb_theta->at(x,y,z), cb_phi->at(x,y,z), -1));
    nNegativePointsGenerated++;
  }
  //Random points
  printf("Generating negative points outside ... \n");
  while(nNegativePointsGenerated < nPositivePoints){
    x = (int)(gsl_rng_uniform(r)*cb->cubeWidth);
    y = (int)(gsl_rng_uniform(r)*cb->cubeHeight);
    z = (int)(gsl_rng_uniform(r)*cb->cubeDepth);
    if(maskTrue->at(x,y,z) ==  0) continue;
    cb->indexesToMicrometers3(x,y,z,mx,my,mz);
    out->points.push_back
      (new Point3Dot( mx, my, mz,
                      cb_theta->at(x,y,z), cb_phi->at(x,y,z), -1));
    nNegativePointsGenerated++;
    printf("nNegativePointsGenerated %i\n", nNegativePointsGenerated);
  }

  out->saveToFile(argv[5]);
}
