
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
// Contact <ggonzale@atenea> for comments & bug reports                //
/////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "Cube.h"
#include <climits>
#include "Cloud.h"

using namespace std;

/** Each voxel is surrounded by 26 neighbors. The [x,y,z] coordinates
    of the cube follow the scheme of neseg. For the neighbors we
    follow the following schma:

View with Z in depth:

Layer 1:         Layer 2:         Layer 3:

0------>X
|  0 ----------3      ----------     1 ----------2
|  | 1 | 2 | 3 |    | 10| 11| 12|    | 18| 19| 20|
|  |--- --- ---|    |--- --- ---|    |--- --- ---|
|  | 4 | 5 | 6 |    | 13| X | 14|    | 21| 22| 23|
Y  |-----------|    |-----------|    |-----------|
   | 7 | 8 | 9 |    | 15| 16| 17|    | 24| 25| 26|
   4-----------7     -----------     5-----------6

*/

/** Table that codes the relationship between the indexes previously
    and the [x,y,z] indexes in the cube*/
int nbrToIdx[27][3] = {
  { 0, 0, 0}, //0
  {-1,-1,-1}, //Layer 1
  { 0,-1,-1},
  { 1,-1,-1},
  {-1, 0,-1},
  { 0, 0,-1}, //5
  { 1, 0,-1},
  {-1, 1,-1},
  { 0, 1,-1},
  { 1, 1,-1},
  {-1,-1, 0}, // 10 - Layer 2
  { 0,-1, 0},
  { 1,-1, 0},
  {-1, 0, 0},
  { 1, 0, 0},
  {-1, 1, 0}, //15
  { 0, 1, 0},
  { 1, 1, 0},
  {-1,-1, 1}, // Layer 3
  { 0,-1, 1},
  { 1,-1, 1}, //20
  {-1, 0, 1},
  { 0, 0, 1},
  { 1, 0, 1},
  {-1, 1, 1},
  { 0, 1, 1}, //25
  { 1, 1, 1}
};

/* From x,y,z to idx.*/
int toLinearIndex(int x, int y, int z, Cube<uchar, ulong>* cube){
  return (z * cube->cubeHeight + y) * cube->cubeWidth + x;
}

void toCubeIndex(int idx, int& x, int& y, int& z, Cube<uchar, ulong>* cube){
  z = idx / (cube->cubeHeight * cube->cubeWidth);
  y = (idx - z*(cube->cubeHeight * cube->cubeWidth))/cube->cubeWidth;
  x = idx - z*(cube->cubeHeight * cube->cubeWidth) - y*cube->cubeWidth;
}


float distance(Cube<uchar, ulong>* orig, int x0, int y0, int z0,
             int x1, int y1, int z1)
{
  return ((float)fabs(orig->at(x0,y0,z0)-orig->at(x1,y1,z1)))/255 ;
    // sqrt( (x0-x1)*(x0-x1) + (y0-y1)*(y0-y1) + 3*(z0-z1))/sqrt(5);
}

class PointDijkstra
{
public:
  int previous;
  float distance;
  PointDijkstra(int p, float d){
    previous = p; distance = d;
  }
};

int main(int argc, char **argv) {

  printf("The size of int is %i\n", sizeof(int));
  printf("The size of float is %i\n", sizeof(float));

  Cube<uchar, ulong>* orig      = new Cube<uchar, ulong>("/media/neurons/cut/aguet_4.00_4.00.nfo");

  map<int, PointDijkstra*>            pointsTakenIntoAccount;
  map<int, PointDijkstra*>::iterator  pointsIt;
  map<int, PointDijkstra*>::iterator  pointsIt2;
  multimap<float, int> boundary; //Ordered by the distances

  //Init points
  // int x0=27,y0=47,z0=23,x1=188,y1=260,z1=34;
  int x0=27,y0=47,z0=23,x1=248,y1=505,z1=38;
  // int x0=93,y0=82,z0=25,x1=136,y1=131,z1=27;
  // int x0=27,y0=47,z0=23,x1=30,y1=47,z1=23;
  boundary.insert(pair<float, int>(0, toLinearIndex(x0,y0,z0,orig)));

  //Next point to be analyzed
  int xN=x0, yN=y0, zN=z0;
  pointsTakenIntoAccount.insert(
          pair<int, PointDijkstra*>(
                   toLinearIndex(x0,y0,z0,orig),
                   new PointDijkstra(0,0)));

  printf("Evaluating %i %i %i\n", xN, yN, zN);
  multimap< float, int >::iterator itb;

  // Begin of the loop
  int nPointsEvaluated = 0;
  while( !((xN==x1) && (yN==y1) && (zN==z1)) ){

    //Eliminate the first point
    itb = boundary.begin();
    boundary.erase(itb);

    pointsIt = pointsTakenIntoAccount.find(toLinearIndex(xN,yN,zN,orig));
    PointDijkstra* peval = pointsIt->second;

    //Add the neighbors of xN,yN,zN to the list
    int xA, yA, zA;
    for(int i = 1; i < 27; i++){
      xA = xN + nbrToIdx[i][0];
      yA = yN + nbrToIdx[i][1];
      zA = zN + nbrToIdx[i][2];
      //borders ...
      if ( (xA < 0) || (yA < 0) || (zA < 0) ||
           (xA >= orig->cubeWidth)  ||
           (yA >= orig->cubeHeight) ||
           (zA >= orig->cubeDepth ) )
        continue;
      // if they have not yet been visited
      if( pointsTakenIntoAccount.find(toLinearIndex(xA,yA,zA,orig))
          == pointsTakenIntoAccount.end()) {
        // int dist = peval->distance + distance(orig,xA,yA,zA,xN,yN,zN);
        float dist = distance(orig,xA,yA,zA,x0,y0,z0);
        boundary.insert(pair<float, int>( dist,
                                       toLinearIndex(xA,yA,zA,orig)));
        pointsTakenIntoAccount.insert(
                            pair<int, PointDijkstra*>(
                                toLinearIndex(xA,yA,zA,orig),
                                new PointDijkstra(toLinearIndex(xN,yN,zN,orig), dist)));
      }
    }

    //Take the closest point of the origin in the boundary
    multimap< float, int >::iterator it = boundary.begin();
    int linIndex = it->second;
    toCubeIndex(linIndex, xN, yN, zN, orig);

    // Save the points in a cloud
    nPointsEvaluated ++;
    if((nPointsEvaluated % 1000)==0){
      printf("Evaluated %i points\r", nPointsEvaluated);
      fflush(stdout);
    }
    if(0){
      nPointsEvaluated ++;
      if((nPointsEvaluated % 1000)==0) {
        // printf("Evaluated %i points\r", nPointsEvaluated);
        char buff[512];
        sprintf(buff,"boundary_%i.cl", nPointsEvaluated);
        Cloud<Point3D>* cloud = new Cloud<Point3D>(buff);
        vector< int > indexes(3);
        vector< float > micrometers(3);
        for(it = boundary.begin(); it != boundary.end(); it++){
          int idx = it->second;
          toCubeIndex(idx, indexes[0], indexes[1], indexes[2], orig);
          orig->indexesToMicrometers(indexes, micrometers);
          cloud->points.push_back(new Point3D(micrometers[0], micrometers[1], micrometers[2]));
        }
        cloud->saveToFile(buff);
      }
    } //if(save)
  } //while


  // And now finds the way back tracing the path
  int xP = x1, yP = y1, zP = z1;
  Cloud<Point3D>* result = new Cloud<Point3D>("result.cl");
  vector< int > indexes(3);
  vector< float > micrometers(3);
  while(!( (xP==x0) && (yP==y0)&& (zP==z0) )){
    indexes[0] = xP;
    indexes[1] = yP;
    indexes[2] = zP;
    orig->indexesToMicrometers(indexes, micrometers);
    result->points.push_back(new Point3D(micrometers[0], micrometers[1], micrometers[2]));
    printf("%i %i %i\n", xP,yP,zP);
    int idx = toLinearIndex(xP,yP,zP,orig);
    pointsIt2 = pointsTakenIntoAccount.find(idx);
    PointDijkstra* pd = pointsIt2->second;
    toCubeIndex(pd->previous, xP, yP, zP, orig);
  }
  result->saveToFile("result.cl");

}
