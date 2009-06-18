#include "CubeLiveWire.h"


void*
CubeLiveWire::computeDistances
(int x0, int y0, int z0)
{
  if((x0 < 0)||(y0<0)||(z0<0)||
     (x0 >= cube->cubeWidth) ||
     (y0 >= cube->cubeHeight) ||
     (z0 >= cube->cubeDepth))
    return NULL;

  computingDistances = true;
  printf("Computing the distances from [%i,%i,%i]\n", x0,y0,z0);

  //Delete the previously found information
  for(int i = 0; i < cube->cubeWidth*cube->cubeHeight*cube->cubeDepth; i++){
    previous_orig[i]  = -1;
    distance_orig[i] = FLT_MAX;
    visited_orig[i]   = false;
  }
  distances[z0][y0][x0] = 0;
  visited[z0][y0][x0]   = true;
  previous[z0][y0][x0]  = toLinearIndex(x0,y0,z0, cube);

  //Initialization of the boundary
  boundary.clear();
  boundary.insert(pair<float, int>(0, toLinearIndex(x0,y0,z0,cube)));

  //Next point to be taken into account
  int xN = x0, yN = y0, zN = z0;
  float alt;
  multimap< float, int >::iterator itb;

  int nPoints = 1; //The first point is already there
  int nIterations = 0;
  int cubeSizeTotal = cube->cubeWidth*cube->cubeHeight*cube->cubeDepth;
  // printf("nIterations: %i, the cubeSize is %i\n", nIterations, cubeSizeTotal);
  printf("Computing the distances from [%i,%i,%i]\n", x0,y0,z0);

  // Algorithm (finally)
  while(nPoints < cubeSizeTotal){
    // if(distances[zN][yN][xN] == FLT_MAX)
      // break;
    if(nIterations%100000 == 0)
      printf("nIterations: %i and point [%i,%i,%i], with distance: %f and the distance calculated of it is %f\n",
             nIterations, xN, yN, zN, distances[zN][yN][xN], distance->distance(xN,yN,zN,xN,yN,zN));
    nIterations +=1;
    itb = boundary.begin();
    boundary.erase(itb);
    int xA, yA, zA;
    for(int i = 1; i < 27; i++){
      xA = xN + nbrToIdx[i][0];
      yA = yN + nbrToIdx[i][1];
      zA = zN + nbrToIdx[i][2];
      //borders ...
      if ( (xA < 0) || (yA < 0) || (zA < 0) ||
           (xA >= cube->cubeWidth)  ||
           (yA >= cube->cubeHeight) ||
           (zA >= cube->cubeDepth ) )
        continue;

      alt = distances[zN][yN][xN] + distance->distance(xA,yA,zA,xN,yN,zN);
      // alt = distance->distance(xA,yA,zA,xN,yN,zN);
      // printf("   the alt is %f and the distance of the original is %f and the point is visited: %i and the distance to xN is %f\n",
      //                                 alt, distances[zN][yN][xN], visited[zA][yA][xA],distance->distance(xA,yA,zA,xN,yN,zN) );
      if(alt < distances[zA][yA][xA]){
        distances[zA][yA][xA] = alt;
        previous[zA][yA][xA]  = toLinearIndex(xN,yN,zN,cube);
        if(!visited[zA][yA][xA]){
          // printf("   added the point [%i,%i,%i] with distance %f\n", xA, yA, zA, alt);
          visited[zA][yA][xA]   = true;
          nPoints++;
          boundary.insert(pair<float, int>(alt, toLinearIndex(xA,yA,zA,cube)));
          // if((nPoints%5000 == 0) || (nPoints>473130))
            // printf("visited %i points and size is %i\n", nPoints, cubeSizeTotal);
        }
      }
    }
    // Get the next point on the boundary
    itb = boundary.begin();
    int linIndex = itb->second;
    toCubeIndex(linIndex, xN, yN, zN, cube);
  }
  printf("Computing the distances from [%i,%i,%i] is done\n", x0,y0,z0);
  computingDistances = false;
  // Cube<uchar, ulong>* cp = dynamic_cast< Cube<uchar, ulong>* >(cube);
  // Cube<float, double>* distances_save = cp->create_blank_cube("distances");
  // for(int x = 0; x<cube->cubeWidth; x++)
    // for(int y = 0; y < cube->cubeHeight; y++)
      // for(int z = 0; z < cube->cubeDepth; z++){
        // distances_save->put(x,y,z,distances[z][y][x]);
      // }
}



Cloud<Point3D>*
CubeLiveWire::findShortestPath
(int x0, int y0, int z0, int x1, int y1, int z1)
{
  Cloud<Point3D>* result = new Cloud<Point3D>();

  // if(! (computingDistances && (x0 == xS) && (y0 == yS) && (z0 == zS) ) ){
    // struct thread_data startPoint;
    // xS = x0; yS = y0; zS = z0;
    // computeDistances(x0,y0,z0);
  // }

  vector< int > indexes(3);
  vector< float > micrometers(3);

  if( (z1 >= 0) && (y1>=0) && (x1>=0) &&
      (z1 < cube->cubeDepth) && (y1 < cube->cubeHeight) && (x1 < cube->cubeWidth) &&
      (z0 >= 0) && (y0>=0) && (z0>=0) &&
      (z0 < cube->cubeDepth) && (y0 < cube->cubeHeight) && (x0 < cube->cubeWidth)
      && visited[z1][y1][x1]){
    // printf("The point [%i %i %i] has been visited\n", x1, y1, z1);
    int xP = x1, yP = y1, zP = z1;
    while(!( (xP==x0) && (yP==y0)&& (zP==z0) )){
      indexes[0] = xP;
      indexes[1] = yP;
      indexes[2] = zP;
      cube->indexesToMicrometers(indexes, micrometers);
      result->points.push_back(new Point3D(micrometers[0], micrometers[1], micrometers[2]));
      int previousInt = previous[zP][yP][xP];
      toCubeIndex(previous[zP][yP][xP], xP, yP, zP, cube);
      printf("%i %i %i -> %i\n", xP,yP,zP, previousInt);
    }
  }
  return result;
}






