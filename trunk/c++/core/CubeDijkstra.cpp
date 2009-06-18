#include "CubeDijkstra.h"



Cloud<Point3D>*
CubeDijkstra::findShortestPath
(int x0, int y0, int z0, int x1, int y1, int z1,
 Cloud<Point3D>& boundaryCl, pthread_mutex_t& mutex)
{
  pathFound = false;
  Cloud<Point3D>* result = new Cloud<Point3D>();
  if (cube->type == "uchar"){
    cube = dynamic_cast<Cube< uchar, ulong>* >(cube);
  } else if (cube->type == "float"){
    cube = dynamic_cast<Cube< float, double>* >(cube);
  }

  map<int, PointDijkstra*>            pointsTakenIntoAccount;
  map<int, PointDijkstra*>::iterator  pointsIt;
  map<int, PointDijkstra*>::iterator  pointsIt2;
  multimap<float, int> boundary; //Ordered by the distances

  boundary.insert(pair<float, int>(0, toLinearIndex(x0,y0,z0,cube)));

  //Next point to be analyzed
  int xN=x0, yN=y0, zN=z0;
  pointsTakenIntoAccount.insert(
          pair<int, PointDijkstra*>(
                   toLinearIndex(x0,y0,z0,cube),
                   new PointDijkstra(0,0)));

  printf("Evaluating %i %i %i\n", xN, yN, zN);
  multimap< float, int >::iterator itb;

  // Begin of the loop
  int nPointsEvaluated = 0;
  while( !((xN==x1) && (yN==y1) && (zN==z1)) ){

    nPointsEvaluated ++;

    //Eliminate the first point
    itb = boundary.begin();
    boundary.erase(itb);

    pointsIt = pointsTakenIntoAccount.find(toLinearIndex(xN,yN,zN,cube));
    PointDijkstra* peval = pointsIt->second;

    //Add the neighbors of xN,yN,zN to the list
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
      // if they have not yet been visited
      if( pointsTakenIntoAccount.find(toLinearIndex(xA,yA,zA,cube))
          == pointsTakenIntoAccount.end())
        {
          float dist = peval->distance + distance->distance(xA,yA,zA,xN,yN,zN);
          // float dist = distance->distance(xA,yA,zA,x0,y0,z0);
          // float dist = distance->distance(xA,yA,zA,xN,yN,zN);
          boundary.insert(pair<float, int>(dist,
                                           toLinearIndex(xA,yA,zA,cube)));
          pointsTakenIntoAccount.insert(
                      pair<int, PointDijkstra*>(
                             toLinearIndex(xA,yA,zA,cube),
                             new PointDijkstra(toLinearIndex(xN,yN,zN,cube), dist)));
        }
    }

    //Take the closest point of the cube in in the boundary
    multimap< float, int >::iterator it = boundary.begin();
    int linIndex = it->second;
    toCubeIndex(linIndex, xN, yN, zN, cube);

    if(&boundaryCl!=NULL){
      if((nPointsEvaluated % 5000)==0) {
        vector< int > indexes(3);
        vector< float > micrometers(3);
        // boundaryCl.points.resize(0);
        // char buff[512];
        Cloud<Point3D> boundaryCl2 = Cloud<Point3D>();
        // boundaryCl.points.resize(0);
        for(it = boundary.begin(); it != boundary.end(); it++){
          int idx = it->second;
          toCubeIndex(idx, indexes[0], indexes[1], indexes[2], cube);
          cube->indexesToMicrometers(indexes, micrometers);
          boundaryCl2.points.push_back
            (new Point3D(micrometers[0], micrometers[1], micrometers[2]));
        } //population of the boundary
        pthread_mutex_lock(&mutex);
        boundaryCl = boundaryCl2;
        pthread_mutex_unlock(&mutex);
        printf("Evaluated %i points, boundary points: %i\n", nPointsEvaluated,
               boundaryCl.points.size());
        fflush(stdout);
      }
    } //if(&boundaryCl!=NULL)

  } //while


  // And now finds the way back tracing the path
  int xP = x1, yP = y1, zP = z1;
  vector< int > indexes(3);
  vector< float > micrometers(3);
  while(!( (xP==x0) && (yP==y0)&& (zP==z0) )){
    indexes[0] = xP;
    indexes[1] = yP;
    indexes[2] = zP;
    cube->indexesToMicrometers(indexes, micrometers);
    result->points.push_back(new Point3D(micrometers[0], micrometers[1], micrometers[2]));
    printf("%i %i %i\n", xP,yP,zP);
    int idx = toLinearIndex(xP,yP,zP,cube);
    pointsIt2 = pointsTakenIntoAccount.find(idx);
    PointDijkstra* pd = pointsIt2->second;
    toCubeIndex(pd->previous, xP, yP, zP, cube);
  }
  pathFound = true;

  return result;
}


void CubeDijkstra::initializeCubePrevious(int x0, int y0, int z0)
{
  this->x0 = x0;
  this->y0 = y0;
  this->z0 = z0;
  if (cube->type == "uchar"){
    cube = dynamic_cast<Cube< uchar, ulong>* >(cube);
  } else if (cube->type == "float"){
    cube = dynamic_cast<Cube< float, double>* >(cube);
  }

  map<int, PointDijkstra*>            pointsTakenIntoAccount;
  map<int, PointDijkstra*>::iterator  pointsIt;
  map<int, PointDijkstra*>::iterator  pointsIt2;
  multimap<float, int> boundary; //Ordered by the distances

  boundary.insert(pair<float, int>(0, toLinearIndex(x0,y0,z0,cube)));

  //Next point to be analyzed
  int xN=x0, yN=y0, zN=z0;
  int xA, yA, zA;
  pointsTakenIntoAccount.insert(
          pair<int, PointDijkstra*>(
                   toLinearIndex(x0,y0,z0,cube),
                   new PointDijkstra(0,0)));
  PointDijkstra* peval;

  printf("Starting Dijkstra from %i %i %i\n", xN, yN, zN);
  printf("And I have been modified\n");
  multimap< float, int >::iterator itb;

  // Begin of the loop
  int nPointsEvaluated = 0;
  int nTotalPoints     = cube->cubeWidth*cube->cubeHeight*cube->cubeDepth;
  int nPointsPerLayer  = cube->cubeWidth*cube->cubeHeight;
  while( nPointsEvaluated <  nTotalPoints){

    nPointsEvaluated ++;

    //Eliminate the first point
    itb = boundary.begin();
    boundary.erase(itb);

    pointsIt = pointsTakenIntoAccount.find(toLinearIndex(xN,yN,zN,cube));
    peval = pointsIt->second;

    //Add the neighbors of xN,yN,zN to the list
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
      // if they have not yet been visited
      if( pointsTakenIntoAccount.find(toLinearIndex(xA,yA,zA,cube))
          == pointsTakenIntoAccount.end())
        {
          float dist = peval->distance + distance->distance(xA,yA,zA,xN,yN,zN);
          // float dist = distance->distance(xA,yA,zA,x0,y0,z0);
          // float dist = distance->distance(xA,yA,zA,xN,yN,zN);
          boundary.insert(pair<float, int>(dist,
                                           toLinearIndex(xA,yA,zA,cube)));
          pointsTakenIntoAccount.insert(
                      pair<int, PointDijkstra*>(
                             toLinearIndex(xA,yA,zA,cube),
                             new PointDijkstra(toLinearIndex(xN,yN,zN,cube), dist)));
          previous_idx->put(xA,yA,zA,toLinearIndex(xN,yN,zN,cube));
        }
    }
    //Take the closest point of the cube in in the boundary
    multimap< float, int >::iterator it = boundary.begin();
    int linIndex = it->second;
    toCubeIndex(linIndex, xN, yN, zN, cube);

    if(nPointsEvaluated%100000 == 0){
      printf("Evaluated pointr %i\n", nPointsEvaluated);
    }

  } //while
}


Cloud<Point3D>* CubeDijkstra::traceBack(int x1, int y1, int z1)
{
  pathFound = false;
  Cloud<Point3D>* result = new Cloud<Point3D>();
  int xP = x1, yP = y1, zP = z1;
  vector< int > indexes(3);
  vector< float > micrometers(3);
  while(!( (xP==x0) && (yP==y0) && (zP==z0) )){
    indexes[0] = xP;
    indexes[1] = yP;
    indexes[2] = zP;
    cube->indexesToMicrometers(indexes, micrometers);
    result->points.push_back(new Point3D(micrometers[0], micrometers[1], micrometers[2]));
    printf("%i %i %i\n", xP,yP,zP);
    int prev = previous_idx->at(xP,yP,zP);
    toCubeIndex(prev, xP, yP, zP, cube);
  }
  pathFound = true;
  return result;

}
