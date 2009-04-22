#include "CubeDijkstra.h"



Cloud<Point3D>*
CubeDijkstra::findShortestPath
(int x0, int y0, int z0, int x1, int y1, int z1)
{
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
          == pointsTakenIntoAccount.end()) {
        // int dist = peval->distance + distance(cube,xA,yA,zA,xN,yN,zN);
        // float dist = distance->distance(cube,xA,yA,zA,x0,y0,z0);
        //FIXMEEE
        float dist = distance->distance(xA,yA,zA,xN,yN,zN);
        boundary.insert(pair<float, int>( dist,
                                       toLinearIndex(xA,yA,zA,cube)));
        pointsTakenIntoAccount.insert(
                            pair<int, PointDijkstra*>(
                                toLinearIndex(xA,yA,zA,cube),
                                new PointDijkstra(toLinearIndex(xN,yN,zN,cube), dist)));
      }
    }

    //Take the closest point of the cubein in the boundary
    multimap< float, int >::iterator it = boundary.begin();
    int linIndex = it->second;
    toCubeIndex(linIndex, xN, yN, zN, cube);

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
          toCubeIndex(idx, indexes[0], indexes[1], indexes[2], cube);
          cube->indexesToMicrometers(indexes, micrometers);
          cloud->points.push_back(new Point3D(micrometers[0], micrometers[1], micrometers[2]));
        }
        cloud->saveToFile(buff);
      }
    } //if(save)
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

  return result;
}
