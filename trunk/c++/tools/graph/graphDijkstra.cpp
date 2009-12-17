
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
#include "Graph.h"
#include <float.h>
#include <omp.h>

using namespace std;

typedef Graph<Point3D, EdgeW<Point3D> > Graph3D;

//Auxiliary class for the points in the dijkstra algorithm
class PD
{
public:
  int idx;
  int prev;
  PD(int _idx, int _prev){
    idx = _idx; prev = _prev;
  }
};


float maxValueMatrix
(vector< vector< float > >& matrix)
{
  float maxVal = FLT_MIN;
  for(int i = 0; i < matrix.size(); i++)
    for(int j = 0; j < matrix[i].size(); j++)
      if(matrix[i][j] > maxVal) maxVal = matrix[i][j];
  return maxVal;
}

void computeAuxStructures
( Graph<Point3D, EdgeW<Point3D> >* gr,
  vector< vector< float > >& distanceMatrix,
  vector< vector< int   > >& neighbors)
{
  int nPoints = gr->cloud->points.size();
  distanceMatrix.resize(nPoints);
  for(int i = 0; i < nPoints; i++){
    distanceMatrix[i].resize(nPoints);
  }
  for(int i = 0; i < nPoints; i++)
    for(int j = 0; j < nPoints; j++)
      distanceMatrix[i][j] = FLT_MAX;
  neighbors.resize(nPoints);
  for(int i = 0; i < gr->eset.edges.size(); i++){
    neighbors[gr->eset.edges[i]->p0].push_back(gr->eset.edges[i]->p1);
    neighbors[gr->eset.edges[i]->p1].push_back(gr->eset.edges[i]->p0);
    EdgeW<Point3D> * edg = dynamic_cast<EdgeW<Point3D>* >(gr->eset.edges[i]);
    distanceMatrix[gr->eset.edges[i]->p0][gr->eset.edges[i]->p1] = edg->w;
    distanceMatrix[gr->eset.edges[i]->p1][gr->eset.edges[i]->p0] = edg->w;
  }
}


void runDijkstra
(Graph<Point3D, EdgeW<Point3D> >* gr,
 int sourceNode,
 vector< float >& distances,
 vector< int   >& previous,
 vector< vector< float > >& distanceMatrix,   //to speed up computation
 vector< vector< int   > >& neighbours)
{
  int nPoints = gr->cloud->points.size();
  distances.resize(nPoints);
  previous .resize(nPoints);
  vector<char> visited(nPoints);
  for(int i = 0; i < nPoints; i++){
    distances[i] = FLT_MAX;
    previous[i]  = -1;
    visited[i]   =  0;
  }
  multimap<float, PD> boundary; //keeps the priority queue
  boundary.insert(pair<float, PD>(0, PD(sourceNode, 0) ) );
  distances[sourceNode] = 0;
  previous [sourceNode] = 0;
  visited  [sourceNode] = 0;

  multimap< float, PD >::iterator itb;
  int pit; //point iteration
  int previt;
  float cit;
  int counter = 0;
  while(!boundary.empty()){
    itb = boundary.begin();  //pop
    cit = itb->first;
    PD tmp = itb->second;
    pit = tmp.idx;
    previt = tmp.prev;
    boundary.erase(itb);
    if(visited[pit]==1)
      continue; //the point is already evaluated
    visited  [pit] = 1;
    distances[pit] = cit;
    previous [pit] = previt;
    counter++;
    //And now expand the point
    for(int i = 0; i < neighbours[pit].size(); i++){
      if(!visited[neighbours[pit][i]]){
        boundary.insert(pair<float, PD>
                       (cit+distanceMatrix[pit][neighbours[pit][i]],
                        PD(neighbours[pit][i], pit)));
      }
    }
  }
  printf("Path for point %03i done\r", sourceNode);
}

void traceBack
(int sourceNode,
 int nodeToStart,
 vector<int>& previous,
 vector<int>& path)
{
  int nodeT = nodeToStart;
  path.resize(0);
  path.push_back(nodeT);
  while(nodeT != sourceNode){
    nodeT = previous[nodeT];
    if(nodeT == -1){
      printf("There is something awfully wrong\n");
      break;
    }
    path.push_back(nodeT);
  }
}

void allShortestPaths
( Graph<Point3D, EdgeW<Point3D> >* gr,
  vector< vector< vector< int   > > >& paths,
  vector< vector< float > >& costs)
{
  // Temporal variables
  vector< vector< float > > distanceMatrix;
  vector< vector< int   > > neighbors;
  vector< float > distances;
  vector< int   > previous ;
  vector< int   > path;

  computeAuxStructures(gr, distanceMatrix, neighbors);

  float maxEdgeVal = maxValueMatrix(distanceMatrix);

  // Output
  int nPoints = gr->cloud->points.size();
  paths.resize(nPoints); costs.resize(nPoints);
  for(int i = 0; i < nPoints; i++){
    paths[i].resize(nPoints); costs[i].resize(nPoints);
  }

// #pragma omp parallel for
  for(int i = 0; i < nPoints; i++){
    runDijkstra(gr, i, distances, previous, distanceMatrix, neighbors);
    for(int j = 0; j < nPoints; j++){
      traceBack(i, j, previous, path);
      for(int nE = 0; nE < path.size(); nE++)
        (paths[i][j]).push_back(path[nE]);
      //###################################### HERE #####################################
      //Here is the cost put
      // costs[i][j] = (distances[j]+maxEdgeVal)/(paths[i][j].size()+1);
      costs[i][j] = (distances[j]+10)/(paths[i][j].size()+1);
      // costs[i][j] = 4*(distances[j]+maxEdgeVal)/(paths[i][j].size()+1);
    }
  }
  printf("\n");
}

void printVector(vector<int>& s){
  for(int i = 0; i < s.size(); i++)
    printf("%i ", s[i]);
  printf("\n");
}

void printSolution(vector<int>& s){
  for(int i = 0; i < s.size(); i++)
    if(s[i] > -1)
      printf("%i ", i);
  printf("\n");
}

bool solutionContains(vector<int>& solution, int n){
  if(solution[n] >= 0)
      return true;
  return false;
}

//The return value is:  0 (none of the starting / end points are in the graph
//                      1 the path is compatible
//                     -1 the path has one of the middle elements in
int isCompatible(vector<int>& S, vector< int>& path)
{
  for(int i = 1; i < path.size()-1; i++)
    if(solutionContains(S, path[i]))
      return -1;
  if(! (solutionContains(S, path[0]) ^
        solutionContains(S, path[path.size()-1]) )
     )
    return 0;
  return 1;
}



void addToSolution(vector< int >& S, vector<int>& path){
  if(solutionContains(S, path[0])){
    for(int i = 1; i < path.size(); i++)
      S[path[i]] = path[i-1];
  }
  else if(solutionContains(S, path[path.size()-1])){
    for(int i = path.size()-2; i >= 0; i--)
      S[path[i]] = path[i+1];
  }
  else{
    S[path[0]] = path[0]; //defines the root
    for(int i = 1; i < path.size(); i++)
      S[path[i]] = path[i-1];
  }
}


Graph<Point3D, EdgeW<Point3D> >*
solutionToGraph
(Graph<Point3D, EdgeW<Point3D> >* cpt,
 vector< int > solution)
{
  Graph<Point3D, EdgeW<Point3D> >* toRet =
    new Graph<Point3D, EdgeW<Point3D> >(cpt->cloud);
  for(int i = 0; i < solution.size(); i++){
    if((solution[i]!=i) && (solution[i] > -1)){
      int nE = cpt->eset.findEdgeBetween(i, solution[i]);
      toRet->eset.edges.push_back
        ( new EdgeW<Point3D>(&cpt->cloud->points, i, solution[i],
                             cpt->eset.edges[nE]->w));
    }
  }
  return toRet;
}

void addSomaToCptGraphAndInitializeSolution
(Graph3D* gr, float xS, float yS, float zS, float R,
 vector< int >& S)
{
  int   pointSoma = gr->cloud->findPointClosestTo(xS,yS,zS);
  vector< int > pointsInSoma = gr->cloud->findPointsCloserToThan(xS,yS,zS,R);

  //Removes all the edges between points in the soma
  for(int i = 0; i < pointsInSoma.size(); i++){
    for(int j = 0; j < pointsInSoma.size(); j++){
      int nE = gr->eset.findEdgeBetween(pointsInSoma[i], pointsInSoma[j]);
      if( nE != -1)
        gr->eset.edges.erase(gr->eset.edges.begin() + nE);
    }
  }

  S[pointSoma] = pointSoma;

  for(int i = 0; i < pointsInSoma.size(); i++)
    if( pointsInSoma[i] != pointSoma){
      gr->eset.edges.push_back
        (new EdgeW<Point3D>(&gr->cloud->points, pointSoma, pointsInSoma[i], 0));
      S[pointsInSoma[i]] = pointSoma;
    }

}



int main(int argc, char **argv) {

  if(argc!=4){
    printf("graphDijkstra complete.gr out.gr solsDirectory\n");
    exit(0);
  }

  string solsDirectory(argv[3]);

  float xS = 27.7;
  float yS = -64.7;
  float zS = 30.4;
  float R  = 20;

  Graph<Point3D, EdgeW<Point3D> >* gr =
    new Graph<Point3D, EdgeW<Point3D> >(argv[1]);

  //From now on improvisation
  int nPoints = gr->cloud->points.size();
  vector< int >   S(nPoints); //solution
  for(int i = 0; i < nPoints; i++)
    S[i] = -1;

  // Initialization of the solution
  addSomaToCptGraphAndInitializeSolution
    (gr, xS, yS, zS, R, S);


  //paths is going to be the main data structure
  vector< vector< vector< int   > > > paths;
  vector< vector< float > > costs;
  allShortestPaths(gr, paths, costs);

  printf("And now merging stuff\n");
  multimap<float, int> Q; //Priority queue

  //Initialization - all to Q
  for(int i = 0; i < nPoints; i++)
    for(int j = i+1; j < nPoints; j++)
      Q.insert(pair<float, int>(costs[i][j], i*nPoints+j));


  //Puts the most probable path as the solution already
  //old
  // addToSolution(S, paths[it->second/nPoints]
                // [it->second - nPoints*floor(it->second/nPoints)]);
  // Q.erase(it);

  //Puts the soma into the solution


  multimap<float, int>::iterator it = Q.begin();
  printf("Initialization done\n");
  printf("S: "); printSolution(S);

  //And now the algorithm
  int counter=0;
  bool weAreDone=false;
  int nComponentsAdded = 0;
  Graph3D* sols;
  char solsName[512];
  while(!Q.empty()){
    bool thereIsSomethingCompatible = false;
    vector< multimap<float, int>::iterator  > markedForDeletion;
    int elementNumber = 0;
    for(it = Q.begin(); it != Q.end(); ++it){
      int isCompatibleN = isCompatible(S,paths[floor(it->second/nPoints)]
                                      [it->second - nPoints*floor(it->second/nPoints)]);
      if( isCompatibleN == 1)
        {
          if(nComponentsAdded == 1000)
            weAreDone = true;
          addToSolution(S,paths[floor(it->second/nPoints)]
                        [it->second - nPoints*floor(it->second/nPoints)]);
          // printf("%i: ",nComponentsAdded);
          // printVector(paths[floor(it->second/nPoints)]
                      // [it->second - nPoints*floor(it->second/nPoints)]);
          // printf("S: ");
          // printSolution(S);
          sols = solutionToGraph(gr, S);
          sprintf(solsName, "%s/sol_%03i.gr",solsDirectory.c_str(), nComponentsAdded);
          sols->saveToFile(solsName);
          Q.erase(it);
          nComponentsAdded++;
          thereIsSomethingCompatible = true;
          break;
        }
      else if (isCompatibleN == -1){
        markedForDeletion.push_back(it);
      }
      elementNumber++;
    }//for
    //Elliminate the elements of Q
    for(int i = markedForDeletion.size()-1; i >= 0; i--)
      Q.erase(markedForDeletion[i]);
    // printf("We have erased %i elements from Q that now has %i elements\n",
           // markedForDeletion.size(), Q.size());
    if(!thereIsSomethingCompatible || weAreDone) break;
    if(counter++%100 == 0)
      printf("Q has %i elements\n", Q.size());
  }//Iteration over Q

  // printVector(S);
  Graph<Point3D, EdgeW<Point3D> >* sol =
    solutionToGraph(gr, S);
  sol->saveToFile(argv[2]);
  
}
