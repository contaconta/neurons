
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

// Code done at Cornell University

#include <iostream>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "Graph.h"
#include <float.h>
#include <omp.h>
#include "SWC.h"
#include "CubeFactory.h"

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

// auxiliary mathematical functions
float maxValueMatrix
(vector< vector< float > >& matrix)
{
  float maxVal = FLT_MIN;
  for(int i = 0; i < matrix.size(); i++)
    for(int j = 0; j < matrix[i].size(); j++)
      if(matrix[i][j] > maxVal) maxVal = matrix[i][j];
  return maxVal;
}

//return the dot product
float v_dot(vector< float >& a, vector< float >& b)
{
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

vector< float > v_subs(vector< float >& a, vector< float >& b)
{
  vector< float > toRet(3);
  toRet[0] = a[0] - b[0];
  toRet[1] = a[1] - b[1];
  toRet[2] = a[2] - b[2];
  return toRet;
}

vector< float > v_add(vector< float >& a, vector< float >& b)
{
  vector< float > toRet(3);
  toRet[0] = a[0] + b[0];
  toRet[1] = a[1] + b[1];
  toRet[2] = a[2] + b[2];
  return toRet;
}


vector< float > v_scale(vector< float >& a, float scale)
{
  vector< float > toRet(3);
  toRet[0] = scale*a[0];
  toRet[1] = scale*a[1];
  toRet[2] = scale*a[2];
  return toRet;
}

float v_norm(vector< float >& a)
{
  return sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
}



// distanceMatrix is the edge value between the point i and j in the matrix
// neighbors is a vector that contains for each point all the other points that are in direct connection
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

//Conputes shortest path between the sourceNode and all the others in the complete graph
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
  printf("Path for point %03i done\n", sourceNode);
}

// Traces the shortest path between the sourceNode and a terminal node
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

//Computes the cost of a path. Includes image and geometrical information
// this function computes the distance from each point to the line between the first
// and ending points of the graph. Not very smart
float computePathCostLine
( Graph<Point3D, EdgeW<Point3D> >* gr,
  vector< int >& path,
  float imageCost
  )
{
  float geomCost = 0;
  // Computes the unit vector of the line
  vector< float > p0 = gr->cloud->points[path[0]]->coords;
  vector< float > p1 = gr->cloud->points[path[path.size()-1]]->coords;
  vector< float > p1p0 = v_subs(p1, p0);
  float mp1p0 = v_norm(p1p0);
  vector< float > p1p0n = v_scale(p1p0, 1.0/mp1p0);

  for(int i = 0; i < path.size(); i++){
    vector< float > pm = gr->cloud->points[path[i]]->coords;
    vector< float > pmp0 = v_subs(pm, p0);
    float vdpa  = v_dot(pmp0,p1p0n);
    vector< float > pa   = v_scale(p1p0n, vdpa);
    pa   = v_add(p0, pa);
    vector< float > pmpa = v_subs(pm, pa);
    float dpa  = v_norm(pmpa);
    geomCost += dpa*dpa;
  }

  return (imageCost + geomCost*0.1 + 250)/(path.size()+1);
}


// The geometrical cost will increase heavily with the angle
float computePathCost
( Graph<Point3D, EdgeW<Point3D> >* gr,
  vector< int >& path,
  float imageCost
  )
{
  float geomCost = 0;
  // Computes the unit vector of the line

  for(int i = 1; i < path.size()-1 ; i++){
    vector< float > p0 = gr->cloud->points[path[i-1]]->coords;
    vector< float > p1 = gr->cloud->points[path[i  ]]->coords;
    vector< float > p2 = gr->cloud->points[path[i+1]]->coords;
    vector< float > p1p0 = v_subs(p1, p0);
    vector< float > p1p0n = v_scale(p1p0, v_norm(p1p0));
    vector< float > p2p1 = v_subs(p2, p1);
    vector< float > p2p1n = v_scale(p2p1, v_norm(p2p1));
    float cos_alpha = v_dot(p1p0n,p2p1n);
    // if(cos_alpha < 0){
      //We do not allow that
      // return FLT_MAX;
    // }
    geomCost -= log10(fabs(cos_alpha));
  }
  return (imageCost + geomCost + 50)/(path.size()+1);
}


float edgeLogProb
(Point* _p0, Point* _p1, Cube<float, double>* probs)
{
  int x0, y0, z0, x1, y1, z1;
  probs->micrometersToIndexes3
    (_p0->coords[0], _p0->coords[1], _p0->coords[2], x0, y0, z0);
  probs->micrometersToIndexes3
    (_p1->coords[0], _p1->coords[1], _p1->coords[2], x1, y1, z1);
  float dist = sqrt((double)
                    (x0-x1)*(x0-x1) +
                    (y0-y1)*(y0-y1) +
                    (z0-z1)*(z0-z1));
  float p1 = probs->at(x0,y0,z0);
  float p2 = probs->at(x1,y1,z1);
  if(fabs(p1-p2) < 1e-4) return -dist*log10(p1);
  return fabs(dist*((log10(p1) * p1 - p1- log10(p2) * p2 + p2) / (-p2 + p1)));
}

float edgeGeomLogProb(Point* _p0, Point* _p1, Point* _p2){
    vector< float > p0 = _p0->coords;
    vector< float > p1 = _p1->coords;
    vector< float > p2 = _p2->coords;
    vector< float > p1p0 = v_subs(p1, p0);
    vector< float > p1p0n = v_scale(p1p0, v_norm(p1p0));
    vector< float > p2p1 = v_subs(p2, p1);
    vector< float > p2p1n = v_scale(p2p1, v_norm(p2p1));
    float cos_alpha = v_dot(p1p0n,p2p1n);
    // if(cos_alpha < 0){
      //We do not allow that
      // return FLT_MAX;
    // }
    return -log10(fabs(cos_alpha));

}


float computePathCost
( Graph<Point3D, EdgeW<Point3D> >* gr,
  vector< int >& path,
  vector< vector< Graph3D* > >& v2v_paths,
  Cube<float, double>* probs
  )
{
  float imageEvidence = 0;
  float geometryEvidence = 0;
  int nElements = 0;
  Graph3D* dsp1;
  Graph3D* dsp2;

  //Loops for all the edges of the path
  // Computes the image evidence
  for(int nep = 0; nep < path.size()-1; nep++){
    dsp2 = v2v_paths[path[nep]][path[nep+1]];
    nElements += dsp2->cloud->points.size()-1;
    if(nep >= 1) dsp1 = v2v_paths[path[nep-1]][path[nep]];
    for(int nP = 0; nP < dsp2->cloud->points.size()-1; nP++){
      imageEvidence +=
        edgeLogProb
        (dsp2->cloud->points[nP], dsp2->cloud->points[nP+1], probs);
      // if(nP > 0)
        // geometryEvidence +=
          // edgeGeomLogProb(dsp2->cloud->points[nP-1],
                          // dsp2->cloud->points[nP],
                        // dsp2->cloud->points[nP+1]);
      if((nP == 0) && (nep > 0)){
        geometryEvidence +=
          edgeGeomLogProb(dsp1->cloud->points[dsp1->cloud->points.size()-1],
                          dsp2->cloud->points[nP],
                          dsp2->cloud->points[nP+1]);
      }
    }
  }

  return (imageEvidence + geometryEvidence + 200)/nElements;

}



// Computes all the shortest paths between all pairs of points and assign them a cost
void allShortestPaths
( Graph<Point3D, EdgeW<Point3D> >* gr,
  vector< vector< vector< int   > > >& paths,
  vector< vector< float > >& costs,
  vector< vector< Graph3D* > >& v2v_paths,
  Cube<float, double>* probs
)
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
      //creates all the possible point to point pahts
      for(int nP = 0; nP < path.size(); nP++){
        paths[i][j].push_back(path[nP]);
      }
      //###################################### HERE #####################################
      //Here is the cost
      // costs[i][j] = (distances[j]+maxEdgeVal)/(paths[i][j].size()+1);
      // costs[i][j] = (distances[j]+10)/(paths[i][j].size()+1);
      // costs[i][j] = 4*(distances[j]+maxEdgeVal)/(paths[i][j].size()+1);
      // costs[i][j] = computePathCost(gr, paths[i][j], distances[j]);
      costs[i][j] = computePathCost(gr, paths[i][j], v2v_paths, probs);
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
      printf("%i %i\n", i, s[i]);
  printf("\n");
}


bool checkSolutionForLoopsAux
(vector<int>& solution,
 vector< vector< int > >& kids,
 vector< int >& visited,
 int np)
{
  //The point has already visited -> loop
  if(visited[np] == 1)
    return false;

  // Mark the point as visited
  visited[np] = 1;

  //If it is a leaf, then it is ok
  if( kids[np].size() == 0){
    return true;
  }

  // If not recursively traverse the treer
  bool toReturn = true;
  for(int i = 0; i < kids[np].size(); i++){
    bool kidCreatesLoops = checkSolutionForLoopsAux
      (solution, kids, visited, kids[np][i]);
    toReturn = toReturn & kidCreatesLoops;
  }
  return toReturn;

}

bool checkSolutionForLoops(vector<int>& solution)
{
  vector< vector< int > > kids(solution.size());
  for(int i = 0; i < solution.size(); i++){
    if( (solution[i] != -1) && (solution[i]!=i))
      kids[solution[i]].push_back(i);
  }
  vector< int > visited(solution.size());
  return checkSolutionForLoopsAux
    (solution, kids, visited, 0);

}

bool checkSolutionIsBinaryTree(vector< int >& solution)
{
  vector< vector< int > > kids(solution.size());
  for(int i = 0; i < solution.size(); i++){
    if( (solution[i] != -1) && (solution[i]!=i))
      kids[solution[i]].push_back(i);
  }
  for(int i = 0; i < kids.size(); i++)
    if(kids[i].size() > 2)
      return false;
  return true;
}

bool checkSolution(vector<int>& solution){

  // printf("checkSolution: solutionSize: %i\n", solution.size());
  // vector< int > numberOfTimesVisited(solution.size());
  // for(int i = 1; i < solution.size(); i++){
    // if(solution[i] != -1)
      // numberOfTimesVisited[solution[i]]++;
  // }
  // printf("SolutionCheck: ");
  // printVector(numberOfTimesVisited);
  bool noLoops = checkSolutionForLoops(solution);
  bool isBinary = checkSolutionIsBinaryTree(solution);
  // printf("SolutionHasLoopsPassed: %i\n", noLoops);
  // printf("SolutionIsBinaryTree: %i\n", isBinary);
  return noLoops & isBinary;
}



bool solutionContains(vector<int>& solution, int n){
  if(solution[n] >= 0)
      return true;
  return false;
}

//The return value is:  0 (none of the starting / end points are in the graph
//                      1 the path is compatible
//                     -1 the path has one of the middle elements in
int isCompatible(vector<int>& S, vector< int>& path, vector< int >& kids)
{
  //if any middle point is part of the solution, then the path is not compatible
  for(int i = 1; i < path.size()-1; i++)
    if(solutionContains(S, path[i]))
      return -1;

  // if no point is part of the solution, then we have no idea
  if( !(solutionContains(S, path[0])) &&
        !(solutionContains(S, path[path.size()-1])) )
    return 0;

  //if both ends are part of the solution, it is not compatible
  if((solutionContains(S, path[0]) &&
      solutionContains(S, path[path.size()-1]) )
     )
    return -1;

  //If the end point links somewhere with two edges already, the solution is incompatible
  if( (solutionContains(S, path[path.size()-1]) &&
       kids[path[path.size()-1]] >= 2) )
    return -1;
  if( (solutionContains(S, path[0]) &&
       kids[path[0]]>= 2) )
    return -1;

  if( (solutionContains(S, path[0]) &&
       kids[path[0]]< 2) )
    return 1;
  if( (solutionContains(S, path[path.size()-1]) &&
       kids[path[path.size()-1]] < 2) )
    return 1;

  printf("isCompatible failed\n");
  exit(0);
  return -1;
}



void addToSolution(vector< int >& S, vector<int>& path, vector< int >& kids){
  if(solutionContains(S, path[0])){
    for(int i = 1; i < path.size(); i++){
      kids[path[i-1]]++;
      if(S[path[i]]!=-1){
        printf("We are adding a node that already has been visited, exit\n");
        exit(0);
      }
      S[path[i]] = path[i-1];
    }
  }
  else if(solutionContains(S, path[path.size()-1])){
    for(int i = path.size()-2; i >= 0; i--){
      kids[path[i+1]]++;
      if(S[path[i]]!=-1){
        printf("We are adding a node that already has been visited, exit\n");
        exit(0);
      }
      S[path[i]] = path[i+1];
    }
  }
  else{
    printf("The path added to the solution does is not compatible with the solution\n");
    exit(0);
  }
  // else{
    // S[path[0]] = path[0]; //defines the root
    // for(int i = 1; i < path.size(); i++)
      // S[path[i]] = path[i-1];
  // }
}


// Translates a solution to a graph
Graph<Point3D, EdgeW<Point3D> >*
solutionToGraph
(Graph<Point3D, EdgeW<Point3D> >* cpt,
 vector< int > solution)
{
  Graph<Point3D, EdgeW<Point3D> >* toRet =
    new Graph<Point3D, EdgeW<Point3D> >(cpt->cloud);
  for(int i = 0; i < solution.size(); i++){
    if((solution[i]!=i) && (solution[i] != -1)){
      int nE = cpt->eset.findEdgeBetween(i, solution[i]);
      toRet->eset.edges.push_back
        ( new EdgeW<Point3D>(&toRet->cloud->points, i, solution[i],
                             1.0));
    }
  }
  return toRet;
}

// Translates a solution to an SWC file
SWC*
solutionToSWC
(Graph<Point3D, EdgeW<Point3D> >* cpt,
 Cube_P* cp,
 vector< int > solution)
{

  //Obtains the graph for the
  Graph<Point3D, EdgeW<Point3D> >* toRet =
    solutionToGraph(cpt, solution);

  int somaIdx;
  for(int i = 0; i < solution.size(); i++){
    if(solution[i]!=i) somaIdx = i;
  }

  Graph<Point3Dw, Edge<Point3Dw> >* forSWC =
    new Graph<Point3Dw, Edge<Point3Dw> >();
  int x, y, z;
  for(int i = 0; i < toRet->cloud->points.size(); i++){
    cp->micrometersToIndexes3
      (toRet->cloud->points[i]->coords[0],
       toRet->cloud->points[i]->coords[1],
       toRet->cloud->points[i]->coords[2], x, y, z);
    forSWC->cloud->points.push_back
      (new Point3Dw
       (x, y, z, 1));
  }
  for(int i = 0; i < toRet->eset.edges.size(); i++)
    forSWC->eset.edges.push_back
      (new Edge<Point3Dw>
       (&forSWC->cloud->points,
        toRet->eset.edges[i]->p0,
        toRet->eset.edges[i]->p1));

  SWC* swc = new SWC();
  swc->gr = forSWC;
  swc->idxSoma = 0;
  return swc;
}





// void addPointSomaToSolution
// (Graph3D* gr, float xS, float yS, float zS, vector< int >& solution)
// {
  // int   pointSoma = gr->cloud->findPointClosestTo(xS,yS,zS);
  // S[pointSoma] = pointSoma;
// }


// Adds the some to the complete graph and creates a star graph arround it
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

void loadAllv2vPaths
(string pathsDirectory,
 Graph3D* gr,
 vector< vector< Graph3D* > >& v2v_paths)
{
  v2v_paths.resize(gr->cloud->points.size());
 for(int i = 0; i < gr->cloud->points.size(); i++){
    v2v_paths[i].resize(gr->cloud->points.size());
    for(int j = 0; j < v2v_paths[i].size(); j++)
      v2v_paths[i][j] = NULL;
  }
 for(int nE = 0; nE < gr->eset.edges.size(); nE++){
   int p0 = gr->eset.edges[nE]->p0;
   int p1 = gr->eset.edges[nE]->p1;
   printf("Loading the path between vertex %i and %i in directory %s, quit\n",
          p0, p1, pathsDirectory.c_str());

   Graph3D* graph;
   char buff[1024];
   sprintf(buff, "%s/path_%04i_%04i.gr", pathsDirectory.c_str(), p0, p1);
   if(fileExists(buff)){
     graph = new Graph3D(buff);}
   else{
     sprintf(buff, "%s/path_%04i_%04i.gr", pathsDirectory.c_str(), p1, p0);
     if(fileExists(buff)){
       graph = new Graph3D(buff);}
     else{
       printf("I can not find the path between vertex %i and %i in directory %s, quit\n",
              p0, p1, pathsDirectory.c_str());
       exit(0);
     }
   }
   v2v_paths[p0][p1] = graph;
   v2v_paths[p1][p0] = graph;
 }
}


/*******************************************************************************
 * S is the tree that is the solution of the problem. Stored as a vector of ints
 *    where each dimension represents point i and the value of that dimension is
 *    -1 if the point is not connected to another one or >0 if the point is
 *    connected to another one
 * paths is a matrix that encodes all the paths between point [i] and [j]
 *    those paths are encoded as a list of sequential points indexes in the graph
 * costs stores the cost associated with the paths
 * Q is the priority queue (implemented as a multimap) of the paths. It stores
 *    the indexes to the paths that should be added
 *******************************************************************************/
int main(int argc, char **argv) {

  bool saveAllSolutions = false;
  if(argc!=9){
    printf("graphDijkstra complete.gr out.gr somaX somaY somaZ"
           " cubeProbs solsDirectory pathsDirectory\n");
    exit(0);
  }

  Graph<Point3D, EdgeW<Point3D> >* gr =
    new Graph<Point3D, EdgeW<Point3D> >(argv[1]);
    string nameOut(argv[2]);

  float xS = atof(argv[3]);
  float yS = atof(argv[4]);
  float zS = atof(argv[5]);
  float R  = 20;
  Cube<float, double>* cp = new Cube<float, double>(argv[6]);
  string solsDirectory(argv[7]);
  string pathsDirectory(argv[8]);
  saveAllSolutions = true;
  if(!directoryExists(solsDirectory))
    makeDirectory(solsDirectory);
  printf("I should save all solutions\n");


  // float xS = -1822.71;
  // float yS = 373.92;
  // float zS = 2.00;
  // float R  = 20;

  // float xS = 27.7;
  // float yS = -64.7;
  // float zS = 30.4;
  // float R  = 20;



  //Initialization of the solution and auxiliary structures
  int nPoints = gr->cloud->points.size();
  vector< int >   S(nPoints); //solution
  vector< int >   kids(nPoints); //stores the number of kids per point
  for(int i = 0; i < nPoints; i++){
    S[i] = -1; kids[i] = 0;
  }

  // addSomaToCptGraphAndInitializeSolution
    // (gr, xS, yS, zS, R, S);
  // addPointSomaToSolution
    // (gr, xS, yS, zS, S);
  S[0]=0;


  //paths is going to be the main data structure. Stores the shortest connection among all points.
  vector< vector< vector< int   > > > paths;
  vector< vector< float > > costs;
  vector< vector< Graph3D* > > v2v_paths;
  loadAllv2vPaths(pathsDirectory, gr, v2v_paths);

  allShortestPaths(gr, paths, costs, v2v_paths, cp);

  printf("And now merging stuff\n");
  multimap<float, int> Q; //Priority queue

  //Initialization - all paths into Q with their costs
  for(int i = 0; i < nPoints; i++)
    for(int j = i+1; j < nPoints; j++)
      Q.insert(pair<float, int>(costs[i][j], i*nPoints+j));


  multimap<float, int>::iterator it = Q.begin();
  printf("Initialization done\n");
  printf("S: "); printSolution(S);

  //And now the algorithm
  int counter=0;
  bool weAreDone=false;
  int nComponentsAdded = 0;
  Graph3D* sols;
  SWC* sols_swc;
  char solsName[512];
  while(!Q.empty()){
    bool thereIsSomethingCompatible = false;
    vector< multimap<float, int>::iterator  > markedForDeletion;
    int elementNumber = 0;
    for(it = Q.begin(); it != Q.end(); ++it){
      int isCompatibleN = isCompatible(S,paths[floor(it->second/nPoints)]
                                       [it->second - nPoints*floor(it->second/nPoints)],
                                       kids);
      if( isCompatibleN == 1)
        {

          printf("nComponentsAdded=%i, kids[1]=%i\n", nComponentsAdded, kids[1]);
          addToSolution(S,paths[floor(it->second/nPoints)]
                        [it->second - nPoints*floor(it->second/nPoints)],
                        kids);

          // printf("%i: ",nComponentsAdded);
          // printVector(paths[floor(it->second/nPoints)]
                      // [it->second - nPoints*floor(it->second/nPoints)]);
          // printf("S: ");
          // printSolution(S);
          if(saveAllSolutions){
            if(nComponentsAdded > 100) continue;
            printf("Saving solution %i\n", nComponentsAdded);
            sols = solutionToGraph(gr, S);
            sprintf(solsName, "%s/sol_%03i.gr",solsDirectory.c_str(), nComponentsAdded);
            sols->saveToFile(solsName);
            sprintf(solsName, "%s/sol_%03i.swc",solsDirectory.c_str(), nComponentsAdded);
            sols_swc = solutionToSWC(gr, cp, S);
            sols_swc->saveToFile(solsName);
          }

          if(!checkSolution(S)){
            printf("Kids=");
            printVector(kids);
            printf("Sol=");
            printSolution(S);
            exit(0);
          }


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
  sol->saveToFile(nameOut);

  printf("Kids=");
  printVector(kids);
}
