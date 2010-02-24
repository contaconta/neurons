
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

typedef Graph<Point3D,  EdgeW<Point3D>  > Graph3D;
typedef Graph<Point3Dw, EdgeW<Point3Dw> > Graph3Dw;

//Here are most of the shortest path computation
#include "graphDijkstra_aux.cpp"

bool solutionContains(vector<int>& solution, int n){
  // if(solution[n] != -1)
  if(solution[n] >= 0)
      return true;
  return false;
}


//The return value is:  0 (none of the starting / end points are in the graph
//                      1 the path is compatible
//                     -1 the path has one of the middle elements in
int isCompatible(vector<int>& S, vector< int>& path, vector< int >& kids)
{

  // If the path starts or ends in  any of the elliminated points, then it is not compatible
  for(int i = 0; i < path.size(); i++)
    if(S[path[i]] == -2)
      return -1;
  // if(S[path[0]] == -2)
    // return -1;
  // if(S[path[path.size()-1]] == -2)
    // return -1;


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




  // Adding a path is not a simple task. First, the connecting point between the
  // path and the solution can be at the beginning or end of the path, then the
  // subpaths that compose the solution can be ordered left-to-right or
  // otherwise. All points that are close to the path should be included into
  // the solution splitting the path into two sub-paths. We will call a path the
  // connection we want to make and a sub-path the connection between two anchor
  // points


void addToSolution
(vector< int >& S,
 vector<int>& path,
 vector< int >& kids,
 Graph3D* cpt,
 vector< vector< Graph3Dw* > >& v2v_paths,
 Cube<uchar, ulong>* notvisited
){
    int pathDirection  = 1;  //point 0 is part of the path
    int pathPointBegin = 1;
    int pathPointLimit = path.size()-1;
    if(!solutionContains(S, path[0])){ //in case we reverse the direction
      pathDirection  = -1;
      pathPointBegin = path.size()-2;
      pathPointLimit = 0;
    }
    for(int nPointPath = pathPointBegin;
        pathDirection*nPointPath <= pathDirection*pathPointLimit;
        nPointPath += pathDirection)
      {
        int currentParent = path[nPointPath - pathDirection];
        int pathChild     = path[nPointPath];
        printf("nPointPath = %i: -> [%i, %i]\n", nPointPath,
               pathChild, currentParent);


        // now it comes the loop for the sub-path points
        Graph3Dw* s_path = v2v_paths[pathChild] [currentParent];
        int s_pathDirection = 1;
        int s_pathPointBegin  = 0;
        int s_pathPointEnd    = s_path->cloud->points.size()-1;
        // check if we go back to front
        Point* seedPtOriginPath = cpt->cloud->points[currentParent];
        Point* s_path_init  = s_path->cloud->points[0];
        Point* s_path_end   = s_path->cloud->points[s_path->cloud->points.size() -1];
        vector< float > s_p_i_o = v_subs(s_path_init->coords, seedPtOriginPath->coords);
        vector< float > s_p_e_o = v_subs(s_path_end->coords , seedPtOriginPath->coords);
        //if the end path is closer to the origin point
        if(v_norm(s_p_i_o) > v_norm(s_p_e_o)){
          s_pathDirection   = -1;
          s_pathPointBegin  = s_path->cloud->points.size()-1;
          s_pathPointEnd    = 0;
        }

        //same loop as before
        for(int s_pointPath = s_pathPointBegin;
            s_pathDirection*s_pointPath <= s_pathDirection*s_pathPointEnd;
            s_pointPath += s_pathDirection)
          {
            Point3Dw* pt = dynamic_cast<Point3Dw*>(s_path->cloud->points[s_pointPath]);
            for(int nS = 0; nS < S.size(); nS++){
              if(S[nS]==-1){
                Point* pts = cpt->cloud->points[nS];
                vector< float > d_v = v_subs(pt->coords, pts->coords);
                float d = v_norm(d_v);
                if(d < pt->weight*1.2){  //PARAMETER
                  kids[currentParent]++;
                  S[nS] = currentParent;
                  currentParent = nS;
                }
              }
            }//nS
          }
      } // through the anchor points


  //mark all the points visited in notvisited as 0
  vector< int > p0;
  vector< int > p1;
  for(int i = 0; i < path.size()-1; i++){
    Graph3Dw* grpath = v2v_paths[path[i]][path[i+1]];
    for(int np = 0; np < grpath->cloud->points.size(); np++){
      Point3Dw* pt = dynamic_cast<Point3Dw*>(grpath->cloud->points[np]);
      notvisited->micrometersToIndexes(grpath->cloud->points[np  ]->coords, p0);
      notvisited->put_value_in_ellipsoid(0, p0[0], p0[1], p0[2], pt->weight,
                                         pt->weight, pt->weight);
      // printf("Drawing\n");
    }
  }

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
  // if(fabs(p1-p2) < 1e-4) return -dist*log(p1);
  // return fabs(dist*((log(p1) * p1 - p1- log(p2) * p2 + p2) / (-p2 + p1)));
  return -log((p1+p2)/2);
}


float edgeLogProb
(Point* p0, Cube<float, double>* probs)
{
  return -log(probs->at_m(p0->coords[0], p0->coords[1], p0->coords[2]));
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
    return -log(0.1+0.9*fabs(cos_alpha));

}



void computeTortuosities
(
 vector< vector< Graph3Dw* > >& v2v_paths,
 vector< vector< float > >& tortuosities,
 float sigma
 )
{
  tortuosities.resize(v2v_paths.size());
  for(int i = 0; i < tortuosities.size(); i++){
    tortuosities[i].resize(v2v_paths.size());
    for(int j = 0; j < tortuosities[i].size(); j++)
      tortuosities[i][j] = 0;
  }

  for(int i = 0; i < tortuosities.size(); i++)
    for(int j = i+1; j < tortuosities.size(); j++){
      float tortuosity = 0;
      Graph3Dw* path = v2v_paths[i][j];
      if(path == NULL) continue;
      vector< float > p0 = path->cloud->points[0]->coords;
      vector< float > p1 = path->cloud->points[path->cloud->points.size()-1]->coords;
      vector< float > p1p0 = v_subs(p1, p0);
      vector< float > p1p0n = v_scale(p1p0, 1.0/v_norm(p1p0));
      for(int nP = 1; nP < path->cloud->points.size()-1; nP++){
        vector< float > p2 = path->cloud->points[nP]->coords;
        vector< float > p2p0 = v_subs(p2, p0);
        float dot = v_dot(p2p0, p1p0n);
        vector< float > dir = v_scale(p1p0, dot);
        vector< float > perp = v_subs(p2p0, dir);
        float dist = v_norm(perp);
        tortuosity += dist*dist/sigma;
      }
      tortuosities[i][j] = tortuosity;
      tortuosities[j][i] = tortuosity;
      // printf("Tortuosity: %i,%i -> %f\n", i, j, tortuosity/ path->cloud->points.size());
    }
}



float computePathCost
( Graph<Point3D, EdgeW<Point3D> >* gr,
  vector< int >& path,
  vector< vector< Graph3Dw* > >& v2v_paths,
  Cube<float, double>* probs
  )
{
  float imageEvidence = 0;
  float tortuosityEvidence = 0;
  Graph3Dw* dsp2;

  Point* p0 = v2v_paths[path[0]][path[1]]->cloud->points[0];
  dsp2 = v2v_paths[path[path.size()-2]][path[path.size()-1]];
  Point* p1 = dsp2->cloud->points[dsp2->cloud->points.size()-1];
  vector< float > p1p0 = v_subs(p1->coords, p0->coords);
  vector< float > p1p0n = v_scale(p1p0, 1.0/v_norm(p1p0));

  //Loops for all the edges of the path
  for(int nep = 0; nep < path.size()-1; nep++){
    dsp2 = v2v_paths[path[nep]][path[nep+1]];
    //Yes, I am ignoring the last point
    for(int nP = 0; nP < dsp2->cloud->points.size()-1; nP++){
      imageEvidence +=
        edgeLogProb
        (dsp2->cloud->points[nP], probs);
      vector< float > p2p0 = v_subs(dsp2->cloud->points[nP]->coords, p0->coords);
      float dot = v_dot(p2p0, p1p0n);
      vector< float > toSub = v_scale(p1p0n, dot);
      vector< float > perp = v_subs(p2p0, toSub );
      float dist = v_norm(perp);
      tortuosityEvidence += dist*dist;
    }
  }
  //Very last point
  imageEvidence += edgeLogProb(dsp2->cloud->points[dsp2->cloud->points.size()-1], probs);

  // printf("path: image %f tortuosity %f\n", imageEvidence, tortuosityEvidence);
  return (imageEvidence + 1.0*tortuosityEvidence/10000 + 4.0);
}




void findCostOfPaths
( Graph<Point3D, EdgeW<Point3D> >* gr,
  vector< vector< vector< int   > > >& paths,
  vector< vector< float > >& costs,
  vector< vector< Graph3Dw* > >& v2v_paths,
  Cube<float, double>* probs
)
{
  int nPoints = gr->cloud->points.size();
  for(int i = 0; i < nPoints; i++){
    for(int j = i+1; j < nPoints; j++){
      if(paths[i][j].size() >= 2){
        costs[i][j] = computePathCost(gr, paths[i][j], v2v_paths, probs);
        costs[j][i] = costs[i][j];
      }
    }
  }
}


void pathBoundingBox
(vector< int >& path,
 vector< vector< Graph3Dw* > >& v2v_paths,
 Cube<uchar, ulong>* notVisited,
 int& x0, int& y0, int& z0,
 int& x1, int& y1, int& z1
 )
{
  x0 = notVisited->cubeWidth;
  y0 = notVisited->cubeHeight;
  z0 = notVisited->cubeDepth;
  x1 = 0;
  y1 = 0;
  z1 = 0;

  int xt, yt, zt;
  Graph3Dw* gr;
  Point3Dw* pt;
  float w;
  for(int nPP = 1; nPP < path.size(); nPP++){
    gr = v2v_paths[path[nPP]][path[nPP-1]];
    for(int nP = 0; nP < gr->cloud->points.size(); nP++){
      pt = dynamic_cast<Point3Dw*>(gr->cloud->points[nP]);
      w = pt->weight;
      notVisited->micrometersToIndexes3
        (pt->coords[0], pt->coords[1], pt->coords[2],
         xt, yt, zt);
      if(max(0, (int)(xt-w)) <= x0)
        x0 = max(0, (int)(xt-w));
      if(max(0, (int)(yt-w)) <= y0)
        y0 = max(0, (int)(yt-w));
      if(max(0, (int)(zt-w)) <= z0)
        z0 = max(0, (int)(zt-w));
      if(min(notVisited->cubeWidth-1, (int)(xt+w)) >= x1)
        x1 = min(notVisited->cubeWidth-1, (int)(xt+w));
      if(min(notVisited->cubeHeight-1, (int)(yt+w)) >= y1)
        y1 = min(notVisited->cubeHeight-1, (int)(yt+w));
      if(min(notVisited->cubeDepth-1, (int)(zt+w)) >= z1)
        z1 = min(notVisited->cubeDepth-1, (int)(zt+w));
    }//np
  }//npp
}

int computePathLength
( vector< int >& path,
  vector< vector< Graph3Dw* > >& v2v_paths,
  Cube<uchar, ulong>* notVisited,
  Cube<uchar, ulong>* blackboard
  )
{
  if(0){
    //Erases the blackboard
    int x0, y0, z0, x1, y1, z1;
    pathBoundingBox(path, v2v_paths, notVisited, x0, y0, z0, x1, y1, z1);
    for(int z = z0; z<=z1; z++)
      for(int y = y0; y<=y1; y++)
        for(int x = x0; x<=x1; x++)
          blackboard->put(x,y,z,0);

    //Draws the path in the blackboard
    Graph3Dw* gr;
    Point3Dw* pt;
    int xt, yt, zt;
    for(int nPP = 1; nPP < path.size(); nPP++){
      gr = v2v_paths[path[nPP]][path[nPP-1]];
      for(int nP = 0; nP < gr->cloud->points.size(); nP++){
        pt = dynamic_cast<Point3Dw*>(gr->cloud->points[nP]);
        blackboard->micrometersToIndexes3
          (pt->coords[0], pt->coords[1], pt->coords[2],
           xt, yt, zt);
        blackboard->put_value_in_ellipsoid(1, xt, yt, zt, pt->weight,
                                           pt->weight, pt->weight);
      }//nPP
    }//np

    int length = 0;
    for(int z = z0; z<=z1; z++)
      for(int y = y0; y<=y1; y++)
        for(int x = x0; x<=x1; x++)
          length += notVisited->at(x,y,z)*blackboard->at(x,y,z);
  }



  int length = 0;
  Graph3Dw* dsp2;
  for(int nep = 0; nep < path.size()-1; nep++){
    dsp2 = v2v_paths[path[nep]][path[nep+1]];
    for(int nP = 0; nP < dsp2->cloud->points.size()-1; nP++){
      length+= notVisited->at_m(dsp2->cloud->points[nP]->coords[0],
                                dsp2->cloud->points[nP]->coords[1],
                                dsp2->cloud->points[nP]->coords[2]);
    }
  }
  return length/255;
}


void findLengthsOfPaths
( Graph<Point3D, EdgeW<Point3D> >* gr,
  vector< vector< vector< int   > > >& paths,
  vector< vector< int > >& lengths,
  vector< vector< Graph3Dw* > >& v2v_paths,
  Cube<uchar, ulong>* notvisited,
  Cube<uchar, ulong>* blackboard
)
{
  int nPoints = gr->cloud->points.size();
  printf("Computing the paht length: ");
  for(int i = 0; i < nPoints; i++){
    for(int j = i+1; j < nPoints; j++){
      lengths[i][j] = computePathLength(paths[i][j], v2v_paths, notvisited, blackboard);
      lengths[j][i] = lengths[i][j];
    }
    printf("%04i\r", i); fflush(stdout);
  }
  printf("\n");
}


void saveSolution
( vector< int >& S,
  Graph3D* gr,
  int nComponentsAdded,
  string solsDirectory,
  string pathsDirectory,
  CubeF* cp
  )
{
  char solsName[512];
  // if(nComponentsAdded > 100) continue;
  // printf("Saving solution %i\n", nComponentsAdded);
  Graph3D* sols = solutionToGraph(gr, S);
  sprintf(solsName, "%s/sol_%03i.gr",solsDirectory.c_str(), nComponentsAdded);
  sols->saveToFile(solsName);
  sprintf(solsName, "%s/sol_%03i.swc",solsDirectory.c_str(), nComponentsAdded);

  SWC* sols_swc = solutionToSWC(gr, cp, S);
  sols_swc->saveToFile(solsName);

  //Save it as a list of paths
  sprintf(solsName, "%s/sol_%03i.lst",solsDirectory.c_str(), nComponentsAdded);
  std::ofstream solsPaths(solsName);
  char pathName[1024];
  for(int nE = 0; nE < sols->eset.edges.size(); nE++){
    sprintf(pathName, "%s/path_%04i_%04i-w.gr", pathsDirectory.c_str(),
            sols->eset.edges[nE]->p0, sols->eset.edges[nE]->p1);
    if(fileExists(pathName)){
      solsPaths << pathName << std::endl;
    } else {
      sprintf(pathName, "%s/path_%04i_%04i-w.gr", pathsDirectory.c_str(),
              sols->eset.edges[nE]->p1, sols->eset.edges[nE]->p0);
      if(fileExists(pathName)){
        solsPaths << pathName << std::endl;
      }
      else {printf("Path does not exist: %s\n", pathName); exit(0);}
    }
  }

  //Saves a cloud with the points that are considered as 'bad'
  //  by the current solution
  Cloud<Point3D>* cl_neg = new Cloud<Point3D>();
  for(int i = 0; i < gr->cloud->points.size(); i++)
    if(S[i]==-2)
      cl_neg->points.push_back(gr->cloud->points[i]);
  cl_neg->v_g = 1.0;
  cl_neg->v_r = 1.0;
  cl_neg->v_b = 0.0;
  sprintf(pathName, "%s/cloud_rejected_%03i.cl", solsDirectory.c_str(),
          nComponentsAdded);
  cl_neg->saveToFile(pathName);
  Cloud<Point3D>* cl_vis = new Cloud<Point3D>();
  for(int i = 0; i < gr->cloud->points.size(); i++)
    if(S[i]>=0)
      cl_vis->points.push_back(gr->cloud->points[i]);
  cl_vis->v_g = 0.0;
  cl_vis->v_r = 0.0;
  cl_vis->v_b = 1.0;
  sprintf(pathName, "%s/cloud_visited_%03i.cl", solsDirectory.c_str(),
          nComponentsAdded);
  cl_vis->saveToFile(pathName);

}

void deleteItemsFromQ
(   multimap<float, int>& Q,
    vector< multimap<float, int>::iterator  >& markedForDeletion
 )
{
  for(int i = markedForDeletion.size()-1; i >= 0; i--)
    Q.erase(markedForDeletion[i]);
}



/*******************************************************************************
 * S is the tree that is the solution of the problem. Stored as a vector of ints
 *    where each dimension represents point i and the value of that dimension is
 *    -1 if the point is not connected to another one or >0 if the point is
 *       connected to another one
 *    -2 if the point is close already to the solution and should not be taken
 *       into account anymore
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
  Cube<uchar, ulong>*  notvisited = cp->create_blank_cube_uchar("visited",0);
  //used to compute temporal masks of paths
  Cube<uchar, ulong>*  blackboard = cp->create_blank_cube_uchar("visited",0);

  notvisited->put_all(255);

  // notvisited->put_value_in_ellipsoid(0, 100, 100, 20, 5.0, 5.0, 5.0);
  // exit(0);

  string solsDirectory(argv[7]);
  string pathsDirectory(argv[8]);
  saveAllSolutions = true;
  if(!directoryExists(solsDirectory))
    makeDirectory(solsDirectory);


  //Initialization of the solution and auxiliary structures
  int nPoints = gr->cloud->points.size();
  vector< int >   S(nPoints); //solution
  vector< int >   kids(nPoints); //stores the number of kids per point
  for(int i = 0; i < nPoints; i++){
    S[i] = -1; kids[i] = 0;
  }

  S[0]=0;

  //paths is going to be the main data structure. Stores the shortest connection among all points.
  vector< vector< vector< int   > > > paths;
  vector< vector< float > >    costs;
  vector< vector< int  > >     lengths;
  vector< vector< Graph3Dw* > > v2v_paths;
  costs.resize(nPoints);
  lengths.resize(nPoints);
  v2v_paths.resize(nPoints);
  for(int i = 0; i < nPoints; i++){
    costs[i].resize(nPoints);
    lengths[i].resize(nPoints);
    v2v_paths[i].resize(nPoints);
  }
  printf("Warmup done ...\n");

  printf("Loading paths ...\n");
  loadAllv2vPaths(pathsDirectory, gr, v2v_paths);


  printf("Computing shortest paths ...\n");
  allShortestPaths(gr, paths, cp, v2v_paths);
  findCostOfPaths
    ( gr,
      paths,
      costs,
      v2v_paths,
      cp
      );

  printf("Computing path lengths ...\n");
  findLengthsOfPaths
    ( gr,
      paths,
      lengths,
      v2v_paths,
      notvisited,
      blackboard
      );


  multimap<float, int> Q; //Priority queue

  //Initialization - all paths into Q with their costs
  printf("Starting Q\n");
  for(int i = 0; i < nPoints; i++)
    for(int j = i+1; j < nPoints; j++)
      Q.insert(pair<float, int>(costs[i][j]/lengths[i][j], i*nPoints+j));


  multimap<float, int>::iterator it = Q.begin();
  printf("Initialization done\n");
  printf("S: "); printSolution(S);

  //And now the algorithm
  int counter=0;
  bool weAreDone=false;
  int nComponentsAdded = 0;
  Graph3D* sols;
  SWC* sols_swc;

  while(!Q.empty()){
    it = Q.begin();
    bool thereIsSomethingCompatible = false;
    vector< multimap<float, int>::iterator  > markedForDeletion;
    int elementNumber = 0;

    //loops until some path compatible to the current solution is found
    //  the paths can be not compatible, not not compatible with the solution
    bool isThereSomethingCompatible = false;
    for(it = Q.begin(); it != Q.end(); ++it){
      int isCompatibleN = isCompatible(S,paths[floor(it->second/nPoints)]
                                       [it->second - nPoints*floor(it->second/nPoints)],
                                       kids);
      if( isCompatibleN == 1)
        {
          isThereSomethingCompatible = true;
          markedForDeletion.push_back(it);
          int i_i, i_j;
          i_i = floor(it->second/nPoints);
          i_j = it->second - nPoints*floor(it->second/nPoints);
          float value = it->first;
          int path_length = computePathLength(paths[i_i][i_j], v2v_paths,
                                              notvisited, blackboard);

          // The value of the path is not up to date
          if(value != costs[i_i][i_j] / path_length){
            deleteItemsFromQ(Q, markedForDeletion);
            Q.insert(pair<float, int>(costs[i_i][i_j]/path_length, i_i*nPoints+i_j));
            printf("The path %i, %i has been re-weighted: costs %f->%f/%i=%f,"
                   "lengths %i->%i\n",
                   i_i, i_j, value, costs[i_i][i_j],path_length,
                   costs[i_i][i_j]/path_length,
                   lengths[i_i][i_j], path_length);
          } else {
            // the value is up to date, add it to the tree
            printf("nComponentsAdded=%i, cost=%f\n", nComponentsAdded, value);
            addToSolution(S,paths[i_i][i_j],
                          kids, gr, v2v_paths, notvisited);
            deleteItemsFromQ(Q, markedForDeletion);
            if(saveAllSolutions)
              saveSolution
                ( S, gr, nComponentsAdded, solsDirectory, pathsDirectory, cp);

            if(!checkSolution(S)){
              printf("The solution found might have some errors\n");
              printf("Kids=");
              printVector(kids);
              printf("Sol=");
              printSolution(S);
              exit(0);
            }

            nComponentsAdded++;
          }//add value to solution
          break; //restart the search for compatible paths
        } // is compatible == 1
      if (isCompatibleN == -1){
        markedForDeletion.push_back(it);
      }
    } // for 
    printf("sizeofQ = %i\n", (int)Q.size());
    if(isThereSomethingCompatible == false)
      break;
  }//Iteration over Q

  // printVector(S);
  Graph<Point3D, EdgeW<Point3D> >* sol =
    solutionToGraph(gr, S);
  sol->saveToFile(nameOut);

  // printf("Kids=");
  // printVector(kids);
}
