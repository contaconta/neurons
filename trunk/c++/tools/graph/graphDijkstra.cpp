
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



void addToSolution
(vector< int >& S,
 vector<int>& path,
 vector< int >& kids,
 Graph3D* cpt,
 vector< vector< Graph3D* > >& v2v_paths,
 Cube<uchar, ulong>* notvisited
){
  if(solutionContains(S, path[0])){
    for(int i = 1; i < path.size(); i++){
      kids[path[i-1]]++;
      if(S[path[i]] >= 0){
        printf("We are adding a node that already has been visited, exit\n");
        exit(0);
      }
      S[path[i]] = path[i-1];
    }
  }
  else if(solutionContains(S, path[path.size()-1])){
    for(int i = path.size()-2; i >= 0; i--){
      kids[path[i+1]]++;
      if(S[path[i]] >= 0){
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

  //mark all the points visited in notvisited as 0
  vector< int > p0;
  vector< int > p1;
  for(int i = 0; i < path.size()-1; i++){
    Graph3D* grpath = v2v_paths[path[i]][path[i+1]];
    for(int np = 0; np < grpath->cloud->points.size(); np++){
      notvisited->micrometersToIndexes(grpath->cloud->points[np  ]->coords, p0);
      notvisited->put_value_in_ellipsoid(0, p0[0], p0[1], p0[2], 5.0, 5.0, 5.0);
      printf("Drawing\n");
    }

      // notvisited->put_m(grpath->cloud->points[np]->coords[0],
                        // grpath->cloud->points[np]->coords[1],
                        // grpath->cloud->points[np]->coords[2], 0);
    // for(int np = 0; np < grpath->cloud->points.size()-1; np++){
      // notvisited->micrometersToIndexes(grpath->cloud->points[np  ]->coords, p0);
      // notvisited->micrometersToIndexes(grpath->cloud->points[np+1]->coords, p1);
      // notvisited->render_cylinder(p0, p1, 5, 0);
    // }
  }

  // And now elliminates from the solutio all those candidate points too close to the
  //  path. It can be heavy, since it is done only few times
  for(int nS = 0; nS < S.size(); nS++){
    if(S[nS]==-1){
      vector< float > p_c = cpt->cloud->points[nS]->coords;
      for(int npe = 0; npe < path.size()-1; npe++){
        int p0 = path[npe];
        int p1 = path[npe+1];
        Graph3D* path_v2v = v2v_paths[p0][p1];
        for(int npp = 0; npp < path_v2v->cloud->points.size(); npp++){
          vector< float > v_dist = v_subs(p_c, path_v2v->cloud->points[npp]->coords);
          if(v_norm(v_dist) < 5.0) //THRESHOLD
            S[nS] = -2;
        }
      }
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
 vector< vector< Graph3D* > >& v2v_paths,
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
      Graph3D* path = v2v_paths[i][j];
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
  vector< vector< Graph3D* > >& v2v_paths,
  Cube<float, double>* probs
  )
{
  float imageEvidence = 0;
  float tortuosityEvidence = 0;
  Graph3D* dsp2;

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
  return (imageEvidence + 0.0*tortuosityEvidence/10000 + 4.0);
}




void findCostOfPaths
( Graph<Point3D, EdgeW<Point3D> >* gr,
  vector< vector< vector< int   > > >& paths,
  vector< vector< float > >& costs,
  vector< vector< Graph3D* > >& v2v_paths,
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


int computePathLength
( vector< int >& path,
  vector< vector< Graph3D* > >& v2v_paths,
  Cube<uchar, ulong>* notVisited
  )
{
  int length = 0;
  Graph3D* dsp2;
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
  vector< vector< Graph3D* > >& v2v_paths,
  Cube<uchar, ulong>* notvisited
)
{
  int nPoints = gr->cloud->points.size();
  for(int i = 0; i < nPoints; i++){
    for(int j = 0; j < nPoints; j++){
      lengths[i][j] = computePathLength(paths[i][j], v2v_paths, notvisited);
      lengths[j][i] = lengths[i][j];
    }
  }
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

  // SWC* sols_swc = solutionToSWC(gr, cp, S);
  // sols_swc->saveToFile(solsName);

  //Save it as a list of paths
  sprintf(solsName, "%s/sol_%03i.lst",solsDirectory.c_str(), nComponentsAdded);
  std::ofstream solsPaths(solsName);
  char pathName[1024];
  for(int nE = 0; nE < sols->eset.edges.size(); nE++){
    sprintf(pathName, "%s/path_%04i_%04i.gr", pathsDirectory.c_str(),
            sols->eset.edges[nE]->p0, sols->eset.edges[nE]->p1);
    if(fileExists(pathName)){
      solsPaths << pathName << std::endl;
    } else {
      sprintf(pathName, "%s/path_%04i_%04i.gr", pathsDirectory.c_str(),
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
  Cube<uchar, ulong>*  notvisited = cp->create_blank_cube_uchar("visited",1);
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
  vector< vector< Graph3D* > > v2v_paths;
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
      notvisited
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
                                              notvisited);

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
    printf("sizeofQ = %i\n", Q.size());
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
