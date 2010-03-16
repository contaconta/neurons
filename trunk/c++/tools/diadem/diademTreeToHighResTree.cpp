
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
#include "CubeFactory.h"

using namespace std;

Graph3Dw* findPathBetween
(int p0, int p1, string pathsDirectory)
{
  char pathName[1024];
  Graph3Dw* solsPaths;
  sprintf(pathName, "%s/path_%04i_%04i-w.gr", pathsDirectory.c_str(),
          p0, p1);
  if(fileExists(pathName)){
    solsPaths = new Graph3Dw(pathName);
  } else {
    sprintf(pathName, "%s/path_%04i_%04i-w.gr", pathsDirectory.c_str(),
            p1, p0);
    if(fileExists(pathName)){
      solsPaths = new Graph3Dw(pathName);
    }
    else {printf("Path does not exist: %s\n", pathName); exit(0);}
  }
  return solsPaths;
}


class Node
{
public:
  Graph3Dw* path;
  vector< Node* > childs;
  int p0;
  int p1;

  Node()
  {
    path = NULL;
    childs.resize(0);
  }
  Node(int p0, int p1, string pathsDirectory)
  {
    childs.resize(0);
    this->p0 = p0;
    this->p1 = p1;
    this->path = findPathBetween(p0,p1, pathsDirectory);
  }

};

void printTree(Node* nd){
  printf("nd: %i %i\n", nd->p0, nd->p1);
  for(int i = 0; i < nd->childs.size(); i++)
    printTree(nd->childs[i]);

}

void expandNode
(Node* toExpand,
 vector< vector< int > >& neighbors,
 vector< int >& visited,
 string pathdir)
{
  printf("Node %i, %i expanded\n", toExpand->p0, toExpand->p1);
  for(int i = 0; i < neighbors[toExpand->p1].size(); i++){
    if(!visited[ neighbors[toExpand->p1][i] ]){
      visited[ neighbors[toExpand->p1][i] ] = 1;
      printf("   expanding to: %i, %i\n", toExpand->p1, neighbors[toExpand->p1][i]);
      Node* nextOne = new Node(toExpand->p1, neighbors[toExpand->p1][i], pathdir);
      toExpand->childs.push_back(nextOne);
      expandNode(nextOne, neighbors, visited, pathdir);
    }
  }
}


Node* convertGraphToTree
(Graph3D* graph, string pathdir)
{
  Node* root = new Node();
  root->p0 = 0; root->p1 = 0;
  vector< vector< int > > neighbors = graph->findNeighbors();
  vector< int >           visited(graph->cloud->points.size());
  visited[0] = 1;
  expandNode(root, neighbors, visited, pathdir);
  return root;
}


convertTreeToHighResGraph
(Node* root,
 Graph3Dw* result,
 Cube<int, ulong>* visited,
 Graph3D* lowres
)
{

  printf("Adding path between %i => %i\n", root->p0, root->p1);

  if(root->p0 == root->p1){ //root node, do nothing else than visiting the points
    visited->put_m
      (lowres->cloud->points[root->p0]->coords[0],
       lowres->cloud->points[root->p0]->coords[1],
       lowres->cloud->points[root->p0]->coords[2],
       0);
    result->cloud->points.push_back
      (new Point3Dw(lowres->cloud->points[root->p0]->coords[0],
                    lowres->cloud->points[root->p0]->coords[1],
                    lowres->cloud->points[root->p0]->coords[2],
                    1.0));
  } else { //not root node, put the path
    //the starting node and end node of the path can be the origin of the path. Let's
    // take the one visited as the starting point
    int indexStart    = 1;
    int indexEnd      = root->path->cloud->points.size() -1;
    int direction     = 1;
    if(visited->at_m
       (root->path->cloud->points[indexEnd]->coords[0],
        root->path->cloud->points[indexEnd]->coords[1],
        root->path->cloud->points[indexEnd]->coords[2])
       >= 0){
      direction = -1;
      indexStart = root->path->cloud->points.size() -2;
      indexEnd   = 0;
    }
    //Now add the edges
    for( int nIdx = indexStart;
         direction*nIdx <= direction*indexEnd;
         nIdx+=direction)
      {
        int visitedPt = visited->at_m
           (root->path->cloud->points[nIdx]->coords[0],
            root->path->cloud->points[nIdx]->coords[1],
            root->path->cloud->points[nIdx]->coords[2]);
        int visitedPrevPt = visited->at_m
           (root->path->cloud->points[nIdx-direction]->coords[0],
            root->path->cloud->points[nIdx-direction]->coords[1],
            root->path->cloud->points[nIdx-direction]->coords[2]);

        if( (visitedPt == -1) && (visitedPrevPt != -1) )
          {
            int nPointInCompleteGraph = result->cloud->points.size();
            Point3Dw* pt = dynamic_cast<Point3Dw*>(root->path->cloud->points[nIdx]);
            result->cloud->points.push_back
              (new Point3Dw(pt->coords[0],
                            pt->coords[1],
                            pt->coords[2],
                            pt->weight));
            result->eset.edges.push_back
              (new EdgeW<Point3Dw>(&result->cloud->points, visitedPrevPt,
                                   nPointInCompleteGraph, 1.0));
            visited->put_m
              (pt->coords[0],
               pt->coords[1],
               pt->coords[2],
               nPointInCompleteGraph);
          }

      }//through the indexes of the path
  }//not root node

  for(int i = 0; i < root->childs.size(); i++){
    convertTreeToHighResGraph
      (root->childs[i],
       result,
       visited,
       lowres
       );
  } // loop for the childs


}


int main(int argc, char **argv) {

  if(argc!=5){
    printf("Usage: diademTreeToHighResTree tree.gr paths_dir/ volume_reference.nfo out.gr\n");
    exit(0);
  }
  Graph3D*  lowRes  = new Graph3D(argv[1]);
  string    pathDir(argv[2]);
  Cube<uchar, ulong>*   cube    = new Cube<uchar, ulong>(argv[3]);
  Cube<int, ulong>*  visited    = new Cube<int, ulong>
    (cube->cubeWidth, cube->cubeHeight, cube->cubeDepth,
     1, 1, 1);
  visited->put_all(-1);
  Graph3Dw* highRes = new Graph3Dw();

  //The first thing is to return a tree structure
  Node* root = convertGraphToTree(lowRes, pathDir);
  printf("Conversion done\n");
  printTree(root);

  Graph3Dw* hres = new Graph3Dw();
  convertTreeToHighResGraph(root, hres, visited, lowRes);

  hres->saveToFile(argv[4]);



}
