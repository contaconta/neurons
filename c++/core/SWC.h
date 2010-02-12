#ifndef SWC_H_
#define SWC_H_

#include "VisibleE.h"
#include "utils.h"
#include "Cloud.h"
#include "Point3Dw.h"
#include "Graph.h"

/********************************************************************
 An SWC file represents a tree as a table. The columns of the table
 represent:
 #id #st #x #y #z #r #parentid

 *******************************************************************/


class SWC : public VisibleE
{
public:

  Graph<Point3Dw, Edge<Point3Dw> >* gr;

  int idxSoma;

  //To be done the distinction among points - later
  // vector< Cloud<Point3Dw>* > cloudTypes;
  // Cloud<Point3Dw>* undefined;
  Cloud<Point3Dw>* soma;
  // Cloud<Point3Dw>* axon;
  // Cloud<Point3Dw>* dendrite;
  // Cloud<Point3Dw>* apicalDendrite;
  // Cloud<Point3Dw>* forkPoint;
  // Cloud<Point3Dw>* endPoint;
  // Cloud<Point3Dw>* custom;
  vector<double>   offset;

  SWC(){}

  SWC(string filename){

    Cloud<Point3Dw>* allPoints;

    string filenameNoExt = getPathWithoutExtension(filename);
    string offsetTXT     = getPathWithoutExtension(filename) + ".txt";
    if(fileExists(offsetTXT)){
      offset = readVectorDouble(offsetTXT);
      printf("Offset read from %s: [%f, %f, %f]\n",
             offsetTXT.c_str(), offset[0], offset[1], offset[2]);
    }
    else {
      offset.resize(0); offset.push_back(0); offset.push_back(0); offset.push_back(0);
    }


    allPoints = new Cloud<Point3Dw>();
    soma = new Cloud<Point3Dw>();
    vector< vector< double > > orig = loadMatrix(filename);
    allPoints->points.resize(orig.size());

    for(int i = 0; i < orig.size(); i++){
      double width = orig[i][5];
      int idxPoint = orig[i][0]-1;
      if (width == 0) width = 1.0;
      allPoints->points[idxPoint] =
        new Point3Dw(orig[i][2] + offset[0],
                      orig[i][3] + offset[1],
                      orig[i][4] + offset[2],
                      width);
      if(orig[i][6] == -1){
        soma->points.push_back
          (new Point3Dw(orig[i][2] + offset[0],
                        orig[i][3] + offset[1],
                        orig[i][4] + offset[2],
                        width));
        idxSoma = i;
      }
    }
    gr = new Graph<Point3Dw, Edge<Point3Dw> >(allPoints);
    for(int i = 0; i < orig.size(); i++){
      int endIdx =  orig[i][6]-1;
      int initIdx = orig[i][0]-1;
      if( (endIdx >= 0) & (endIdx < allPoints->points.size()))
        gr->eset.addEdge(initIdx, endIdx);
    }
    soma->v_r=1.0; soma->v_g= 1.0;
    // allPoints->v_radius = 0.1;
  }

  void draw(){
    // gr->cloud->draw();
    gr->draw();
    soma->draw();
    glColor3f(0.0,1.0,0.0);
    gr->cloud->points[idxSoma]->draw(2.0);
  }

  void save(ostream &out)
  {
    // The data is stored into a graph, where the edges are not directional, thus we need to go throught the points creating the order

    vector< vector< int > > connections(gr->cloud->points.size());
    vector< int > visited(gr->cloud->points.size());
    // vector< int > parents(gr->cloud->points.size());
    vector< int > newPointIdx(gr->cloud->points.size());
    for(int i = 0; i < visited.size(); i++)
      visited[i] = 0;

    //fills the connections according to the edges
    for(int i = 0; i < gr->eset.edges.size(); i++){
      //printf("e=%i, connectionsSize = %i\n", i, connections.size());
      int p0 = gr->eset.edges[i]->p0;
      int p1 = gr->eset.edges[i]->p1;
      //printf("e=%i, p0=%i, p1=%i\n", i, p0, p1);
      connections[p0].push_back(p1);
      connections[p1].push_back(p0);
    }
    vector< int > toProcess;
    // outputs the soma, since it is the special case (no parent
    int pointToSaveIdx = 1;
    Point3Dw* pt = dynamic_cast<Point3Dw*>(gr->cloud->points[idxSoma]);
    out << pointToSaveIdx << " " <<
      2 << " " <<
      pt->coords[0] << " " <<
      pt->coords[1] << " " <<
      pt->coords[2] << " " <<
      1             << " " <<
     -1 << "\n";
    visited[idxSoma] = 1;
    // parents[idxSoma] = -2;
    newPointIdx[idxSoma] = pointToSaveIdx;
    pointToSaveIdx++;
    toProcess.push_back(idxSoma);

    while(toProcess.size()!=0){
      //Finds the next point to process
      int idx = toProcess.back();
      toProcess.pop_back();
      //Forces a maximum of two kids per point
      int limitPoints = 0;
      for(int i = 0; i < connections[idx].size(); i++){
        //it is a new point, add it
        if(visited[connections[idx][i]]==0){
          if(limitPoints >= 2)
            continue;
          else limitPoints ++;
          Point3Dw* pt = dynamic_cast<Point3Dw*>(gr->cloud->points[connections[idx][i]]);
          out << pointToSaveIdx << " " << 2 << " " <<
            pt->coords[0] << " " <<
            pt->coords[1] << " " <<
            pt->coords[2] << " " <<
            pt->weight    << " " <<
            newPointIdx[idx]  << "\n";

          visited[connections[idx][i]]=1;
          toProcess.push_back(connections[idx][i]);
          newPointIdx[connections[idx][i]]=pointToSaveIdx;
          pointToSaveIdx++;
        }//if
      }//for
    }//while

  }//save


  // Answers 1 if the tree is binary, 0 otherwise
  int isBinaryTree(){
    vector< vector< int > > connections(gr->cloud->points.size());
    vector< int > visited(gr->cloud->points.size());
    vector< int > parents(gr->cloud->points.size());
    vector< int > toProcess;
    for(int i = 0; i < visited.size(); i++)
      visited[i] = 0;

    //fills the connections according to the edges
    for(int i = 0; i < gr->eset.edges.size(); i++){
      int p0 = gr->eset.edges[i]->p0;
      int p1 = gr->eset.edges[i]->p1;
      connections[p0].push_back(p1);
      connections[p1].push_back(p0);
    }

    parents[idxSoma] = -1;
    visited[idxSoma] = 1 ;
    toProcess.push_back(idxSoma);

    while(toProcess.size()!=0){
      //Finds the next point to process
      int idx = toProcess.back();
      toProcess.pop_back();

      for(int i = 0; i < connections[idx].size(); i++){
        //it is a new point, add it
        if(visited[connections[idx][i]]==0){
          visited[connections[idx][i]]=1;
          toProcess.push_back(connections[idx][i]);
          parents[connections[idx][i]]=idx;
        }//if
      }//for
    }//while

    //Now let's see the number of childs per node
    vector< int > childsPerNode(gr->cloud->points.size());
    for(int i = 0; i < gr->cloud->points.size(); i++)
      if(parents[i] > 0){
        childsPerNode[parents[i]]++;
        if(childsPerNode[parents[i]] > 2)
          return -1;
      }
    return 1;

  }

  Graph<Point3Dw>* toGraphInMicrometers(Cube_P* cube)
  {
    Graph<Point3Dw>* out = new Graph<Point3Dw>();
    float mx, my, mz;
    for(int i = 0; i < gr->cloud->points.size(); i++){
      Point3Dw* pt = dynamic_cast<Point3Dw*>(gr->cloud->points[i]);
      cube->indexesToMicrometers3((int)pt->coords[0], (int)pt->coords[1],
                                  (int)pt->coords[2], mx, my, mz);
      out->cloud->points.push_back
        (new Point3Dw(mx, my, mz, pt->weight));
    }
    out->eset = gr->eset;
    return out;
  }



  bool load(istream &in){};

  string className(){return "SWC";}

};
#endif
