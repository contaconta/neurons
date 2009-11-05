
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
// Contact < german.gonzalez@epfl.ch > for comments & bug reports      //
/////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "Cube.h"
#include "CubeLiveWire.h"
#include "Graph.h"
#include "GraphFactory.h"
#include "CloudFactory.h"
#include <gsl/gsl_rng.h>
#include <map>
#include "EdgeW.h"
#include "Neuron.h"
#ifdef WITH_OPENMP
#include <omp.h>
#endif



using namespace std;


class DMST : public DistanceDijkstra {
public:
  Cube<float, double>* cubeFloat;
  double ratioY, ratioZ;

  DMST(Cube_P* cube){
    this->cubeFloat = dynamic_cast<Cube< float, double>* >(cube);
    ratioZ = cube->voxelDepth/cube->voxelWidth;
    ratioY = cube->voxelHeight/cube->voxelWidth;
  }
  float distance(int x0, int y0, int z0, int x1, int y1, int z1){
    double dist;
    //    dist = sqrt((double) (x0-x1)*(x0-x1) + ratioY*(y0-y1)*(y0-y1) + ratioZ*(z0-z1)*(z0-z1));
    //    return 1-cubeFloat->at(x1,y1,z1)/dist;
    //    return -log(cubeFloat->at(x1,y1,z1)/dist);

    dist = sqrt((double) (x0-x1)*(x0-x1) +
                ratioY*ratioY*(y0-y1)*(y0-y1) +
                ratioZ*ratioZ*(z0-z1)*(z0-z1));
    float p1 = cubeFloat->at(x0,y0,z0);
    float p2 = cubeFloat->at(x1,y1,z1);
    //return fabs(dist*((log(p1) * p1 - p1- log(p2) * p2 + p2) / (-p2 + p1)));
    return -dist*log((p1+p2)/2);
  }
};



int main(int argc, char **argv) {

  if (argc!= 2){
    printf("Usage: MSTthroughShortestPath directory\n");
    exit(0);
  }

  string directory(argv[1]);

  const gsl_rng_type * T2;
  gsl_rng * r;
  gsl_rng_env_setup();
  T2 = gsl_rng_default;
  r = gsl_rng_alloc (T2);

  Cloud_P* decimatedCloud = CloudFactory::load
    (directory + "/decimated.cl");
  Cloud<Point3D>* seedPointsSelected = new Cloud<Point3D>
    ();

  //  Neuron* neuronita = new Neuron("/home/ggonzale/mount/cvlabfiler/n7_4/n7_4_fix.asc");
  //vector< double > lengths=  neuronita->getAllEdgesLength();
  //double m_length = 0;
  //double v_length = 0;
  //secondStatistics(lengths,  &m_length, &v_length);
  //printf("The mean is %f and the variance is %f\n", m_length, v_length);

  Cloud<Point3D>* soma = new Cloud<Point3D>();
  soma->points.push_back(decimatedCloud->points[0]);
  soma->v_r = 1.0;
  soma->v_g = 1.0;
  soma->v_b = 0.0;
  soma->v_radius = 1.5;
  soma->saveToFile
    (directory + "/tree/soma.cl");


  // For the multi-threaded implementation
  vector< Graph<Point3D, EdgeW<Point3D> >* > cptGraphs;
  Graph<Point3D, EdgeW<Point3D> >* cptGraph;

  // Computation of the complete graph
  if(1){
    //Definition of the window where to search for neighbors
    int windowX = 200;
    int windowY = 200;
    int windowZ = 40;
    // int iROIx,iROIy,iROIz;


    int nthreads = 1;
    vector< CubeLiveWire* > cubeLiveWires;
    vector< Cube<float, double>*> cubes;
    cptGraphs.resize(nthreads);

#ifdef WITH_OPENMP
    nthreads = omp_get_max_threads();
    omp_set_num_threads(nthreads);
    cptGraphs.resize(nthreads);
#endif

    for(int j = 0; j < nthreads; j++){
      cptGraphs[j] =
        new Graph<Point3D, EdgeW< Point3D> >();
      for(int i = 0; i < decimatedCloud->points.size(); i++)
        cptGraphs[j]->cloud->points.push_back
          ( new Point3D(decimatedCloud->points[i]->coords[0],
                        decimatedCloud->points[i]->coords[1],
                        decimatedCloud->points[i]->coords[2])
            );
    }


    printf("Performing detection with N = %i threads\n", nthreads);
    cubeLiveWires.resize(nthreads);
    cubes.resize(nthreads);
    for(int i = 0; i < nthreads; i++){
      cubes[i] = new Cube<float, double>
        (directory + "/2/merged_248p.nfo");
      //      DistanceDijkstraColorNegatedEuclideanAnisotropic* djkc
      // = new DistanceDijkstraColorNegatedEuclideanAnisotropic(cubes[i]);
            //DistanceDijkstraLogProb* djkc
            //= new DistanceDijkstraLogProb(cubes[i]);
      DMST* djkc
        = new DMST(cubes[i]);

      cubeLiveWires[i] = new CubeLiveWire(cubes[i], djkc);
    }

    int endPoint = decimatedCloud->points.size();
#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
    for(int i = 0; i < endPoint; i ++){
      // for(int i = 50; i < 51; i ++){
      int nth = 0;
#ifdef WITH_OPENMP
      nth = omp_get_thread_num();
#endif
      vector<int>   idxs(3);
      vector<int>   idxs2(3);
      vector<float> microm(3);
      vector<float> microm2(3);
      char graphName[1024];
      printf("Computing distances for point %i with thread %i\n", i, nth);
      fflush(stdout);
      microm[0] = decimatedCloud->points[i]->coords[0];
      microm[1] = decimatedCloud->points[i]->coords[1];
      microm[2] = decimatedCloud->points[i]->coords[2];
      cubes[nth]->micrometersToIndexes(microm, idxs);
      cubeLiveWires[nth]->iROIx = max(0, idxs[0]-windowX/2);
      cubeLiveWires[nth]->iROIy = max(0, idxs[1]-windowY/2);
      cubeLiveWires[nth]->iROIz = max(0, idxs[2]-windowZ/2);
      cubeLiveWires[nth]->eROIx = min((int)cubes[nth]->cubeWidth -1, idxs[0]+windowX/2);
      cubeLiveWires[nth]->eROIy = min((int)cubes[nth]->cubeHeight-1, idxs[1]+windowY/2);
      cubeLiveWires[nth]->eROIz = min((int)cubes[nth]->cubeDepth -1, idxs[2]+windowZ/2);
      cubeLiveWires[nth]->computeDistances(idxs[0], idxs[1], idxs[2]);

      //Check if the point is in the whereabouts of the cloud
      for(int j = 0; j < decimatedCloud->points.size(); j++){
        if(j==i) continue;
        microm2[0] = decimatedCloud->points[j]->coords[0];
        microm2[1] = decimatedCloud->points[j]->coords[1];
        microm2[2] = decimatedCloud->points[j]->coords[2];
        cubes[nth]->micrometersToIndexes(microm2, idxs2);
        if(  (idxs2[0] >= cubeLiveWires[nth]->iROIx) &&
             (idxs2[0] <= cubeLiveWires[nth]->eROIx) &&
             (idxs2[1] >= cubeLiveWires[nth]->iROIy) &&
             (idxs2[1] <= cubeLiveWires[nth]->eROIy) &&
             (idxs2[2] >= cubeLiveWires[nth]->iROIz) &&
             (idxs2[2] <= cubeLiveWires[nth]->eROIz)
             )
          {
            Graph<Point3D, EdgeW<Point3D> >* shortestPath =
              cubeLiveWires[nth]->findShortestPathG(idxs[0] ,idxs[1] ,idxs[2],
                                                    idxs2[0],idxs2[1],idxs2[2]);
            sprintf(graphName,
                    "%s/tree/paths/path_%04i_%04i.gr", directory.c_str(), i, j);
            double cost = 0;
            for(int nedge=0; nedge < shortestPath->eset.edges.size(); nedge++){
              EdgeW<Point3D>* edge = dynamic_cast<EdgeW<Point3D>*>
                (shortestPath->eset.edges[nedge]);
              cost += edge->w;
            }

            shortestPath->cloud->v_r = cost;
            shortestPath->cloud->v_g = 1-cost;
            shortestPath->cloud->v_b = 0;
            shortestPath->cloud->v_b = gsl_rng_uniform(r);
            shortestPath->cloud->v_radius = 0.4;
            shortestPath->saveToFile(graphName);
            double length = sqrt((microm2[0]-microm[0])*(microm2[0]-microm[0]) +
                                 (microm2[1]-microm[1])*(microm2[1]-microm[1]) +
                                 (microm2[2]-microm[2])*(microm2[2]-microm[2]) );
            cptGraphs[nth]->eset.edges.push_back
              (new EdgeW< Point3D>
               (&cptGraphs[nth]->cloud->points, i,j,
                cost ) );
          }
      }
    } //End of the loop through all the points in the cloud

    // Creates the complete Graph
    printf("Building the complete graph\n"); fflush(stdout);
    cptGraph = new  Graph<Point3D, EdgeW<Point3D> >();
    for(int i = 0; i < decimatedCloud->points.size(); i++)
      cptGraph->cloud->points.push_back
        (new Point3D(decimatedCloud->points[i]->coords[0],
                     decimatedCloud->points[i]->coords[1],
                     decimatedCloud->points[i]->coords[2])
         );
    for(int j = 0; j < nthreads; j++){
      for(int i = 0; i < cptGraphs[j]->eset.edges.size(); i++){
        EdgeW<Point3D>* edgeToAdd =
          dynamic_cast<EdgeW<Point3D>*>(cptGraphs[j]->eset.edges[i]);
        cptGraph->eset.edges.push_back
          (new EdgeW< Point3D > ( &cptGraph->cloud->points,
                                  edgeToAdd->p0,
                                  edgeToAdd->p1,
                                  edgeToAdd->w )
           );
      }
    }


    cptGraph->saveToFile(directory + "/tree/complete.gr");
  } else {
    Graph_P* cptGraphP =
      GraphFactory::load(directory + "/tree/complete.gr");
    cptGraph =
      dynamic_cast< Graph<Point3D, EdgeW<Point3D> >* >(cptGraphP);
  }

  printf("Computing the MST on the complete graph\n");
  fflush(stdout);
  Graph<Point3D, EdgeW<Point3D> >* mst =
    cptGraph->primFromThisGraphFast();
  printf("Saving the MST as a graph\n");
  mst->saveToFile(directory + "/tree/mstFromCptGraph.gr");


  //Saves the MST as a list of paths
  if(1){
    printf("Saving the MST as a list\n");
    std::ofstream out((directory + "/tree/mstFromCptGraph.lst").c_str());
    char buff[1024];
    for(int i =0; i < mst->eset.edges.size(); i++){
      if( (mst->eset.edges[i]->p0 == mst->eset.edges[i]->p1 ) ||
          (mst->eset.edges[i]->p0 == -1) ||
          (mst->eset.edges[i]->p1 == -1) )
        continue;
      sprintf(buff, "%s/tree/paths/path_%04i_%04i.gr",
              directory.c_str(),
              mst->eset.edges[i]->p0,
              mst->eset.edges[i]->p1);
      out << buff << std::endl;
    }
    out.close();
  }
}//main
