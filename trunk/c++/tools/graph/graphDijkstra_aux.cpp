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
  vector< vector< int   > >& neighbors,
  CubeF* probs,
  vector< vector< Graph3Dw* > >& v2v_paths
)
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
    // Here we compute the edge cost as we want.
    Graph3Dw* gr2 = v2v_paths[gr->eset.edges[i]->p1][gr->eset.edges[i]->p0];
    float cost = 0;
    for(int k = 0; k < gr2->cloud->points.size(); k++)
      cost -= log(probs->at_m(gr2->cloud->points[k]->coords[0],
                              gr2->cloud->points[k]->coords[1],
                              gr2->cloud->points[k]->coords[2]));
    distanceMatrix[gr->eset.edges[i]->p0][gr->eset.edges[i]->p1] = cost;
    distanceMatrix[gr->eset.edges[i]->p1][gr->eset.edges[i]->p0] = cost;
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
  printf("Path for point %03i done\r", sourceNode);
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
    if( (solution[i] >= 0) && (solution[i]!=i))
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
    if( (solution[i] >= 0) && (solution[i]!=i))
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









// Translates a solution to a graph
Graph<Point3D, EdgeW<Point3D> >*
solutionToGraph
(Graph<Point3D, EdgeW<Point3D> >* cpt,
 vector< int > solution)
{
  Graph<Point3D, EdgeW<Point3D> >* toRet =
    new Graph<Point3D, EdgeW<Point3D> >(cpt->cloud);
  for(int i = 0; i < solution.size(); i++){
    if((solution[i]!=i) && (solution[i] >= 0)){
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
 vector< vector< Graph3Dw* > >& v2v_paths)
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
   // printf("Loading the path between vertex %i and %i in directory %s, quit\n",
          // p0, p1, pathsDirectory.c_str());

   Graph3Dw* graph;
   char buff[1024];
   sprintf(buff, "%s/path_%04i_%04i-w.gr", pathsDirectory.c_str(), p0, p1);
   if(fileExists(buff)){
     graph = new Graph3Dw(buff);}
   else{
     sprintf(buff, "%s/path_%04i_%04i-w.gr", pathsDirectory.c_str(), p1, p0);
     if(fileExists(buff)){
       graph = new Graph3Dw(buff);}
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

// Computes all the shortest paths between all pairs of points and assign them a cost
void allShortestPaths
( Graph<Point3D, EdgeW<Point3D> >* gr,
  vector< vector< vector< int   > > >& paths,
  CubeF* probs,
  vector< vector< Graph3Dw* > >& v2v_paths
)
{
  // Temporal variables
  vector< vector< float > > distanceMatrix;
  vector< vector< int   > > neighbors;
  vector< float > distances;
  vector< int   > previous ;
  vector< int   > path;
  vector< vector< float > > v2v_tortuosities;;
  float sigma = 1.0;

  computeAuxStructures(gr, distanceMatrix, neighbors, probs, v2v_paths);

  float maxEdgeVal = maxValueMatrix(distanceMatrix);

  // Output
  int nPoints = gr->cloud->points.size();
  paths.resize(nPoints);
  for(int i = 0; i < nPoints; i++){
    paths[i].resize(nPoints);
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
    }
  }
  printf("\n");
}


// // Old method 
// void addToSolution
// (vector< int >& S,
 // vector<int>& path,
 // vector< int >& kids,
 // Graph3D* cpt,
 // vector< vector< Graph3Dw* > >& v2v_paths,
 // Cube<uchar, ulong>* notvisited
// ){
  // if(solutionContains(S, path[0])){
    // for(int i = 1; i < path.size(); i++){
      // kids[path[i-1]]++;
      // if(S[path[i]] >= 0){
        // printf("We are adding a node that already has been visited, exit\n");
        // exit(0);
      // }
      // S[path[i]] = path[i-1];
    // }
  // }
  // else if(solutionContains(S, path[path.size()-1])){
    // for(int i = path.size()-2; i >= 0; i--){
      // kids[path[i+1]]++;
      // if(S[path[i]] >= 0){
        // printf("We are adding a node that already has been visited, exit\n");
        // exit(0);
      // }
      // S[path[i]] = path[i+1];
    // }
  // }
  // else{
    // printf("The path added to the solution does is not compatible with the solution\n");
    // exit(0);
  // }

  // //mark all the points visited in notvisited as 0
  // vector< int > p0;
  // vector< int > p1;
  // for(int i = 0; i < path.size()-1; i++){
    // Graph3Dw* grpath = v2v_paths[path[i]][path[i+1]];
    // for(int np = 0; np < grpath->cloud->points.size(); np++){
      // Point3Dw* pt = dynamic_cast<Point3Dw*>(grpath->cloud->points[np]);
      // notvisited->micrometersToIndexes(grpath->cloud->points[np  ]->coords, p0);
      // notvisited->put_value_in_ellipsoid(0, p0[0], p0[1], p0[2], pt->weight,
                                         // pt->weight, pt->weight);
      // // printf("Drawing\n");
    // }
  // }

  // // And now elliminates from the solutio all those candidate points too close to the
  // //  path. It can be heavy, since it is done only few times
  // for(int nS = 0; nS < S.size(); nS++){
    // if(S[nS]==-1){
      // vector< float > p_c = cpt->cloud->points[nS]->coords;
      // for(int npe = 0; npe < path.size()-1; npe++){
        // int p0 = path[npe];
        // int p1 = path[npe+1];
        // Graph3Dw* path_v2v = v2v_paths[p0][p1];
        // for(int npp = 0; npp if(S[path[path.size()-1]]!=-1)< path_v2v->cloud->points.size(); npp++){
          // Point3Dw* pt = dynamic_cast<Point3Dw*>(path_v2v->cloud->points[npp]);
          // vector< float > v_dist = v_subs(p_c, pt->coords);
          // if(v_norm(v_dist) < pt->weight) //THRESHOLD
            // S[nS] = -2;
        // }
      // }
    // }
  // }

// }
