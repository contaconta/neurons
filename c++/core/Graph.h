#ifndef GRAPH_H_
#define GRAPH_H_

#include "Cloud.h"
#include "EdgeSet.h"
#include "Graph_P.h"

template< class P=Point, class E=Edge<P> >
class Graph : public Graph_P
{
public:
  EdgeSet<P, E> eset;
//   Cloud<P>   cloud;
  Cloud_P cloud;

  Graph() : Graph_P(){}

  Graph(string filename);

  Graph(Cloud_P* cl);

  bool load(istream &in);

  void save(ostream &out);

  void draw();

  double distanceEuclidean(Point* p1, Point* p2);

  // Calculates the Minimum spanning tree using prim algorithm
  void prim();


} ;

// FOR THE MOMENT THE DEFINITIONS GO HERE, BUT NEED TO BE TAKEN OUT

typedef Graph<Point3D, Edge<Point3D> > g1;

template< class P, class E>
Graph<P,E>::Graph(string filename) : Graph_P(){
  v_saveVisibleAttributes = true;
  loadFromFile(filename);
}

template< class P, class E>
Graph<P,E>::Graph(Cloud_P* cl) : Graph_P(){
  v_saveVisibleAttributes = true;
  this->cloud = *cl;
  eset.setPointVector(&cloud.points);
}

template< class P, class E>
bool Graph<P,E>::load(istream &in){
  string s;
  in >> s;
  assert(s=="<Graph");
  in >> s;
  assert(s == P::className());
  in >> s;
  assert(s == (E::className()+">"));
  assert(VisibleE::load(in));
  assert(cloud.load(in));
  assert(eset.load(in));
  eset.setPointVector(&cloud.points);
  return true;
}

template< class P, class E>
void Graph<P,E>::save(ostream &out){
  out << "<Graph " << P::className() << " " << E::className() << ">\n";
  VisibleE::save(out);
  cloud.save(out);
  eset.save(out);
  out << "</Graph>\n";
}

template< class P, class E>
void Graph<P,E>::draw(){
  if(v_glList == 0){
    v_glList = glGenLists(1);
    glNewList(v_glList, GL_COMPILE);
    VisibleE::draw();
    eset.draw();
    cloud.draw();
    glEndList();
  }
  else{
    glCallList(v_glList);
  }
  // VisibleE::draw();
  // eset.draw();
  // cloud.draw();
}

template< class P, class E>
double Graph<P,E>::distanceEuclidean(Point* p1, Point* p2){
  return p1->distanceTo(p2);
}

template< class P, class E>
void Graph<P,E>::prim(){
  printf("Graph<P,E>::prim of %i points  [", cloud.points.size());
  //Erases the edges
  eset.edges.resize(0);

  //Implementation of Prim's algtrithm as described in "Algorithms in C", by Robert Sedgewick
  vector< int > parents(cloud.points.size());
  vector< int > closest_in_tree(cloud.points.size());
  vector< double > distances_to_tree(cloud.points.size());

  //Initialization
  for(int i = 0; i < cloud.points.size(); i++)
    {
      parents[i] = -1;
      closest_in_tree[i] = 0;
      distances_to_tree[i] = distanceEuclidean(cloud.points[0],cloud.points[i]);
    }
  parents[0] = 0;

  for(int i = 0; i < cloud.points.size(); i++)
    {
      //Find the point that is not in the graph but closer to the graph
      int cls_idx = 0;
      double min_distance = 2e9;
      for(int cls = 0; cls < cloud.points.size(); cls++)
        if( (parents[cls] == -1) && (distances_to_tree[cls] < min_distance) ){
            min_distance = distances_to_tree[cls];
            cls_idx = cls;}

      //Now we update the data structures for new iterations
      parents[cls_idx] = closest_in_tree[cls_idx];
      double distance_update = 0;
      for(int cls2 = 0; cls2 < cloud.points.size(); cls2++)
        {
          if (parents[cls2] == -1)
            {
              distance_update = distanceEuclidean(cloud.points[cls2], cloud.points[cls_idx]);
              if(distance_update < distances_to_tree[cls2])
                {
                  distances_to_tree[cls2] = distance_update;
                  closest_in_tree[cls2] = cls_idx;
                }
            }
        }
      // if(i%max(cloud.points.size()/100,1)==0)
        // { printf("#"); fflush(stdout);}
    }

  printf("]\n");

  for(int i = 0; i < parents.size(); i++)
    eset.addEdge(i, parents[i]);
}



#endif
