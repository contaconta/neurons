#ifndef GRAPHCUT_H_
#define GRAPHCUT_H_

// Include graph.h first to avoid compilation errors
// due to the re-definition of symbols
#include "graphCut/dynamicGraph.h"
#include "graphCut/dynamicGraph.cpp"
#include <sstream>
#include "Point.h"
#include "VisibleE.h"
#include "float.h"
#include "Cube.h"

//template < class P=Point, class T=int, class U=int>
template < class P=Point>
class GraphCut : public VisibleE
{
private:
    static int graphcut_id;

    void init();

public:
    string graphcut_name;

    vector< Point* >* sink_points;

    vector< Point* >* source_points;

    GraphCut();

//Contour(vector< Point* >* _points);

    ~GraphCut();

    void addSinkPoint(Point* point);

    void addSourcePoint(Point* point);

    void clear();

    void draw();

    void save(const string& filename);

    bool load(istream &in);

template <class T, class U>
void run_maxflow(Cube<T,U>* cube);

    static string className(){
        return "GraphCut";
    }

};

template< class P>
int GraphCut<P>::graphcut_id = 0;

template< class P>
GraphCut<P>::GraphCut() : VisibleE(){
    init();
    source_points = new vector<Point*>;
    sink_points = new vector<Point*>;
}

template< class P>
void GraphCut<P>::init(){
    std::string s;
    std::stringstream out;
    out << graphcut_id;
    graphcut_name = "graphcut " + out.str();
    graphcut_id++;
}

template< class P>
//GraphCut<P>::~GraphCut() : ~Visible(){
GraphCut<P>::~GraphCut() {
    for(vector< Point* >::iterator itPoints = source_points->begin();
        itPoints != source_points->end(); itPoints++)
    {
        delete *itPoints;
    }
    delete source_points;
   for(vector< Point* >::iterator itPoints = sink_points->begin();
        itPoints != sink_points->end(); itPoints++)
    {
        delete *itPoints;
    }
    delete sink_points;
}

template< class P>
void GraphCut<P>::clear(){
    source_points->clear();
    sink_points->clear();
}

template< class P>
void GraphCut<P>::draw(){
    glColor3f(1,0,0);
    glPushAttrib(GL_LINE_BIT);
    glLineWidth(6.0f);
    glBegin(GL_LINE_STRIP);
    for(vector< Point* >::iterator itPoints = source_points->begin();
        itPoints != source_points->end(); itPoints++)
    {
        glVertex3f((*itPoints)->coords[0],(*itPoints)->coords[1],(*itPoints)->coords[2]);
    }
    for(vector< Point* >::iterator itPoints = sink_points->begin();
        itPoints != sink_points->end(); itPoints++)
    {
        glVertex3f((*itPoints)->coords[0],(*itPoints)->coords[1],(*itPoints)->coords[2]);
    }
    glEnd();
    glPopAttrib();
}

template< class P>
void GraphCut<P>::save(const string& filename){

  if(source_points->size()==0 && sink_points->size()==0)
        return;

    std::ofstream writer(filename.c_str());

    if(!writer.good())
    {
        printf("Error while creating file %s\n", filename.c_str());
        return;
    }

    for(vector< Point* >::iterator itPoints = source_points->begin();
        itPoints != source_points->end(); itPoints++)
    {
        writer << **itPoints <<  std::endl;
    }
    for(vector< Point* >::iterator itPoints = sink_points->begin();
        itPoints != sink_points->end(); itPoints++)
    {
        writer << **itPoints <<  std::endl;
    }

    writer.close();
}

template< class P>
bool GraphCut<P>::load(istream &in){
//  int start = in.tellg();
//  in >> p0;
//  if(in.fail()){
//    in.clear();
//    in.seekg(start+1);
//    return false;
//  }
//  in >> p1;
//  if(in.fail()){
//    in.clear();
//    in.seekg(start+1);
//    return false;
//  }
//  return true;
}

template< class P>
void GraphCut<P>::addSourcePoint(Point* point)
{
    source_points->push_back(point);
}

template< class P>
void GraphCut<P>::addSinkPoint(Point* point)
{
    sink_points->push_back(point);
}

template< class P>
template<class T, class U>
void GraphCut<P>::run_maxflow(Cube<T,U>* cube)
{
  typedef DynamicGraph<int,int,int> GraphType;
  // TODO / Computer correct parameters
  GraphType *g = new GraphType(/*estimated # of nodes*/ 2,
			       /*estimated # of edges*/ 1); 
  float weightToSource;
  float weightToSink;
  float weight;
  const float sigma = 5.f;
  // TODO : compute the weight k
  // (weight of edge between mark object to the source or mark background to the sink)
  float K = FLT_MAX;
  GraphType::node_id node_ids[cube->cubeWidth][cube->cubeHeight][cube->cubeDepth];

  int k = 1; // debug

  for(int i = 0;i<cube->cubeWidth;i++)
    {
    for(int j = 0;j<cube->cubeHeight;j++)
      //for(int k = 0;k<cube->cubeDepth;k++)
	{
	  weightToSink = K;
	  weightToSource = K;
	  // Compute weights to source and sink nodes
	  for(vector< Point* >::iterator itPoint=source_points->begin();
	      itPoint != source_points->end();itPoint++)
	    {
	      if((*itPoint)->coords[0] == i && (*itPoint)->coords[1] == j && (*itPoint)->coords[2] == k)
		weightToSource = 0;
	    }
	  for(vector< Point* >::iterator itPoint=sink_points->begin();
	      itPoint != sink_points->end();itPoint++)
	    {
	      if((*itPoint)->coords[0] == i && (*itPoint)->coords[1] == j && (*itPoint)->coords[2] == k)
		weightToSink = 0;
	    }
	  node_ids[i][j][k] = g->add_node();
	  g->edit_tweights(node_ids[i][j][k],weightToSource,weightToSink);
// 	  if(i!=0)
// 	    {
// 	      weight = exp(-pow((cube->at(i,j,k)-cube->at(i-1,j,k))/sqrt(2.f)/sigma,2.f)); */
// 	      g->add_edge(node_ids[i][j][k], node_ids[i-1][j][k], weight, weight);

// 	      if(j!=0)
// 		{
// 		  weight = exp(-pow((cube->at(i,j,k)-cube->at(i-1,j-1,k))/sqrt(2.f)/sigma,2.f));
// 		  g->add_edge(node_ids[i][j][k], node_ids[i-1][j-1][k], weight, weight);
// 		}
// 	    }
// 	  else if(j!=0)
// 	    {
// 	      weight = exp(-pow((cube->at(i,j,k)-cube->at(i,j-1,k))/sqrt(2.f)/sigma,2.f));
// 	      g->add_edge(node_ids[i][j][k], node_ids[i][j-1][k], weight, weight);
// 	    }

	  // Add edges
	  for(int s=i-1;s<=i;s++)
	    {
	      if(s!=0)
		{
		  for(int t=j-1;t<=j;t++)
		    {
		      if(t!=0)
			{
			  for(int u=k-1;u<=k;u++)
			    {
			      if(u!=0)
				{
				  weight = exp(-pow((cube->at(i,j,k)-cube->at(s,t,u))/sqrt(2.f)/sigma,2.f));
				  g->add_edge(node_ids[i][j][k], node_ids[s][t][u], weight, weight);
				}
			    }
			}
		    }
		}
	    }
	}
    }

  int flow = g->maxflow();
			
  // TODO : debug only, get rid of this part
  IplImage* img = cvCreateImage( cvSize(cube->cubeWidth, cube->cubeHeight), 8, 1 );
  uchar* ptrImage;

  for(int i = 0;i<cube->cubeWidth;i++)
    for(int j = 0;j<cube->cubeHeight;j++)
      {
	ptrImage = &((uchar*)(img->imageData + img->widthStep*j))[i];
	if(g->what_segment(node_ids[i][j][k]) == GraphType::SOURCE)
	  *ptrImage = 255;
	else
	  *ptrImage = 0;
      }

  cvSaveImage("maxflow.jpg", img);
}

#endif //GRAPHCUT_H_
