#ifndef GRAPHCUT_H_
#define GRAPHCUT_H_

// Include graph.h first to avoid compilation errors
// due to the re-definition of symbols
#include "kgraph.h"
#include "kgraph.cpp"
#include "maxflow.cpp"
#include <sstream>
#include "Point.h"
#include "VisibleE.h"
#include "float.h"
#include "Cube.h"

class Point3Di
{
 public:
  vector<float> w_coords; // world coordinates
  vector<int> coords;

  friend ostream& operator <<(ostream &os,const Point3Di &point);
};

template < class P=Point>
class GraphCut : public VisibleE
{
private:
    static int graphcut_id;

    void init();

public:
    string graphcut_name;

    vector< Point3Di* >* sink_points;

    vector< Point3Di* >* source_points;

    GraphCut();

    ~GraphCut();

    void addSinkPoint(Point3Di* point);

    void addSourcePoint(Point3Di* point);

    void clear();

    void draw(int x, int y, int z);

    void save(const string& filename);

    bool load(const char* fileName);

    template <class T, class U>
    void run_maxflow(Cube<T,U>* cube, int layer_id);

    static string className(){
        return "GraphCut";
    }

};

template< class P>
int GraphCut<P>::graphcut_id = 0;

template< class P>
GraphCut<P>::GraphCut() : VisibleE(){
    init();
    source_points = new vector<Point3Di*>;
    sink_points = new vector<Point3Di*>;
}

template< class P>
void GraphCut<P>::init(){
    std::stringstream out;
    out << graphcut_id;
    graphcut_name = "graphcut_" + out.str();
    graphcut_id++;
}

template< class P>
//GraphCut<P>::~GraphCut() : ~Visible(){
GraphCut<P>::~GraphCut() {
    for(vector< Point3Di* >::iterator itPoints = source_points->begin();
        itPoints != source_points->end(); itPoints++)
    {
        delete *itPoints;
    }
    delete source_points;
   for(vector< Point3Di* >::iterator itPoints = sink_points->begin();
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
void GraphCut<P>::draw(int x, int y, int z){
  glColor3f(1,0,0);
  for(vector< Point3Di* >::iterator itPoints = source_points->begin();
      itPoints != source_points->end(); itPoints++)
    {
      if(z==-1 || (*itPoints)->coords[2] == z)
	{
	  glPushMatrix();
	  glTranslatef((*itPoints)->w_coords[0],(*itPoints)->w_coords[1],(*itPoints)->w_coords[2]);
	  glutSolidSphere(0.5, 10, 10);
	  glPopMatrix();
	}
    }
  glColor3f(0,1,0);
  for(vector< Point3Di* >::iterator itPoints = sink_points->begin();
      itPoints != sink_points->end(); itPoints++)
    {
      if(z==-1 || (*itPoints)->coords[2] == z)
	{
	  glPushMatrix();
	  glTranslatef((*itPoints)->w_coords[0],(*itPoints)->w_coords[1],(*itPoints)->w_coords[2]);
	  glutSolidSphere(0.5, 10, 10);
	  glPopMatrix();
	}
    }
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

  writer << "SOURCE\n";
  for(vector< Point3Di* >::iterator itPoints = source_points->begin();
      itPoints != source_points->end(); itPoints++)
    {
      writer << (*itPoints)->coords[0] << " " << (*itPoints)->coords[1] << " " << (*itPoints)->coords[2] <<  std::endl;
    }

  writer << "SINK\n";
  for(vector< Point3Di* >::iterator itPoints = sink_points->begin();
      itPoints != sink_points->end(); itPoints++)
    {
      writer << (*itPoints)->coords[0] << " " << (*itPoints)->coords[1] << " " << (*itPoints)->coords[2] <<  std::endl;
    }

  writer.close();
}

template< class P>
bool GraphCut<P>::load(const char* fileName){
  // TODO
  std::ifstream reader(fileName);
  if(!reader.good())
    {
      printf("Graphcut: loader can not find the file: %s\n",fileName);
      return false;
    }

/*   while(!reader.eof()) */
/*     { */
/*       reader.getline(line,1024); */
/*       sprintf(); */
/*     } */

  string gLine;
  char type = 0; // SOURCE
  while(getline(reader, gLine))
    {
      if(gLine == "SOURCE")
	{
	  printf("Graphcut : SOURCE\n");
	  type = 0;
	}
      if(gLine == "SINK")
	{
	  printf("Graphcut : SINK\n");
	  type = 1;
	}
      else
	{
	  istringstream gStream(gLine);
	  string gElement;
	  Point3Di* p = new Point3Di();
	  // read every element from the line that is seperated by spaces
	  while(getline(gStream, gElement, ' '))
	    {
	      istringstream iss(gElement);
	      float element;
	      p->coords.push_back(element);
	    }
	  if(type == 0)
	    {
	      source_points->push_back(p);
	    }
	  else
	    {
	      sink_points->push_back(p);
	    }
	}
    }


  reader.close();
  return true;
}

template< class P>
void GraphCut<P>::addSourcePoint(Point3Di* point)
{
    source_points->push_back(point);
}

template< class P>
void GraphCut<P>::addSinkPoint(Point3Di* point)
{
    sink_points->push_back(point);
}

template< class P>
template<class T, class U>
  void GraphCut<P>::run_maxflow(Cube<T,U>* cube, int layer_id)
{
  typedef maxflow::Graph<float,float,float> GraphType;
 
  int startK, endK;
  int ni,nj,nk;
  float weightToSource;
  float weightToSink;
  float weight;
  const float sigma = 5.f;
  // TODO : compute the weight k
  // (weight of edge between mark object to the source or mark background to the sink)
  //float K = FLT_MAX;
  float K = 100;

  if(layer_id == -1)
    {
      startK = 0;
      endK = cube->cubeDepth;
      ni = cube->cubeWidth;
      nj = cube->cubeHeight;
      nk = cube->cubeDepth;
    }
  else
    {
      startK = layer_id;
      endK = layer_id + 1;
      ni = cube->cubeWidth;
      nj = cube->cubeHeight;
      nk = 1;
    }

  GraphType::node_id node_ids[ni][nj][nk];

  // TODO / Computer correct parameters
  int nNodes = ni*nj*nk;
  int nEdges = nNodes*3;
  GraphType *g = new GraphType(/*estimated # of nodes*/ nNodes,
			       /*estimated # of edges*/ nEdges);

  // Debug
  //int k = 0;
  printf("Cube : %d %d %d\n", ni, nj, nk);
  nEdges = 0;

  int i,j,k;
  for(i = 0;i<cube->cubeWidth;i++)
    {
      for(j = 0;j<cube->cubeHeight;j++)
	{
	  for(k = 0;k<nk;k++)
	    {
	      node_ids[i][j][k] = g->add_node();
	    }
	}
    }

  printf("node_ids[i][j][k] : %d %d %d %d\n", i,j,k,node_ids[i-1][j-1][k-1]);

  //for(int i = 1;i<cube->cubeWidth-2;i++)
  for(int i = 0;i<cube->cubeWidth;i++)
    {
    //for(int j = 1;j<cube->cubeHeight-2;j++)
      for(int j = 0;j<cube->cubeHeight;j++)
	for(int k = 0;k<nk;k++)
	{
	  weightToSink = 0;
	  weightToSource = 0;
	  // Compute weights to source and sink nodes
	  for(vector< Point3Di* >::iterator itPoint=source_points->begin();
	      itPoint != source_points->end();itPoint++)
	    {
	      if((*itPoint)->coords[0] == i && (*itPoint)->coords[1] == j) // && (*itPoint)->coords[2] == k)
		{
		  printf("Source found %d %d\n", i, j);
		  weightToSource = K;
		}
	    }
	  for(vector< Point3Di* >::iterator itPoint=sink_points->begin();
	      itPoint != sink_points->end();itPoint++)
	    {
	      if((*itPoint)->coords[0] == i && (*itPoint)->coords[1] == j) // && (*itPoint)->coords[2] == k)
		{
		  printf("Sink found %d %d\n", i, j);
		  weightToSink = K;
		}
	    }

          g->add_tweights(node_ids[i][j][k],weightToSource,weightToSink);

	  if(i+1 < ni && (node_ids[i][j][k] != node_ids[i+1][j][k]))
	    {
	      weight = exp(-pow((cube->at(i,j,k+startK)-cube->at(i+1,j,k+startK))/sqrt(2.f)/sigma,2.f));
	      g->add_edge(node_ids[i][j][k], node_ids[i+1][j][k], weight, weight);
	      nEdges++;
	    }

	  if(j+1 < nj && (node_ids[i][j][k] != node_ids[i][j+1][k]))
	    {
	      weight = exp(-pow((cube->at(i,j,k+startK)-cube->at(i,j+1,k+startK))/sqrt(2.f)/sigma,2.f));
	      g->add_edge(node_ids[i][j][k], node_ids[i][j+1][k], weight, weight);
	      nEdges++;
	    }

	  if(k+1 < nk && (node_ids[i][j][k] != node_ids[i][j][k+1]))
	    {
	      weight = exp(-pow((cube->at(i,j,k+startK)-cube->at(i,j,k+1+startK))/sqrt(2.f)/sigma,2.f));
	      g->add_edge(node_ids[i][j][k], node_ids[i][j][k+1], weight, weight);
	      nEdges++;
	    }


/*
	  for(vector< Point3Di* >::iterator itPoint=source_points->begin();
	      itPoint != source_points->end();itPoint++)
	    {
	      if((*itPoint)->coords[0] == (i-1) && (*itPoint)->coords[1] == (j-1)) // && (*itPoint)->coords[2] == k)
		{
		  printf("Source found %d %d\n", i-1, j-1);
		  weightToSource = K;
		}
	    }
	  for(vector< Point3Di* >::iterator itPoint=sink_points->begin();
	      itPoint != sink_points->end();itPoint++)
	    {
	      if((*itPoint)->coords[0] == (i-1) && (*itPoint)->coords[1] == (j-1)) // && (*itPoint)->coords[2] == k)
		{
		  printf("Sink found %d %d\n", i-1, j-1);
		  weightToSink = K;
		}
	    }

	  g->edit_tweights(node_ids[i-1][j-1],weightToSource,weightToSink);

	  weight = exp(-pow((cube->at(i-1,j-1,k)-cube->at(i,j-1,k))/sqrt(2.f)/sigma,2.f));
	  g->add_edge(node_ids[i-1][j-1], node_ids[i][j-1], weight, weight);

	  weight = exp(-pow((cube->at(i-1,j-1,k)-cube->at(i,j,k))/sqrt(2.f)/sigma,2.f));
	  g->add_edge(node_ids[i-1][j-1], node_ids[i][j], weight, weight);

	  weight = exp(-pow((cube->at(i-1,j-1,k)-cube->at(i-1,j,k))/sqrt(2.f)/sigma,2.f));
	  g->add_edge(node_ids[i-1][j-1], node_ids[i-1][j], weight, weight);

	  nEdges +=3;

	  if(i>1)
	    {
	      weight = exp(-pow((cube->at(i-1,j-1,k)-cube->at(i-2,j,k))/sqrt(2.f)/sigma,2.f));
	      g->add_edge(node_ids[i-1][j-1], node_ids[i-2][j], weight, weight);
	      nEdges++;
	    }
*/

/*
	  for(int s=i-1;s<=i;s++)
	    {
	      if(s>=0)
		{
		  for(int t=j-1;t<=j;t++)
		    {
		      if(t>=0)
			{
			  //int u = k; // debug
			  for(int u=k-1;u<=k;u++)
			  {
			    if(u>=0 && (node_ids[i][j][k] != node_ids[s][t][u])) //(i!=s || j!=t || k!=u))
			    //if(i!=s || j!=t)
				{
				  weight = exp(-pow((cube->at(i,j,k)-cube->at(s,t,u))/sqrt(2.f)/sigma,2.f));
				  g->add_edge(node_ids[i][j][k], node_ids[s][t][u], weight, weight);
				  //g->add_edge(node_ids[i][j], node_ids[s][t], weight, weight);
				  nEdges++;

				 //if(cube->type == "uchar")
				   //printf("i: %d, j: %d, s: %d, t: %d, k: %d, w: %f, c:%c\n",i,j,s,t,k,weight, cube->at(i,j,k));
 				  //else if(cube->type == "float")
 				    //printf("i: %d, j: %d, s: %d, t: %d, k: %d, w: %f, c:%f\n",i,j,s,t,k,weight, cube->at(i,j,k));
				}
			      }
			}
		    }
		}
	    }
*/

	}
    }

  int flow = g->maxflow();

  printf("Edges: %d %d\n", nEdges, flow);
			
  // TODO : debug only, get rid of this part
  for(int k = 0;k<nk;k++)
    {
      IplImage* img = cvCreateImage( cvSize(cube->cubeWidth, cube->cubeHeight), 8, 1 );
      uchar* ptrImage;

      for(int i = 0;i<cube->cubeWidth;i++)
	for(int j = 0;j<cube->cubeHeight;j++)
	  {
	    ptrImage = &((uchar*)(img->imageData + img->widthStep*j))[i];
	    if(g->what_segment(node_ids[i][j][k]) == GraphType::SOURCE)
	      {
		//printf("Image : SOURCE\n");
		*ptrImage = 255;
	      }
	    else
	      {
		//printf("Image : SINK\n");
		*ptrImage = 0;
	      }
	  }

      std::string s;
      std::stringstream out;
      out << k;
      s = "graphcut_" + out.str();
      s += ".jpg";
      cvSaveImage(s.c_str(), img);
      cvReleaseImage(&img);
    }

  delete g;
}

#endif //GRAPHCUT_H_
