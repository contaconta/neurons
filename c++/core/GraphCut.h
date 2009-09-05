#ifndef GRAPHCUT_H_
#define GRAPHCUT_H_

// Include graph.h first to avoid compilation errors
// due to the re-definition of symbols
#include "kgraph.h"
#include "kgraph.cpp"
#include "maxflow.cpp"
#include <iostream>
#include <sstream>
#include "Point.h"
#include "VisibleE.h"
#include "float.h"
#include "Cube.h"
#include "DoubleSet.h"

using namespace std;

// TODO : Move definition to a global file
//#define USE_ALPHA

typedef maxflow::Graph<float,float,float> GraphType;

template <class P=Point>
class GraphCut : public VisibleE
{
 private:
 static int graphcut_id;

 void init();

 Cube_P* m_cube;

 GraphType::node_id*** m_node_ids;
 int ni,nj,nk;

 // Variables used to know if we have to re-generate the display list
 int lastX, lastY, lastZ;

 // id of the display list used to draw the result of the segmentation
 GLuint m_segDL;

 // Temporary variables declared as members to speed things up
 int si,sj,sk;
 int ei,ej,ek;
 float fx,fy,fz;

 public:
 GraphType *m_graph;

 string graphcut_name;

 // true if the run_maxflow method is running
 bool running_maxflow;

 DoubleSet<P>* set_points;

 GraphCut(Cube_P* cube);

 ~GraphCut();
 
 void clear();
 
 void drawSeeds(int x, int y, int z);
 
 void draw(int x, int y, int z);
 
 void draw_in_DL(int x, int y, int z);

 void list();
 
 template <class T, class U>
 bool load(Cube<T,U>* cube, const char* fileName);
 
 void save(const string& filename);
 
 void setCube(Cube_P* cube);

 template <class T, class U>
 void run_maxflow(Cube<T,U>* cube, int layer_id);
 
 virtual string className(){
   return "GraphCut";
 }

};

template< class P>
int GraphCut<P>::graphcut_id = 0;

template< class P>
GraphCut<P>::GraphCut(Cube_P* cube) : VisibleE(){
    init();
    set_points = 0;
    m_node_ids = 0;
    m_graph = 0;
    m_cube = cube;
    lastX = lastY = lastZ = -999; // dummy value to indicate that the DL is not initialized
    m_segDL = -1;
    running_maxflow = false;
}

template< class P>
void GraphCut<P>::init(){
    std::stringstream out;
    out << graphcut_id;
    graphcut_name = "graphcut_" + out.str();
    graphcut_id++;
}

template< class P>
GraphCut<P>::~GraphCut() {
  // Free memory
  if(m_node_ids!=0)
    {
      for(int i = 0;i<ni;i++)
	{
	  for(int j = 0;j<nj;j++)
	    {
	      delete m_node_ids[i][j];
	    }
	  delete m_node_ids[i];
	}
      delete[] m_node_ids;
    }
  if(m_graph!=0)
    delete m_graph;

#ifdef USE_ALPHA
  m_cube->delete_alphas(ni,nj,nk);
#endif

  delete set_points;
}

template< class P>
void GraphCut<P>::clear(){
  set_points->clear();
}

template< class P>
void GraphCut<P>::drawSeeds(int x, int y, int z){
  /*
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
  */
  //set_points->draw();
}

template< class P>
void GraphCut<P>::draw(int x, int y, int z){
  //if(m_graph==0 && !running_maxflow)
  if(m_graph==0)
    return;

  if(lastX != x || lastY != y || lastZ != z)
    {
      // Create display list
      //printf("Create display list %d %d %d\n",x,y,z);

      if(m_segDL != -1)
	glDeleteLists(m_segDL, 1);

      m_segDL = glGenLists(1);
  
      glNewList(m_segDL,GL_COMPILE);
      draw_in_DL(x, y, z);
      glEndList();

      lastX = x;
      lastY = y;
      lastZ = z;
    }
  glCallList(m_segDL);
}

template< class P>
void GraphCut<P>::draw_in_DL(int x, int y, int z){

  if(x==-1)
    {
      si = 0;
      ei = ni;
    }
  else
    {
      si = x;
      ei = x + 1;
    }
  if(y==-1)
    {
      sj = 0;
      ej = nj;
    }
  else
    {
      sj = y;
      ej = y +1;
    }
  if(z==-1)
    {
      sk = 0;
      ek = nk;
    }
  else
    {
      sk = z;
      ek = z + 1;
    }

  glColor4f(0.0f, 0.0f, 1.0f, 0.5f);
  glBegin(GL_POINTS);
  for(int k=sk;k<ek;k++)
    for(int i = si;i<ei;i++)
      for(int j = sj;j<ej;j++)
	{
	  if(m_graph->what_segment(m_node_ids[i][j][k]) == GraphType::SOURCE)
	    {
#ifdef USE_ALPHA
              m_cube->alphas[i][j][k] = 1;
#endif
	      m_cube->indexesToMicrometers3(i,j,k,fx,fy,fz);
	      //cout << "vFloats: " << vFloats[0] << " " << vFloats[1] << " " << vFloats[2] << endl;
	      glVertex3f(fx, fy, fz);
	      //glPushMatrix();
	      //glTranslatef(vFloats[0],vFloats[1],vFloats[2]-1.0f);
	      //glutSolidSphere(0.5, 10, 10);
	      //glPopMatrix();
	    }
#ifdef USE_ALPHA
          else
            m_cube->alphas[i][j][k] = 0;
#endif
	}
  glEnd();
}

template< class P>
void GraphCut<P>::save(const string& filename){
  set_points->save(filename);
}

template< class P>
template<class T, class U>
bool GraphCut<P>::load(Cube<T,U>* cube, const char* fileName)
{
  set_points->load(fileName);
  return true;
}

template< class P>
void GraphCut<P>::setCube(Cube_P* cube)
{
  m_cube = cube;
}

/*
 * Run Min-Cut/Max-Flow algorithm for energy minimization
 * Energy function is defined as E = B + R
 * B = Boundary term, R = Regional term  
 */
template< class P>
template<class T, class U>
  void GraphCut<P>::run_maxflow(Cube<T,U>* cube, int layer_id)
{ 
  int i,j,k;
  int startK, endK;
  float weightToSource;
  float weightToSink;
  float weight;
  typename vector< PointDs<P>* >::iterator itPoint;
  const float sigma = 1/5.0f;
  // TODO : compute the weight k
  // (weight of edge between mark object to the source or mark background to the sink)
  //float K = FLT_MAX;
  float K = 100;

  running_maxflow = true;

  // Free memory
  if(m_node_ids!=0)
    {
      for(i = 0;i<ni;i++)
	{
	  for(j = 0;j<nj;j++)
	    {
	      delete m_node_ids[i][j];
	    }
	  delete m_node_ids[i];
	}
      delete[] m_node_ids;
    }
  if(m_graph!=0)
    delete m_graph;

  // Retrieve parameters
  // TODO : pass one argument for each axis.
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

  printf("GraphCut : %d %d %d\n", ni, nj, nk);

  // TODO : Compute correct parameters
  int nNodes = ni*nj*nk;
  int nEdges = nNodes*3;
  m_graph = new GraphType(nNodes, nEdges);

  m_node_ids = new GraphType::node_id**[ni];
  for(i = 0;i<ni;i++)
    {
      m_node_ids[i] = new GraphType::node_id*[nj];
      for(j = 0;j<nj;j++)
	{
	  m_node_ids[i][j] = new GraphType::node_id[nk];

	  for(k = 0;k<nk;k++)
	    {
	      m_node_ids[i][j][k] = m_graph->add_node();
	    }
	}
    }

#ifdef USE_ALPHA
  printf("GraphCut : Allocate alpha values\n");
  if(m_cube->alphas)
    m_cube->delete_alphas(ni, nj, nk);
  m_cube->allocate_alphas(ni, nj, nk);
#endif

  printf("GraphCut : Nodes added\n");

  // Compute histogram for boundary term
  /*
  int** histo_B = new int*[3];
  for(int i=0;i<3;i++)
    histo_B[i] = new int[255];
  */

  const int nbItemsPerBin = 25;
  const int histoSize = 255/nbItemsPerBin;
  float* histoSource = new float[histoSize];
  int binId;
  memset(histoSource,0,histoSize);
  for(itPoint=set_points->set1.begin();
      itPoint != set_points->set1.end();itPoint++)
    {
      binId = (int)(cube->at((*itPoint)->indexes[0], (*itPoint)->indexes[1], (*itPoint)->indexes[2])/nbItemsPerBin) - 1;
      histoSource[binId]++;
    }

  float* histoSink = new float[histoSize];
  memset(histoSink,0,histoSize);
  for(itPoint=set_points->set2.begin();
      itPoint != set_points->set2.end();itPoint++)
    {
      binId = (int)(cube->at((*itPoint)->indexes[0], (*itPoint)->indexes[1], (*itPoint)->indexes[2])/nbItemsPerBin) - 1;
      histoSink[binId]++;
    }

  // Normalize histograms
  for(int i = 0;i<histoSize;i++)
    {
      histoSource[i] /= histoSize;
      histoSink[i] /= histoSize;
    }

  nEdges = 0;
  for(int i = 0;i<ni;i++)
    {
      printf("* %d/%d\n",i,ni);
      for(int j = 0;j<nj;j++)
	{
	for(int k = 0;k<nk;k++)
	{
	  // Compute regional term
	  weightToSink = 0;
	  weightToSource = 0;

	  // Compute weights to source and sink nodes          
	  for(itPoint=set_points->set1.begin();
              itPoint != set_points->set1.end();itPoint++)
	    {
	      if((*itPoint)->indexes[0] == i && (*itPoint)->indexes[1] == j && (*itPoint)->indexes[2] == (k+startK))
		{
		  //printf("Source found %d %d %d\n", i, j, k+startK);
		  weightToSource = K;
                  break;
		}
	    }

	  for(itPoint=set_points->set2.begin();
              itPoint != set_points->set2.end();itPoint++)
	    {
	      if((*itPoint)->indexes[0] == i && (*itPoint)->indexes[1] == j && (*itPoint)->indexes[2] == (k+startK))
		{
		  //printf("Sink found %d %d %d\n", i, j, k+startK);
		  weightToSink = K;
                  break;
		}
	    }

          if(weightToSource != K)
            {
              // Get value from histogram
              binId = (int)(cube->at(i,j,k)/nbItemsPerBin) - 1;
              /*if(binId >= histoSize)
                printf("binId >= histoSize\n");*/
              weightToSource = histoSource[binId];
            }
          if(weightToSink != K)
            {
              // Get value from histogram
              binId = (int)(cube->at(i,j,k)/nbItemsPerBin) - 1;
              weightToSink = histoSink[binId];
            }

          m_graph->add_tweights(m_node_ids[i][j][k],weightToSource,weightToSink);


	  // Compute boundary term
	  // B(p,q) = exp(-(Ip - Iq)^2 / 2*sigma)/dist(p,q)
	  if(i+1 < ni && (m_node_ids[i][j][k] != m_node_ids[i+1][j][k]))
	    {
	      weight = exp(-pow((cube->at(i,j,k+startK)-cube->at(i+1,j,k+startK))*sigma,2.f));
	      //weight = exp(-pow((cube->at(i,j,k+startK)-cube->at(i+1,j,k+startK))/sqrt(2.f)/5.0f,2.f));
	      //weight = exp(-pow(((cube->at(i,j,k+startK)-cube->at(i+1,j,k+startK))/sqrt(2.f))/5.0f,2.f));
	      m_graph->add_edge(m_node_ids[i][j][k], m_node_ids[i+1][j][k], weight, weight);
	      nEdges++;
	    }

	  if(j+1 < nj && (m_node_ids[i][j][k] != m_node_ids[i][j+1][k]))
	    {
	      weight = exp(-pow((cube->at(i,j,k+startK)-cube->at(i,j+1,k+startK))*sigma,2.f));
	      m_graph->add_edge(m_node_ids[i][j][k], m_node_ids[i][j+1][k], weight, weight);
	      nEdges++;
	    }
	  if(k+1 < nk && (m_node_ids[i][j][k] != m_node_ids[i][j][k+1]))
	    {
	      weight = exp(-pow((cube->at(i,j,k+startK)-cube->at(i,j,k+1+startK))*sigma,2.f));
	      m_graph->add_edge(m_node_ids[i][j][k], m_node_ids[i][j][k+1], weight, weight);
	      nEdges++;
	    }
	
	  /*	  
	  if(i+1 < ni && (m_node_ids[i][j][k] != m_node_ids[i+1][j][k]))
	    {
	      weight = exp(-pow((cube->at(i,j,k+startK)-cube->at(i+1,j,k+startK))/sqrt(2.f)/sigma,2.f));
	      m_graph->add_edge(m_node_ids[i][j][k], m_node_ids[i+1][j][k], weight, weight);
	      nEdges++;
	    }

	  if(j+1 < nj && (m_node_ids[i][j][k] != m_node_ids[i][j+1][k]))
	    {
	      weight = exp(-pow((cube->at(i,j,k+startK)-cube->at(i,j+1,k+startK))/sqrt(2.f)/sigma,2.f));
	      m_graph->add_edge(m_node_ids[i][j][k], m_node_ids[i][j+1][k], weight, weight);
	      nEdges++;
	    }

	  if(k+1 < nk && (m_node_ids[i][j][k] != m_node_ids[i][j][k+1]))
	    {
	      weight = exp(-pow((cube->at(i,j,k+startK)-cube->at(i,j,k+1+startK))/sqrt(2.f)/sigma,2.f));
	      m_graph->add_edge(m_node_ids[i][j][k], m_node_ids[i][j][k+1], weight, weight);
	      nEdges++;
	    }
	  */
	  

/*
	  for(vector< Point3Di* >::iterator itPoint=source_points->begin();
	      itPoint != source_points->end();itPoint++)
	    {
	      if((*itPoint)->indexes[0] == (i-1) && (*itPoint)->indexes[1] == (j-1)) // && (*itPoint)->indexes[2] == k)
		{
		  printf("Source found %d %d\n", i-1, j-1);
		  weightToSource = K;
		}
	    }
	  for(vector< Point3Di* >::iterator itPoint=sink_points->begin();
	      itPoint != sink_points->end();itPoint++)
	    {
	      if((*itPoint)->indexes[0] == (i-1) && (*itPoint)->indexes[1] == (j-1)) // && (*itPoint)->indexes[2] == k)
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
    }

  printf("GraphCut : Computing max flow\n");
  int flow = m_graph->maxflow();
			
  // TODO : debug only, get rid of this part
  for(int k = 0;k<nk;k++)
    {
      IplImage* img = cvCreateImage( cvSize(cube->cubeWidth, cube->cubeHeight), 8, 1 );
      uchar* ptrImage;

      for(int i = 0;i<cube->cubeWidth;i++)
	for(int j = 0;j<cube->cubeHeight;j++)
	  {
	    ptrImage = &((uchar*)(img->imageData + img->widthStep*j))[i];
	    if(m_graph->what_segment(m_node_ids[i][j][k]) == GraphType::SOURCE)
	      {
#ifdef USE_ALPHA
                m_cube->alphas[i][j][k] = 255;
		printf("Image : SOURCE %d\n", m_cube->alphas[i][j][k]);
#endif
		*ptrImage = 255;
	      }
	    else
	      {
		//printf("Image : SINK\n");
		*ptrImage = 0;
#ifdef USE_ALPHA
                m_cube->alphas[i][j][k] = 0;
#endif
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

#ifdef USE_ALPHA
  printf("GraphCut : Reloading cube texture\n");
  // reload cube texture
  m_cube->load_texture_brick(m_cube->nRowToDraw, m_cube->nColToDraw);
#endif

  // Cleaning
  delete[] histoSource;
  delete[] histoSink;

  printf("GraphCut : Max flow algorithm has ended\n");
  running_maxflow = false;
}

template< class P>
void GraphCut<P>::list() {
    for(vector< PointDs<>* >::iterator itPoints = set_points->set1.begin();
        itPoints != set_points->set1.end(); itPoints++)
    {
      cout << "Coords: " << (*itPoints)->coords[0] << " " << (*itPoints)->coords[1] << " " << (*itPoints)->coords[2] << endl;
      cout << "indexes: " << (*itPoints)->indexes[0] << " " << (*itPoints)->indexes[1] << " " << (*itPoints)->indexes[2] << endl;
    }
   for(vector< PointDs<>* >::iterator itPoints = set_points->set2.begin();
        itPoints != set_points->set2.end(); itPoints++)
    {
      cout << "Coords: " << (*itPoints)->coords[0] << " " << (*itPoints)->coords[1] << " " << (*itPoints)->coords[2] << endl;
      cout << "indexes: " << (*itPoints)->indexes[0] << " " << (*itPoints)->indexes[1] << " " << (*itPoints)->indexes[2] << endl;
    }
}

#endif //GRAPHCUT_H_
