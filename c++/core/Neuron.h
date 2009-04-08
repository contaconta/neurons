/** Class Neuron.
 * Data structure to keep the information contained in the NeuroLucida asc format
 *
 * German Gonzalez
 * 20060919
 */

#ifndef NEURON_H_
#define NEURON_H_

class NeuronVolume;
class ascParser2;

#include "neseg.h"

#include "ascParser2.h"
#include "Cube.h"
#include "VisibleE.h"
#include "Image.h"
#include "utils.h"
#include "CloudFactory.h"
#include "Cloud_P.h"
#include "Cube_P.h"

using namespace std;
//using std::string;

class NeuronPoint
{
public:
  vector< float > coords;
  int noSenseNumber;
  int pointNumber;
  NeuronPoint();
  NeuronPoint
  (float x, float y, float z, float w,
   int noSenseNumber = 1, int pointNumber = 1, NeuronSegment* parent = NULL);
  NeuronPoint(vector< float > coords);
  virtual ~NeuronPoint();
  void print();
  string name;
  NeuronSegment* parent;
  NeuronVolume*  neuronVolume;
};

class NeuronColor
{
public:
  vector< float > coords;
  NeuronColor();
  NeuronColor(float R, float G, float B);
  NeuronColor(vector< float > coords);
  void set(float R, float G, float B);
  virtual ~NeuronColor();
  string name;
  void print();
};

/** Neurolucida markers*/
class NeuronMarker
{
public:
  NeuronMarker();
  NeuronMarker(string name, string type, NeuronColor color);
  NeuronMarker(string name, string type, vector< NeuronPoint > points, NeuronColor color);
  virtual ~NeuronMarker();
  void print();
  void draw();

  string name;
  string type;
  NeuronColor color;
  vector< NeuronPoint > points;
};


class NeuronSegment
{
public:
  NeuronSegment();

  NeuronSegment(NeuronSegment* parent,
          string ending = "Incomplete",
          NeuronColor color = NeuronColor(0,0,0));

  NeuronSegment(NeuronSegment* parent,
                vector< NeuronPoint > points,
                vector< NeuronMarker > markers,
                vector< NeuronSegment* > childs,
                string ending = "",
                NeuronColor color = NeuronColor(0,0,0));
  virtual ~NeuronSegment();

  NeuronSegment* parent;
  vector< NeuronPoint > points;
  vector< NeuronPoint > spines;
  vector< NeuronMarker > markers;
  vector< NeuronSegment* > childs;
  NeuronPoint root;
  string ending;

  /** Name of the segment. It will be recursive and in the form:
   * A[n]-1-2-1-1-1...  for axones, where n is the number of axon
   * D[n]-1-1-2-1-1...  for dendrites.
   */
  string name;
  NeuronColor  color;

  void print(int n = 0);
};

class NeuronContour
{
public:
  NeuronContour();
  virtual ~NeuronContour();
  void print();

  string name;
  vector< NeuronPoint > points;
  NeuronColor color;
};




class Neuron : public VisibleE
{
public:

  //End of the extra classes

  Neuron();
  Neuron(string fileName);
  virtual ~Neuron();

  /** Saves the neuron as an asc file*/
  void save(string filename);

  /** Prints the neuron.*/
  void print();

  /** Draws the neuron in Opengl.*/
  void draw();

  /** Draws the neuron in opengl. No matrix support / configuration.*/
  void drawInOpenGl(bool includeCorrection = false, float max_width = 1e6);

  /** Draws the neuron in opengl. */
  void drawInOpenGlAsLines(bool includeCorrection = false);

  /** Draws the neuron in opengl as a sequence of balls.No matrix support / configuration.*/
  void drawInOpenGlAsDots(bool includeCorrection = false);

  /** Draws the neuron in opengl with the matrix provided and including the offset correction in micrometers*/
  void drawInOpenGlWithCorrectionInMicrometers
  (
   float neuron_disp3DX = 0,  float neuron_disp3dY = 0, float neuron_disp3DZ = 0,
   float neuron_zoom3DX = 0,  float neuron_zoom3DY = 0, float neuron_zoom3DZ = 0,
   float neuron_rot3DX = 0,   float neuron_rot3DY = 0,  float neuron_rot3DZ = 0 );

  /** Default GL matrices support.*/
  void setUpGlMatrices(); //

  /** Draws a segment as a set of connected cylinders. It is recursive, so that the childs are
   * also drawn.*/
  void drawSegment
  (NeuronSegment* segment,vector< float > root,
   bool includeCorrection = false, float max_width = 0);

  /** Draws a segment with a sphere on each of it's points*/
  void drawSegmentAsDots
  (NeuronSegment* segment, vector< float > root, bool includeCorrection = false);

  void drawSegmentAsLines(NeuronSegment* segment, vector< float > root, bool includeOffset);

  /** Draws a segment with the correction in pixels (micrometers)*/
  void drawSegmentWithCorrectionInMicrometers
  (NeuronSegment* segment,vector< float > root,
   float neuron_disp3DX = 0,  float neuron_disp3dY = 0, float neuron_disp3DZ = 0,
   float neuron_zoom3DX = 0,  float neuron_zoom3DY = 0, float neuron_zoom3DZ = 0,
   float neuron_rot3DX = 0,   float neuron_rot3DY = 0,  float neuron_rot3DZ = 0 );

  //Draws the spines as spheres
  void drawSpines(vector< NeuronPoint>& spines);

  void getMaxPoint(NeuronSegment* segment, double* maxX, double* maxY, double* maxZ
                   , double* minX, double* minY, double* minZ,
                   int* nSegments);

  /** For easy operations on vectors of floats */
  vector<float> addVectors(vector< float > a, vector< float > b){
    vector< float > result;
    for(int i = 0; i < a.size(); i++)
      result.push_back(a[i]+b[i]);
    return result;
  }

  /** Finds the closest axon or Dendrite to a given point.*/
  NeuronSegment* findClosestAxonOrDendrite(double x, double y, double z);

  /** Finds the closest segment to a given point from the whole Neuron.*/
  NeuronSegment* findClosestSegment(double x, double y, double z);

  /** Finds the next point to a given one in the neuron tree.*/
  NeuronPoint*    findNextPoint(double x, double y, double z);

  /** Finds the index of the closest point of a segment*/
  int findIndexOfClosestPointInSegment(double x, double y, double z, NeuronSegment* segment);

  /** Finds the closest subsegment from a given one to a point.*/
  NeuronSegment* findClosestSubsegment
  (double x, double y, double z, NeuronSegment* segment, double* distance);

  /** Calculates the euclidean distance between a point and a segment.*/
  double distancePointSegmentWithChilds
  (double x, double y, double z, NeuronSegment* pepe);

  /** Gets the distance between the points of a segment and the point. Does not include children.*/
  double distancePointSegment
  (double x, double y, double z, NeuronSegment* pepe);

  /** Euclidean distance among two points.*/
  double distancePointPoint(double x, double y, double z, NeuronPoint& point);

  /** Breaks a segment in two. FIXME!! It is not working*/
  NeuronSegment* splitSegment(NeuronSegment* toSplit, int pointIdx);

  /** Adds the diffx, diffy, diffz to all the points that follow the given one in the segment. The units are neuron units. */
  void applyRecursiveOffset
  (NeuronSegment* closestS,
   int pointNumber, 
   double diffx, double diffy, double diffz
   );

  /** Changes the position of the point at position i of segment j.*/
  void changePointPosition
  (NeuronSegment* segment,
   int pointNumber,
   double posx,
   double posy,
   double posz);

  /** Prints some info of the neuron.*/
  void printStatistics();

  /** Gets the widths of the segment and puts them in the vectos.*/
  void getWidthDistribution
  (vector<float>& widths, vector<int>& occurrences, NeuronSegment* segment);

  /** Returns a random point of the neuron. */
  NeuronPoint* getRandomPoint();

  /**Find the closest point of the neuron to the coordinates and returns its name.*/
  NeuronPoint* findClosestPoint(double x, double y, double z);

  /** Loads the offsets calculated with the cross-correlation from an offset file into the correlationOffsets map*/
  void loadCorrelationOffsets(string path = "/media/data/neuron1/volumes_064_064_064/positive/");

  /** Returns a vector with the points of the neuron that have minWidth < width < maxWidth. The width is given in neuron coordinates, and the points also in neuron coordinates*/
  vector< NeuronPoint > getPointsWithWidth(float minWidth, float maxWidth);

  /** Gets all the points of the segment with minWidth < width < maxWidth. */
  void getPointsWithWidthRecursively
  (float minWidth, float maxWidth, NeuronSegment* segment, vector< NeuronPoint >& points);

  /** Draws the neuron as voxels in the cube.*/
  void renderInCube(Cube<uchar,ulong>* cube,
                    Cube<float,double>* theta = NULL,
                    Cube<float,double>* phi = NULL,
                    Cube<float,double>* scale = NULL,
                    float min_width = 0,
                    float renderScale = 1.0);

  /**Draws the segment between P1 and P2 in the cube with radius width*/
  void renderEdgeInCube
  (NeuronPoint* p1, NeuronPoint* p2,
   Cube<uchar,ulong>* cube,
   Cube<float,double>* theta = NULL,
   Cube<float,double>* phi = NULL,
   Cube<float,double>* scale = NULL,
   float renderScale = 1.0);

  void renderSegmentInCube
  (NeuronSegment* segment, Cube<uchar,ulong>* cube,
   Cube<float, double>* theta = NULL,
   Cube<float, double>* phi = NULL,
   Cube<float,double>* scale = NULL,
   float min_width = 0,
   float renderScale = 1.0);

  /** Draws the dendrite in an image.*/
  void renderInImage(Image<float>* img,
                     Image<float>* theta = NULL,
                     Image<float>* width = NULL,
                     float min_width = 0,
                     float width_scale = 1.0);

  /**Draws the segment between P1 and P2 in the image with radius width*/
  void renderEdgeInImage
  (NeuronPoint* p1, NeuronPoint* p2,
   Image<float>* img,
   Image<float>* theta = NULL,
   Image<float>* width = NULL,
   float width_scale = 1.0);

  void renderSegmentInImage
  (NeuronSegment* segment, Image<float>* img,
   Image<float>* theta = NULL, Image<float>* width = NULL,
   float min_width = 0,
   float width_scale = 1.0);

  //Returns a vector with the lengths of all the segments
  vector< double >  getAllEdgesLength();
  void getAllEdgesLength(NeuronSegment* segment, vector<double> &toRet);

  /** Elliminates duplicated points. (whose distance is lower than threshold) */
  void elliminateDuplicatedPoints(double threshold);
  void elliminateDuplicatedPoints(NeuronSegment* segment, double threshold);


  /** Calculates the average length of the edge in the neuron.*/
  void getEdgeDistance(NeuronSegment* segment, int& nEdges, double& distances);

  /** Calculates the average length of the edge in the neuron.*/
  void getEdgeVariance
  (NeuronSegment* segment, int& nEdges, double& distances, double& mean_edge_distance);

  GLUquadricObj* gcylinder;

  NeuronContour soma;

  /** Axones of the neuron.*/
  vector< NeuronSegment* > axon;

  /** Dendrites of the neuron.*/
  vector< NeuronSegment* > dendrites;

  /** Keeps a copy of all the points in the neuron. It is absolutely inneficient, but easier to code.*/
  vector< NeuronPoint >  allPointsVector;

  /** Stores the conversion matrix between the asc file units and micrometers. It is stored as a matrix for OpenGl, the order of the
   * elements can be found in: http://www.mevis.de/opengl/glLoadMatrix.html . It is loaded when an ascParser is called */
  vector< double > projectionMatrix;

  /** Stores the inverse of the projectionMatrix. I.e. to transform micrometers to neuron coordinates.*/
  vector< double > projectionMatrixInv;

  /** Map with the name of the points and the offset calculated with the cross correlation.*/
  map< std::string, vector< float > > correlationOffsetsNeuron;
  map< std::string, vector< float > > correlationOffsetsPixels;

  /** Converts a position in neuron coordinates to a position in micrometers according to the matrix
      of the neuron.It just multiplies the neuronCoords vector with the projection matrix*/
  void neuronToMicrometers(vector< float > neuronCoords, vector< float > &micromCoords);

  /** Converts a position in micrometers to neuron coordinates. */
  void micrometersToNeuron(vector< float > micromCoords, vector< float > &neuronCoords);

  /** Gets some points and edges to train EM on them */
  void toCloudOld
  (string points_file,
   string edges_file,
   float width_samples,
   Cube<uchar, ulong>* cube);

  void toCloudOld(std::ofstream& points,
               std::ofstream& edges,
               float width_sampled,
               NeuronSegment* segment,
               Cube<uchar, ulong>* cube,
               int& last_point_number);

  Cloud_P* toCloud(string cloudName,
               bool saveOrientation=false,
               bool saveType=false,
               Cube_P* cubeLimit = NULL);


  void toCloud(NeuronSegment* segment,
               Cloud_P* cloud,
               bool saveOrientation = false,
               bool saveType = false,
               Cube_P* cubeLimit = NULL);


  /** To parse the neuron.*/
  ascParser2* asc;

  /** Name of the neuron.*/
  string className(){
    return "Neuron";
  }

};


#endif /*NEURON_H_*/
