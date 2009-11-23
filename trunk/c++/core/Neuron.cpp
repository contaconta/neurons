#include "Neuron.h"


NeuronPoint::NeuronPoint()
{
  coords.reserve(4);
  coords[0] = 0;
  coords[1] = 0;
  coords[2] = 0;
  coords[3] = 0;
  noSenseNumber = 1;
  pointNumber = 1;
}

NeuronPoint::NeuronPoint(float x, float y, float z, float d, int noSenseNumber, int pointNumber, NeuronSegment* parent){
  coords.resize(4);
  coords[0] = x;
  coords[1] = y;
  coords[2] = z;
  coords[3] = d;
  this->noSenseNumber = noSenseNumber;
  this->pointNumber   = pointNumber;
  this->parent = parent;
}

NeuronPoint::NeuronPoint(vector< float > coords){
  if(coords.size()!=4){
    std::cout << "Error constructing the point, coords has " << coords.size() << " dimensions" << std::endl;
  }
  this->coords = coords;
}

NeuronPoint::~NeuronPoint()
{
}

void NeuronPoint::print()
{
  //printf("[%.2f,%.2f,%.2f,%.2f]\n", coords[0], coords[1], coords[2], coords[3]);
  std::cout << coords[0] << " " << coords[1] << " " << coords[2] << " " << coords[3] << std::endl;
}

NeuronColor::NeuronColor()
{
  coords.resize(3);
  coords[0] = 0;
  coords[1] = 0;
  coords[2] = 0;
}

NeuronColor::NeuronColor(float R, float G, float B){
  coords.resize(3);
  coords[0] = R;
  coords[1] = G;
  coords[2] = B;
}

NeuronColor::NeuronColor(vector< float > coords)
{
  if(coords.size()!=3){
    printf("Error in the size of the color vector\n");
    coords.resize(3);
  }
  this->coords = coords;

}

NeuronColor::~NeuronColor()
{
}

void NeuronColor::set(float R, float G, float B)
{
  coords[0] = R;
  coords[1] = G;
  coords[2] = B;
}

void NeuronColor::print()
{
  std::cout << "NeuronColor: " << std::endl;
  std::cout << "-red   : " << coords[0] << std::endl;
  std::cout << "-green : " << coords[1] << std::endl;
  std::cout << "-blue  : " << coords[2] << std::endl;
}

NeuronMarker::NeuronMarker()
{
  name = "";
  type = "";
  color.set(0,0,0);
}

NeuronMarker::NeuronMarker(string _name, string _type, NeuronColor _color)
{
  name = _name;
  type = _type;
  color = _color;
}

NeuronMarker::NeuronMarker(string name, string type, vector< NeuronPoint > points, NeuronColor color)
{
  this->name  = name;
  this->type  = type;
  this->color = color;
  this->points = points;
}

NeuronMarker::~NeuronMarker()
{
}

void NeuronMarker::print()
{
  std::cout << "Marker  : " << type << std::endl;
  std::cout << "  name  : " << name << std::endl;
  std::cout << "  points: " << points.size() << std::endl;
}


void NeuronMarker::draw()
{
  glColor3f(color.coords[0],
            color.coords[1],
            color.coords[2]);

  for(int i = 0; i < points.size(); i++)
    {
      glPushMatrix();
      glTranslatef(points[i].coords[0], points[i].coords[1], points[i].coords[2]);
      glutSolidSphere(0.5, 10, 10);
      glPopMatrix();

    }


}

NeuronSegment::NeuronSegment()
{
  this->parent = NULL;
  this->ending = "Incomplete";
  this->color.set(0,0,0);
  this->name = "";
  this->childs = vector<NeuronSegment*>();
}

NeuronSegment::NeuronSegment(NeuronSegment* parent, string ending, NeuronColor color)
{
  this->parent = parent;
  this->ending = ending;
  this->color  = color;
  this->name = "";
  this->childs = vector<NeuronSegment*>();
}

NeuronSegment::NeuronSegment(
                         NeuronSegment* parent,
                         vector< NeuronPoint > points,
                         vector< NeuronMarker > markers,
                         vector< NeuronSegment* > childs,
                         string ending,
                         NeuronColor color)
{
  this->parent = parent;
  this->points = points;
  this->markers = markers;
  this->childs = childs;
  this->ending = ending;
  this->color = color;
  this->name = "";
}

NeuronSegment::~NeuronSegment()
{
}

void NeuronSegment::print(int n)
{
//   string blanks = "";
//   for(int i = 0; i < n; i++ )
//     blanks += "  ";

//   std::cout << blanks << "Segment" << std::endl;
//   std::cout << blanks << "  childs  : " << childs.size() << std::endl;
//   std::cout << blanks << "  points  : " << points.size() << std::endl;
//   for(int i = 0; i < childs.size(); i++)
//     childs[i].print(n+1);
}

NeuronContour::NeuronContour()
{
  color.set(0,0,0);
  name = "";
}

NeuronContour::~NeuronContour()
{
}

void NeuronContour::print()
{
  std::cout << "Contour " << name << " with " << points.size() << " points" << std::endl;
  for(int i = 0; i < points.size(); i++)
    points[i].print();
}

Neuron::Neuron() : VisibleE()
{
  gcylinder = gluNewQuadric();
  allPointsVector.resize(0);
  projectionMatrix.resize(16);
  projectionMatrixInv.resize(16);
  //Creates the empty matrices
  for(int i = 0; i < 16; i++){
    projectionMatrix[i] = 0;
    projectionMatrixInv[i] = 0;
  }
  for(int i = 0; i < 16; i+=5){
    projectionMatrix[i] = 1;
    projectionMatrixInv[i] = 1;
  }
  //loadCorrelationOffsets();
}


Neuron::Neuron(string fileName) : VisibleE()
{
  
  gcylinder = gluNewQuadric();
  allPointsVector.resize(0);
  projectionMatrix.resize(16);
  projectionMatrixInv.resize(16);
  for(int i = 0; i < 16; i++){
    projectionMatrix[i] = 0;
    projectionMatrixInv[i] = 0;
  }
  for(int i = 0; i < 16; i+=5){
    projectionMatrix[i] = 1;
    projectionMatrixInv[i] = 1;
  }

  if(fileExists(fileName)){
    asc = new ascParser2(fileName);
    asc->parseFile(this);
  } else{
    asc = new ascParser2(fileName);
  }
}

Neuron::~Neuron()
{
}

void Neuron::print()
{
  std::cout << "--------- Printing soma --------------" << std::endl;
  soma.print();

  std::cout << "--------- Printing axons -------------" << std::endl;
  std::cout << "Number of axons: " << axon.size() << std::endl;
  for(int i = 0; i < axon.size(); i++)
    axon[i]->print();

  std::cout << "--------- Printing dendrites ---------" << std::endl;
  std::cout << "Number of dendrites: " << dendrites.size() << std::endl;
  for(int i = 0; i < dendrites.size(); i++)
    dendrites[i]->print();


}

void Neuron::drawSegment(NeuronSegment* segment, vector< float > root, bool includeOffset, float max_width)
{


  int lastPoint = 0;
  vector< float > xyz;
  vector< float > xyz0;
  vector< float > diff;

  //Draws cylinders in the segments
  glColor3f(segment->color.coords[0],
            segment->color.coords[1],
            segment->color.coords[2]);

  if ( (segment->color.coords[0] > .95) && (segment->color.coords[1] > .95) && (segment->color.coords[2] > .95) )
    glColor3f(0.0,0.0,1.0);

  for(int i = 0; i < segment->points.size(); i++)
    {

      //Gets the point name
      char pointName[1024];
      sprintf(pointName, "%s-P%02i", segment->name.c_str(), segment->points[i].pointNumber);
      vector< float > corrOffset;

      if(includeOffset){
        corrOffset = correlationOffsetsNeuron[pointName];
        corrOffset.push_back(0); //For the width of the offset
      }
      else
        for(int i = 0; i < 4; i++) corrOffset.push_back(0);

      xyz = addVectors(segment->points[i].coords,corrOffset);
      //xyz = segment->points[i].coords;

      char prevPointName[1024];

      if(i==0)
        {
          xyz0 = root;
          sprintf(prevPointName, "%s-P%02i", segment->name.c_str(), 0);
        }
      else
        {
          xyz0 = segment->points[i-1].coords;
          sprintf(prevPointName, "%s-P%02i", segment->name.c_str(), segment->points[i-1].pointNumber);
        }

      vector< float > corrOffset0;
      if(includeOffset){
        corrOffset0 = correlationOffsetsNeuron[prevPointName];
        corrOffset0.push_back(0);
      }
      else
        for(int i = 0; i < 4; i++) corrOffset0.push_back(0);

      xyz0 = addVectors(xyz0, corrOffset0);

      glPushMatrix();

      glTranslatef(xyz0[0], xyz0[1], xyz0[2]);

      diff = xyz;
      diff[0] -= xyz0[0];
      diff[1] -= xyz0[1];
      diff[2] -= xyz0[2];

      float segmLen =
    	diff[0]*diff[0] +
    	diff[1]*diff[1] +
    	diff[2]*diff[2];
      segmLen = sqrt(segmLen);

//       glColor3f(0.0,1.0,0.0);
      glRotatef(-180*acosf(diff[2]/segmLen)/3.14159,
                diff[1],-diff[0],0);
//       if((segment->points[i].coords[3] > 0.6) )
      gluCylinder(gcylinder,xyz[3],xyz[3],segmLen,10,10);
      glPopMatrix();
    }
  glFlush();

  for(int i = 0; i < segment->markers.size(); i++)
    {
      segment->markers[i].draw();
    }

  drawSpines(segment->spines);

  //Returns to the color of the segment
  glColor3f(segment->color.coords[0],
            segment->color.coords[1],
            segment->color.coords[2]);


  for(int i = 0; i < segment->childs.size(); i++)
    {
      drawSegment(segment->childs[i], xyz, includeOffset);

    }

}


void Neuron::drawSpines(vector< NeuronPoint>& spines)
{
  //glColor3f(1.0,0.0,0.0);
  for(int i = 0; i < spines.size(); i++)
    {
      //printf("Drawing spine at %f %f %f\n", spines[i].coords[0], spines[i].coords[1], spines[i].coords[2]);
      glPushMatrix();
      glTranslatef(spines[i].coords[0], spines[i].coords[1], spines[i].coords[2]);
      glutSolidSphere(spines[i].coords[3], 5, 5);
      glPopMatrix();
    }
}

void Neuron::drawSegmentAsLines(NeuronSegment* segment, vector< float > root, bool includeOffset)
{


  int lastPoint = 0;
  vector< float > xyz;
  vector< float > xyz0;
  vector< float > diff;

  //Draws cylinders in the segments
  glColor3f(segment->color.coords[0],
            segment->color.coords[1],
            segment->color.coords[2]);

  for(int i = 0; i < segment->points.size(); i++)
    {

      //Gets the point name
      char pointName[1024];
      sprintf(pointName, "%s-P%02i", segment->name.c_str(), segment->points[i].pointNumber);
      vector< float > corrOffset;

      xyz = segment->points[i].coords;

      char prevPointName[1024];

      if(i==0)
        {
          xyz0 = root;
          sprintf(prevPointName, "%s-P%02i", segment->name.c_str(), 0);
        }
      else
        {
          xyz0 = segment->points[i-1].coords;
          sprintf(prevPointName, "%s-P%02i", segment->name.c_str(), segment->points[i-1].pointNumber);
        }

      glEnable(GL_LINE_SMOOTH);
      glColor3f(140.0/255,40.0/255,40.0/255);
      glLineWidth(6.0);
      glBegin(GL_LINES);
      glVertex3f(xyz0[0],xyz0[1],xyz0[2]);
      glVertex3f(xyz[0],xyz[1],xyz[2]);
      glEnd();

      glColor3f(225.0/255,155.0/255,1.0);
      glLineWidth(2.0);
//       Edge<P>::draw();
      glBegin(GL_LINES);
      glVertex3f(xyz0[0],xyz0[1],xyz0[2]);
      glVertex3f(xyz[0],xyz[1],xyz[2]);
      glEnd();

      glLineWidth(1.0);

//       glPushMatrix();
//       glTranslatef(xyz[0], xyz[1], xyz[2]);
//       glutSolidSphere(segment->points[i].coords[3], 10, 10);
//       // glutWireSphere(1, 5, 5);
//       glPopMatrix();

    }


  for(int i = 0; i < segment->markers.size(); i++)
    {
      segment->markers[i].draw();
    }

  drawSpines(segment->spines);

  //Returns to the color of the segment
  glColor3f(segment->color.coords[0],
            segment->color.coords[1],
            segment->color.coords[2]);


  for(int i = 0; i < segment->childs.size(); i++)
    {
      drawSegmentAsLines(segment->childs[i], xyz, includeOffset);
    }

}


void Neuron::drawSegmentAsDots(NeuronSegment* segment, vector< float > root, bool includeOffset)
{


  int lastPoint = 0;
  vector< float > xyz;
  vector< float > xyz0;
  vector< float > diff;

  // Inverts the color of the segment to make it look nice
  glColor3f(1 - segment->color.coords[0],
  	    1 - segment->color.coords[1],
  	    1 - segment->color.coords[2]);

  for(int i = 0; i < segment->points.size(); i++)
    {

      //Gets the point name
      char pointName[1024];
      sprintf(pointName, "%s-P%02i", segment->name.c_str(), segment->points[i].pointNumber);
      vector< float > corrOffset;

      if(includeOffset){
        corrOffset = correlationOffsetsNeuron[pointName];
        corrOffset.push_back(0); //For the width of the offset
      }
      else
        for(int i = 0; i < 4; i++) corrOffset.push_back(0);

      xyz = addVectors(segment->points[i].coords,corrOffset);


      glPushMatrix();

      glTranslatef(xyz[0], xyz[1], xyz[2]);
      glutSolidSphere(0.1, 10, 10);

      glPopMatrix();
    }
  glFlush();

  for(int i = 0; i < segment->markers.size(); i++)
    {
      segment->markers[i].draw();
    }

  drawSpines(segment->spines);

  //Returns to the color of the segment
//   glColor3f(0.0,0.0,0.5);

  for(int i = 0; i < segment->childs.size(); i++)
    {
      drawSegmentAsDots(segment->childs[i], xyz, includeOffset);

    }

}

void Neuron::draw(){
  if(v_glList == 0){
    v_glList = glGenLists(1);
    glNewList(v_glList, GL_COMPILE);
//     drawInOpenGl(false, 1e6);
    drawInOpenGlAsLines(false);
    glEndList();
  }
  else{
    glCallList(v_glList);
  }
}


void Neuron::drawInOpenGl(bool includeCorrection, float max_width)
{

  //The GL Matrix setup should be done before calling this method
  setUpGlMatrices();

  if(includeCorrection)
    printf("!!! Drawing the neuron with the correction of the crossCorrelation\n");

  //Puts a red ball in the origin of the Neuron
  glColor3f(1,0,0);
  glutSolidSphere(1, 10, 10);

  //Draw the soma of the neuron
  //Draws the contour as lines
  glColor3f(soma.color.coords[0],
            soma.color.coords[1],
            soma.color.coords[2]);

  glBegin(GL_POLYGON);
//   glVertex3f(0,0,0);
  for(int i = 0; i < this->soma.points.size(); i++)
    {
      vector< float > xyz = this->soma.points[i].coords;
      glVertex3f(xyz[0],xyz[1],xyz[2]);
    }
  glEnd();

  //Draws balls in the points
  glColor3f(0,0,1);
  for(int i = 0; i < this->soma.points.size(); i++)
    {
      vector< float > xyz = this->soma.points[i].coords;
      glPushMatrix();
      glTranslatef(xyz[0], xyz[1], xyz[2]);
      glutSolidSphere(0.5, 10, 10);
      glPopMatrix();
    }

  //Draws the axon of the neuron
  for(int i = 0; i < this->axon.size(); i++)
    drawSegment(this->axon[i], this->axon[i]->root.coords, includeCorrection, max_width);

  //Draws the dendrites
  for(int i = 0; i < this->dendrites.size(); i++)
    {
      drawSegment(this->dendrites[i], this->dendrites[i]->points[0].coords, includeCorrection, max_width);
    }

  glPopMatrix();
}


void Neuron::drawInOpenGlAsLines(bool includeCorrection)
{

  setUpGlMatrices();

  //Puts a red ball in the origin of the Neuron
  glColor3f(1,0,0);
  glutSolidSphere(1, 10, 10);

  //Draw the soma of the neuron
  //Draws the contour as lines
  glColor3f(soma.color.coords[0],
            soma.color.coords[1],
            soma.color.coords[2]);

  glBegin(GL_POLYGON);
  for(int i = 0; i < this->soma.points.size(); i++)
    {
      vector< float > xyz = this->soma.points[i].coords;
      glVertex3f(xyz[0],xyz[1],xyz[2]);
    }
  glEnd();

  //Draws balls in the points
  if(0){
    glColor3f(0,0,1);
    for(int i = 0; i < this->soma.points.size(); i++)
      {
        vector< float > xyz = this->soma.points[i].coords;
        glPushMatrix();
        glTranslatef(xyz[0], xyz[1], xyz[2]);
        glutSolidSphere(0.5, 10, 10);
        glPopMatrix();
      }
  }
  //Draws the axon of the neuron
  for(int i = 0; i < this->axon.size(); i++)
    drawSegmentAsLines(this->axon[i], this->axon[i]->root.coords, includeCorrection);

  //Draws the dendrites
  for(int i = 0; i < this->dendrites.size(); i++)
    {
      drawSegmentAsLines(this->dendrites[i], this->dendrites[i]->points[0].coords, includeCorrection);
    }
  glPopMatrix();
}


void Neuron::drawInOpenGlAsDots(bool includeCorrection)
{

  //The GL Matrix setup should be done before calling this method
  setUpGlMatrices();

  if(includeCorrection)
    printf("!!! Drawing the neuron as dots with the correction of the crossCorrelation\n");

  //Puts a red ball in the origin of the Neuron
  glColor3f(1,0,0);
  glutSolidSphere(1, 10, 10);

  //Draw the soma of the neuron
  //Draws the contour as lines
  glColor3f(soma.color.coords[0],
            soma.color.coords[1],
            soma.color.coords[2]);

  glBegin(GL_POLYGON);
  for(int i = 0; i < this->soma.points.size(); i++)
    {
      vector< float > xyz = this->soma.points[i].coords;
      glVertex3f(xyz[0],xyz[1],xyz[2]);
    }
  glEnd();

  //Draws balls in the points
  glColor3f(0,0,1);
  for(int i = 0; i < this->soma.points.size(); i++)
    {
      vector< float > xyz = this->soma.points[i].coords;
      glPushMatrix();
      glTranslatef(xyz[0], xyz[1], xyz[2]);
      glutSolidSphere(0.5, 10, 10);
      glPopMatrix();
    }

  //Draws the axon of the neuron
  for(int i = 0; i < this->axon.size(); i++)
    drawSegmentAsDots(this->axon[i], this->axon[i]->root.coords, includeCorrection);

  //Draws the dendrites
  for(int i = 0; i < this->dendrites.size(); i++)
    {
      drawSegmentAsDots(this->dendrites[i], this->dendrites[i]->points[0].coords, includeCorrection);
    }

  glPopMatrix();
}



void Neuron::drawInOpenGlWithCorrectionInMicrometers
(float neuron_disp3DX,
 float neuron_disp3DY,
 float neuron_disp3DZ,
 float neuron_zoom3DX,
 float neuron_zoom3DY,
 float neuron_zoom3DZ,
 float neuron_rot3DX,
 float neuron_rot3DY,
 float neuron_rot3DZ )
{
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();

  //Loads the matrix used to adjust the neuron in the first instance;
  double matrixForNeuron[16];
  for(int i = 0; i < 16; i++)
    {
      matrixForNeuron[i] = this->projectionMatrix[i];
    }

  glMultMatrixd(matrixForNeuron);

  //Put the fine tune hand parameters (passed by the drawing function)
  glTranslatef(neuron_disp3DX,neuron_disp3DY,-neuron_disp3DZ);
  glScalef(neuron_zoom3DX, neuron_zoom3DY, neuron_zoom3DZ);
  glRotatef(neuron_rot3DX,1,0,0);
  glRotatef(neuron_rot3DY,0,1,0);
  glRotatef(neuron_rot3DZ,0,0,1);


  //Puts a red ball in the origin of the Neuron
  glColor3f(1,0,0);
  glutSolidSphere(1, 10, 10);

  //Draw the soma of the neuron
  //Draws the contour as lines
  glColor3f(soma.color.coords[0],
            soma.color.coords[1],
            soma.color.coords[2]);

  glBegin(GL_POLYGON);
  glVertex3f(0,0,0);
  for(int i = 0; i < this->soma.points.size(); i++)
    {
      vector< float > xyz = this->soma.points[i].coords;
      glVertex3f(xyz[0],xyz[1],xyz[2]);
    }
  glEnd();

  //Draws balls in the points
  glColor3f(0,0,1);
  for(int i = 0; i < this->soma.points.size(); i++)
    {
      vector< float > xyz = this->soma.points[i].coords;
      glPushMatrix();
      glTranslatef(xyz[0], xyz[1], xyz[2]);
      glutSolidSphere(0.5, 10, 10);
      glPopMatrix();
    }

  // Returns to the matrix configuration given to the method.
  glPopMatrix();

  //Draws the axon of the neuron
  for(int i = 0; i < this->axon.size(); i++)
    drawSegmentWithCorrectionInMicrometers(this->axon[i], this->axon[i]->root.coords,
                                           neuron_disp3DX,  neuron_disp3DY, neuron_disp3DZ,
                                           neuron_zoom3DX,  neuron_zoom3DY, neuron_zoom3DZ,
                                           neuron_rot3DX,   neuron_rot3DY,  neuron_rot3DZ
					   );

  //Draws the dendrites
  for(int i = 0; i < this->dendrites.size(); i++)
    {
      drawSegmentWithCorrectionInMicrometers(this->dendrites[i], this->dendrites[i]->points[0].coords,
                                             neuron_disp3DX,  neuron_disp3DY, neuron_disp3DZ,
                                             neuron_zoom3DX,  neuron_zoom3DY, neuron_zoom3DZ,
                                             neuron_rot3DX,   neuron_rot3DY,  neuron_rot3DZ );
    }
}


void Neuron::drawSegmentWithCorrectionInMicrometers(NeuronSegment* segment,vector< float > root,
                                                    float neuron_disp3DX,  float neuron_disp3DY, float neuron_disp3DZ ,
                                                    float neuron_zoom3DX,  float neuron_zoom3DY, float neuron_zoom3DZ,
                                                    float neuron_rot3DX,   float neuron_rot3DY,  float neuron_rot3DZ)
{

  int lastPoint = 0;
  vector< float > xyz;

  // Inverts the color of the segment to make it look nice
  glColor3f(1 - segment->color.coords[0],
  	    1 - segment->color.coords[1],
  	    1 - segment->color.coords[2]);

  // Puts the matrix for the neuron in general
  glPushMatrix();

  double matrixForNeuron[16];
  for(int i = 0; i < 16; i++)
    {
      matrixForNeuron[i] = this->projectionMatrix[i];
    }


  //Put the fine tune hand parameters (passed by the drawing function)
  //glTranslatef(neuron_disp3DX,neuron_disp3DY,-neuron_disp3DZ);
  //glScalef(neuron_zoom3DX, neuron_zoom3DY, neuron_zoom3DZ);
  //glRotatef(neuron_rot3DX,1,0,0);
  //glRotatef(neuron_rot3DY,0,1,0);
  //glRotatef(neuron_rot3DZ,0,0,1);


  for(int i = 0; i < segment->points.size(); i++)
    {

      glPushMatrix(); // Each points needs its matrix

      //Gets the point name
      char pointName[1024];
      sprintf(pointName, "%s-P%02i", segment->name.c_str(), segment->points[i].pointNumber);

      // Gets the offset
      vector< float > corrOffset;
      corrOffset = correlationOffsetsPixels[pointName];

      // FIXME The translation between pixels and micrometers (coords in the opengl) is hardCoded. And review the change of coordinates between Matlab and the Neuron.
      glTranslatef(corrOffset[1]*177.04/1288, -corrOffset[0]*132.85/966, corrOffset[2]*0.8);


      // Generates a matrix that diplaces the point according to the offset in the cross_correlation
      glMultMatrixd(matrixForNeuron);
      xyz = segment->points[i].coords;

      glTranslatef(xyz[0], xyz[1], xyz[2]);
      glutSolidSphere(0.5, 10, 10);

      glPopMatrix(); // The matrix for the point
    }
  glFlush();

  glPopMatrix(); // The general Matrix

  for(int i = 0; i < segment->markers.size(); i++)
    {
      segment->markers[i].draw();
    }

  drawSpines(segment->spines);

  //Returns to the color of the segment
//   glColor3f(0.0,0.0,0.5);

  for(int i = 0; i < segment->childs.size(); i++)
    {
      drawSegmentWithCorrectionInMicrometers(segment->childs[i], xyz,
                                             neuron_disp3DX,  neuron_disp3DY, neuron_disp3DZ,
                                             neuron_zoom3DX,  neuron_zoom3DY, neuron_zoom3DZ,
                                             neuron_rot3DX,   neuron_rot3DY,  neuron_rot3DZ
                                             );

    }

}


void Neuron::setUpGlMatrices()
{
  GLdouble matrix[16];
  for(int i = 0; i < 16; i++)
    matrix[i] = projectionMatrix[i];
  glPushMatrix();
  glMultMatrixd(matrix);
}


void Neuron::getMaxPoint(NeuronSegment* segment, double* maxX, double* maxY, double* maxZ
                         , double* minX, double* minY, double* minZ,
                         int* nSegments)
{
  for(int i = 0; i < segment->points.size(); i++)
    {
      if(segment->points[i].coords[0] > *maxX)
        *maxX = segment->points[i].coords[0];
      if(segment->points[i].coords[1] > *maxY)
        *maxY = segment->points[i].coords[1];
      if(segment->points[i].coords[2] > *maxZ)
        *maxZ = segment->points[i].coords[2];
      if(segment->points[i].coords[0] < *minX)
        *minX = segment->points[i].coords[0];
      if(segment->points[i].coords[1] < *minY)
        *minY = segment->points[i].coords[1];
      if(segment->points[i].coords[2] < *minZ)
        *minZ = segment->points[i].coords[2];
    }

  *nSegments = *nSegments + 1;

  for(int i = 0; i < segment->childs.size(); i++)
    {
      getMaxPoint(segment->childs[i],maxX,maxY,maxZ, minX, minY, minZ, nSegments);
    }

}

void Neuron::printStatistics()
{
  double maxX = 0;
  double maxY = 0;
  double maxZ = 0;
  double minX = 0;
  double minY = 0;
  double minZ = 0;
  int nSegments = 0;

  if(axon.size()>0){
    getMaxPoint(axon[0], &maxX, &maxY, &maxZ, &minX, &minY, &minZ, &nSegments);
    printf("Axon has   %i  segments\n", nSegments);
  }
  printf("Neuron has %i  dendrites\n", (int)dendrites.size());


  for(int i = 0; i < dendrites.size(); i++)
    {
      nSegments = 0;
      getMaxPoint(dendrites[i], &maxX, &maxY, &maxZ, &minX, &minY, &minZ, &nSegments);
      printf("dendrites[%i] has %i ramifications\n", i, nSegments);
    }

  printf("Coord   Max    Min   \n");
  printf("X       %.3f    %.3f   \n", maxX, minX);
  printf("Y       %.3f    %.3f   \n", maxY, minY);
  printf("Z       %.3f    %.3f   \n", maxZ, minZ);

  vector< float > widths;
  vector< int >   occurrences;

  for(int i = 0; i < axon.size(); i++)
    getWidthDistribution(widths, occurrences, axon[i]);

  for(int i = 0; i < dendrites.size(); i++)
    getWidthDistribution(widths, occurrences, dendrites[i]);

  //Prints the widths shorted
  int nPoints = 0;
  for(int i = 0; i < occurrences.size(); i++)
      nPoints += occurrences[i];
  printf("The neuron has %i points\n", nPoints);

  printf(" ==== WIDTH TABLE ====\n");
  printf(" width    occurrences\n");
  while(widths.size() > 0)
    {
      //Gets the maxWidth;
      int maxWidthIdx = 0;
      float maxWidth = 0;
      for(int j = 0; j < widths.size(); j++)
        {
          if(widths[j] > maxWidth){
            maxWidth = widths[j];
            maxWidthIdx = j;
          }
        }
      printf(" %03.03f    %i\n", widths[maxWidthIdx], occurrences[maxWidthIdx]);
      widths.erase(widths.begin() + maxWidthIdx);
      occurrences.erase(occurrences.begin() + maxWidthIdx);
    }

  //From the reconstruction tree. Gets the distribution of the click length.
  int nEdges = 0;
  double distance = 0;
  for(int i = 0; i < axon.size(); i++)
    getEdgeDistance(axon[i], nEdges, distance);
  for(int i = 0; i < dendrites.size(); i++)
    getEdgeDistance(dendrites[i], nEdges, distance);

  double mean_edge = distance / nEdges;
  nEdges = 0;
  double variance = 0;
  for(int i = 0; i < axon.size(); i++)
    getEdgeVariance(axon[i], nEdges, variance, mean_edge);
  for(int i = 0; i < dendrites.size(); i++)
    getEdgeVariance(dendrites[i], nEdges, variance, mean_edge);
  variance = variance / nEdges;


  printf("The average edge distance is %f with variance %f\n", mean_edge, variance);

}

vector< double > Neuron::getAllEdgesLength(){
  vector< double > toRet;
  for(int i = 0; i < axon.size(); i++)
    getAllEdgesLength(axon[i], toRet);
  for(int i = 0; i < dendrites.size(); i++)
    getAllEdgesLength(dendrites[i], toRet);
  return toRet;
}

void Neuron::getAllEdgesLength(NeuronSegment* segment, vector<double> &toRet){
  for(int i = 0; i < segment->childs.size(); i++)
    getAllEdgesLength(segment->childs[i], toRet);

  vector<float> p1;
  vector<float> p2;
  for(int i = 1; i < segment->points.size(); i++){
    neuronToMicrometers(segment->points[i-1].coords, p1);
    neuronToMicrometers(segment->points[i  ].coords, p2);
    toRet.push_back(sqrt( (p1[0]-p2[0])*(p1[0]-p2[0]) +
                          (p1[1]-p2[1])*(p1[1]-p2[1]) +
                          (p1[2]-p2[2])*(p1[2]-p2[2])));
  }
}

void Neuron::elliminateDuplicatedPoints(double threshold){
  for(int i = 0; i < axon.size(); i++)
    elliminateDuplicatedPoints(axon[i], threshold);
  for(int i = 0; i < dendrites.size(); i++)
    elliminateDuplicatedPoints(dendrites[i], threshold);
}

void Neuron::elliminateDuplicatedPoints(NeuronSegment* segment, double threshold){
  for(int i = 0; i < segment->childs.size(); i++)
    elliminateDuplicatedPoints(segment->childs[i], threshold);

  vector<float> p1;
  vector<float> p2;
  vector< NeuronPoint > newPoints;
  newPoints.push_back(
                      NeuronPoint(segment->points[0].coords[0],
                                  segment->points[0].coords[1],
                                  segment->points[0].coords[2],
                                  segment->points[0].coords[3],
                                  segment->points[0].noSenseNumber,
                                  segment->points[0].pointNumber
                                  ));


  double distance;
  for(int i = 1; i < segment->points.size(); i++){
    neuronToMicrometers(segment->points[i-1].coords, p1);
    neuronToMicrometers(segment->points[i  ].coords, p2);
    distance = sqrt( (p1[0]-p2[0])*(p1[0]-p2[0]) +
                     (p1[1]-p2[1])*(p1[1]-p2[1]) +
                     (p1[2]-p2[2])*(p1[2]-p2[2]));
    if(distance > threshold){
      newPoints.push_back(
                          NeuronPoint(segment->points[i].coords[0],
                                      segment->points[i].coords[1],
                                      segment->points[i].coords[2],
                                      segment->points[0].coords[3],
                                      segment->points[i].noSenseNumber,
                                      segment->points[i].pointNumber
                                      )
                          );
    }
  }
  segment->points = newPoints;
}


void Neuron::getEdgeDistance(NeuronSegment* segment, int& nEdges, double& distances)
{
  for(int i = 0; i < segment->childs.size(); i++)
    getEdgeDistance(segment->childs[i], nEdges, distances);

  vector<float> p1;
  vector<float> p2;

  for(int i = 1; i < segment->points.size(); i++){
    neuronToMicrometers(segment->points[i-1].coords, p1);
    neuronToMicrometers(segment->points[i  ].coords, p2);
    distances += sqrt( (p1[0]-p2[0])*(p1[0]-p2[0]) +
                       (p1[1]-p2[1])*(p1[1]-p2[1]) +
                       (p1[2]-p2[2])*(p1[2]-p2[2]));
    nEdges++;
  }
}

void Neuron::getEdgeVariance(NeuronSegment* segment, int& nEdges, double& variance, double& mean_edge_distance)
{
  for(int i = 0; i < segment->childs.size(); i++)
    getEdgeVariance(segment->childs[i], nEdges, variance, mean_edge_distance);

  vector<float> p1;
  vector<float> p2;

  for(int i = 1; i < segment->points.size(); i++){
    neuronToMicrometers(segment->points[i-1].coords, p1);
    neuronToMicrometers(segment->points[i  ].coords, p2);
    variance +=
        pow(sqrt((p1[0]-p2[0])*(p1[0]-p2[0]) +
              (p1[1]-p2[1])*(p1[1]-p2[1]) +
              (p1[2]-p2[2])*(p1[2]-p2[2]))
            - mean_edge_distance, 2.0);
    nEdges++;
  }
}


void Neuron::getWidthDistribution(vector<float>& widths, vector<int>& occurrences, NeuronSegment* segment)
{
  for(int i = 0; i < segment->points.size(); i++)
    {
      bool pointAlreadyInWidths = false;
      for(int j = 0; j < widths.size(); j++)
        {
          if( segment->points[i].coords[3] == widths[j] )
            {
              occurrences[j]++;
              pointAlreadyInWidths = true;
              break;
            }
        }
      if(!pointAlreadyInWidths)
        {
          widths.push_back(segment->points[i].coords[3]);
          occurrences.push_back(1);
        }
    }

  for(int i = 0; i < segment->childs.size(); i++)
    {
      getWidthDistribution(widths, occurrences, segment->childs[i]);
    }
}


NeuronSegment* Neuron::findClosestAxonOrDendrite(double x, double y, double z)
{
  double minDist;
  NeuronSegment* bestSeg;

  if(axon.size()>0)
    {
      minDist = distancePointSegmentWithChilds(x,y,z,axon[0]);
      bestSeg = axon[0];
    }
  else if( dendrites.size()>0)
    {
      minDist = distancePointSegmentWithChilds(x,y,z,dendrites[0]);
      bestSeg = dendrites[0];
    }
  else
    return NULL;



  for(int i = 0; i < axon.size(); i++)
    {
      if(distancePointSegmentWithChilds(x,y,z,axon[i]) < minDist)
        {
          minDist = distancePointSegmentWithChilds(x,y,z,axon[i]);
          bestSeg = axon[i];
        }

    }

  for(int i = 0; i < dendrites.size(); i++)
    {
      if(distancePointSegmentWithChilds(x,y,z,dendrites[i]) < minDist)
        {
          minDist = distancePointSegmentWithChilds(x,y,z,dendrites[i]);
          bestSeg = dendrites[i];
        }
    }

  return bestSeg;
}

double Neuron::distancePointPoint(double x, double y, double z, NeuronPoint& point)
{
  double dist =  sqrt( (point.coords[0]-x)*(point.coords[0]-x) +
                       (point.coords[1]-y)*(point.coords[1]-y) +
                       (point.coords[2]-z)*(point.coords[2]-z) );

  return dist;
}


double Neuron::distancePointSegmentWithChilds(double x, double y, double z, NeuronSegment* pepe)
{
  double dinMin = 100000000;

  for(int i = 0; i < pepe->points.size(); i++)
    dinMin = (distancePointPoint(x,y,z,pepe->points[i])>dinMin) ?
      dinMin :
      distancePointPoint(x,y,z,pepe->points[i]);

  for(int i = 0; i < pepe->childs.size(); i++)
    {
      double dps = distancePointSegmentWithChilds(x,y,z,pepe->childs[i]);
      dinMin = (dps > dinMin) ?
        dinMin :
        dps;
    }

  return dinMin;
}


double Neuron::distancePointSegment(double x, double y, double z, NeuronSegment* pepe)
{
  double dinMin = 100000000;

  for(int i = 0; i < pepe->points.size(); i++)
    dinMin = (distancePointPoint(x,y,z,pepe->points[i])>dinMin) ?
      dinMin :
      distancePointPoint(x,y,z,pepe->points[i]);

  return dinMin;
}

NeuronSegment* Neuron::findClosestSubsegment(double x, double y, double z, NeuronSegment* segment, double* distance)
{
  double minDistance = distancePointSegment(x,y,z,segment);
  NeuronSegment* minSegment = segment;

  double distChild = minDistance;
  NeuronSegment*  childClose;

  for(int i = 0; i < segment->childs.size(); i++)
    {
      childClose = findClosestSubsegment(x,y,z,segment->childs[i], &distChild);

      if(distChild < minDistance)
        {
          minSegment   = childClose;
          minDistance  = distChild;
        }

    }

  *distance = minDistance;

  return minSegment;
}


NeuronSegment* Neuron::findClosestSegment(double x, double y, double z)
{
  double minDist = 1000000;
  NeuronSegment* minSegment = NULL;

  double distance = 1000000;
  NeuronSegment* currSegment;

  for(int i = 0; i < axon.size(); i++){
    currSegment = findClosestSubsegment(x,y,z,axon[i], &distance);
    if(distance < minDist)
      {
        minSegment = currSegment;
        minDist = distance;
      }
  }

  for(int i = 0; i < dendrites.size(); i++){
    currSegment = findClosestSubsegment(x,y,z,dendrites[i], &distance);
    if(distance < minDist)
      {
        minSegment = currSegment;
        minDist = distance;
      }
  }


  return minSegment;
}


NeuronPoint* Neuron::findClosestPoint(double x, double y, double z)
{

  NeuronSegment* segment = findClosestSegment(x,y,z);
  if(segment == NULL)
    return NULL;
  int nClosePoint = 0;
  double minDistance = 0;
  if(segment->points.size() > 0)
    minDistance = distancePointPoint(x,y,z,segment->points[0]);

  //std::cout << "Segment has " << segment->points.size() << " points." << std::endl;

  for(int i = 0; i < segment->points.size(); i++)
    {

      if(distancePointPoint(x,y,z,segment->points[i]) < minDistance)
        {
          minDistance = distancePointPoint(x,y,z,segment->points[i]);
          nClosePoint = i;
        }
    }

  return &segment->points[nClosePoint];
}

//FIXME
int Neuron::findIndexOfClosestPointInSegment(double x, double y, double z, NeuronSegment* segment)
{
  double min_dist = 1000000000;
  int points_index = -1;

  for( int i = 0; i < segment->points.size(); i++)
    {
      if(distancePointPoint(x,y,z,segment->points[i]) < min_dist)
        {
          min_dist = distancePointPoint(x,y,z,segment->points[i]);
          points_index = i;
        }
    }

  return points_index;
}


NeuronPoint* Neuron::findNextPoint(double x, double y, double z)
{
  NeuronSegment* segmentThatBelongs = findClosestSegment(x,y,z);

  int point_index =  findIndexOfClosestPointInSegment(x,y,z,segmentThatBelongs);

  if(point_index+1 < segmentThatBelongs->points.size())
    return &segmentThatBelongs->points[point_index+1];

  else if(segmentThatBelongs->childs.size() != 0)
    return &segmentThatBelongs->childs[0]->points[0];

  else
    {
      NeuronPoint* pepe = new NeuronPoint(x,y,z,0);
      return pepe;
    }
}


/** FIXME Implement it before calling it */
NeuronPoint* getRandomPoint()
{
  printf("getRandomPoint is not implemented!!!!\n");
  return NULL;
}



void Neuron::loadCorrelationOffsets(string path)
{
  string path_names = path + "pointsNames.txt";
  std::ifstream names(path_names.c_str());
  if(!names.good())
    {
      printf("Neuron::loadCorrelationOffsets can not find the file: %s\n",path_names.c_str());
      return;
    }

  string path_offsets = path + "pointsOffsetCrossCorrelationNeuron.txt";

  std::ifstream offsets(path_offsets.c_str());
  if(!offsets.good())
    {
      printf("Neuron::loadCorrelationOffsets can not find the file: %s\n",path_offsets.c_str());
      return;
    }

  string path_offsets_pixels = path + "pointsOffsetCrossCorrelationPixels.txt";

  std::ifstream offsets_pixels(path_offsets_pixels.c_str());
  if(!offsets_pixels.good())
    {
      printf("Neuron::loadCorrelationOffsets can not find the file: %s\n",path_offsets_pixels.c_str());
      return;
    }


  while(!(names.eof() || offsets.eof()))
    {
      char name_b[1024];
      names.getline(name_b,1024);
      string name = name_b;
      vector< float > offset;
      vector< float > offset_pixels;
      float dummy;
      for(int i = 0; i < 3; i++)
	{
	  offsets >> dummy;
	  offset.push_back(dummy);
	  offsets_pixels >> dummy;
	  offset_pixels.push_back(dummy);
	}
      correlationOffsetsNeuron.insert( make_pair( name, offset));
      correlationOffsetsPixels.insert( make_pair( name, offset_pixels));
    }

  names.close();
  offsets.close();
}


void Neuron::applyRecursiveOffset
(NeuronSegment* closestS, int pointNumber,
 double diffx, double diffy, double diffz)
{
  // The numbering of the points is 1-based
  for(int i = pointNumber-1; i < closestS->points.size(); i++)
    {
      closestS->points[i].coords[0] += diffx;
      closestS->points[i].coords[1] += diffy;
      closestS->points[i].coords[2] += diffz;
    }

  for(int k = 0; k < closestS->spines.size(); k++)
    {
      if(closestS->spines[k].pointNumber >= pointNumber)
        {
          closestS->spines[k].coords[0] += diffx;
          closestS->spines[k].coords[1] += diffy;
          closestS->spines[k].coords[2] += diffz;
        }
    }

  for(int i = 0; i < closestS->childs.size(); i++)
    applyRecursiveOffset(closestS->childs[i], 1, diffx, diffy, diffz);

}

void Neuron::changePointPosition
  (NeuronSegment* segment,
   int pointNumber,
   double posx,
   double posy,
   double posz)
{
  segment->points[pointNumber-1].coords[0] = posx;
  segment->points[pointNumber-1].coords[1] = posy;
  segment->points[pointNumber-1].coords[2] = posz;
}



void Neuron::neuronToMicrometers(vector< float > neuronCoords, vector< float > &micromCoords)
{
  if(projectionMatrix.size() == 0)
    return;

  micromCoords.resize(4);

  micromCoords[0] =
    neuronCoords[0]*projectionMatrix[0] +
    neuronCoords[1]*projectionMatrix[4] +
    neuronCoords[2]*(projectionMatrix[8]) +
    projectionMatrix[12];

  micromCoords[1] =
    neuronCoords[0]*projectionMatrix[1] +
    neuronCoords[1]*projectionMatrix[5] +
    neuronCoords[2]*(projectionMatrix[9]) +
    projectionMatrix[13];

  micromCoords[2] =
    neuronCoords[0]*projectionMatrix[2] +
    neuronCoords[1]*projectionMatrix[6] +
    neuronCoords[2]*(projectionMatrix[10]) +
    projectionMatrix[14];

  micromCoords[3] = neuronCoords[3];

}

void Neuron::micrometersToNeuron(vector< float > micromCoords, vector< float > &neuronCoords)
{
  /** Why here we do not need any - in the z coordinate is a mystery */

  if(projectionMatrix.size() == 0)
    return;

  neuronCoords.resize(4);

  neuronCoords[0] =
    micromCoords[0]*projectionMatrixInv[0] +
    micromCoords[1]*projectionMatrixInv[4] +
    micromCoords[2]*(projectionMatrixInv[8]) +
    projectionMatrixInv[12];

  neuronCoords[1] =
    micromCoords[0]*projectionMatrixInv[1] +
    micromCoords[1]*projectionMatrixInv[5] +
    micromCoords[2]*(projectionMatrixInv[9]) +
    projectionMatrixInv[13];

  neuronCoords[2] =
    micromCoords[0]*projectionMatrixInv[2] +
    micromCoords[1]*projectionMatrixInv[6] +
    micromCoords[2]*(projectionMatrixInv[10]) +
    projectionMatrixInv[14];

  neuronCoords[3] = micromCoords[3];
}


vector< NeuronPoint > Neuron::getPointsWithWidth(float minWidth, float maxWidth)
{
  vector< NeuronPoint > final_out;
  for(int i = 0; i < this->axon.size(); i++)
    {
      vector< NeuronPoint > tmp;
      getPointsWithWidthRecursively(minWidth, maxWidth, this->axon[i], tmp);
      for(int j = 0; j < tmp.size(); j++)
        final_out.push_back(tmp[j]);
    }

  for(int i = 0; i < this->dendrites.size(); i++)
    {
      vector< NeuronPoint > tmp;
      getPointsWithWidthRecursively(minWidth, maxWidth, this->dendrites[i], tmp);
      for(int j = 0; j < tmp.size(); j++)
        final_out.push_back(tmp[j]);
    }

  return final_out;

}

void Neuron::getPointsWithWidthRecursively(float minWidth, float maxWidth, NeuronSegment* segment, vector< NeuronPoint >& points)
{
  for(int i = 0; i < segment->points.size(); i++)
    {
      if( (segment->points[i].coords[3] >= minWidth) &&
          (segment->points[i].coords[3] <= maxWidth) )
        points.push_back(segment->points[i]);
    }

  for(int i = 0; i < segment->childs.size(); i++)
    getPointsWithWidthRecursively(minWidth, maxWidth, segment->childs[i], points);
}


void Neuron::save(string filename)
{
  asc->saveNeuron(this, filename);
}


/** Draws the dendrite in an image.*/
void Neuron::renderInImage(Image<float>* img,
                           Image<float>* theta,
                           Image<float>* width,
                           float min_width,
                           float width_scale)
{
  img->put_all(255);
  printf("  going for segments [");
  for(int i = 0; i < axon.size(); i++){
    renderSegmentInImage(axon[i], img, theta, width, min_width, width_scale);
  }
  for(int i = 0; i < dendrites.size(); i++){
    renderSegmentInImage(dendrites[i], img, theta, width, min_width, width_scale);
    printf("#"); fflush(stdout);
  }
  printf("  ]\n");
}

void Neuron::renderSegmentInImage
(NeuronSegment* segment, Image<float>* img,
 Image<float>* theta, Image<float>* width,
 float min_width,float width_scale)
{
  if(segment->parent != NULL){
    NeuronPoint* p1 = &segment->parent->points[segment->parent->points.size()-1];
    NeuronPoint* p2 = &segment->points[0];
    if((p1->coords[3] > min_width) &&
       (p2->coords[3] > min_width) )
      renderEdgeInImage(p1, p2, img, theta, width, width_scale);
  }

  for(int i = 1; i < segment->points.size(); i++){
    if((segment->points[i-1].coords[3] > min_width) &&
       (segment->points[i].coords[3]   > min_width) )
      renderEdgeInImage(&segment->points[i-1], &segment->points[i], img, theta, width, width_scale);
  }
  for(int i = 0; i < segment->childs.size(); i++)
    renderSegmentInImage(segment->childs[i], img, theta, width, min_width, width_scale);

}


/**Draws the segment between P1 and P2 in the image with radius width*/
void Neuron::renderEdgeInImage
(NeuronPoint* _p1, NeuronPoint* _p2,
 Image<float>* img,
 Image<float>* theta,
 Image<float>* width_img,
 float width_scale)
{
  //This is a hack to convert the rendering from neuron coordinates to micrometers
  //We will create the points p1 and p2 with the coordinates of _p1 and _p2 in global
  // micrometers and not in neuron ones.
  NeuronPoint* p1 = new NeuronPoint();
  NeuronPoint* p2 = new NeuronPoint();

  this->neuronToMicrometers(_p1->coords, p1->coords);
  this->neuronToMicrometers(_p2->coords, p2->coords);

  double seglen = sqrt(
    (p1->coords[0]-p2->coords[0])*(p1->coords[0]-p2->coords[0]) +
    (p1->coords[1]-p2->coords[1])*(p1->coords[1]-p2->coords[1]) +
    (p1->coords[2]-p2->coords[2])*(p1->coords[2]-p2->coords[2]) );

  double width_microm = (p1->coords[3] + p2->coords[3])/2;
  double radius = 2*sqrt((seglen/2)*(seglen/2) + (width_microm/2)*(width_microm/2));

  //Calculates the medium point
  vector< float > micrometers(3);
  micrometers[0] = (p1->coords[0]+p2->coords[0])/2;
  micrometers[1] = (p1->coords[1]+p2->coords[1])/2;
  micrometers[2] = (p1->coords[2]+p2->coords[2])/2;
  vector< int > indexes_orig(3);
  img->micrometersToIndexes(micrometers, indexes_orig);

  vector<int> p1idx(3);
  img->micrometersToIndexes(p1->coords, p1idx);
  vector<int> p2idx(3);
  img->micrometersToIndexes(p2->coords, p2idx);

  double dist_calc = 0;
  vector<int> p2p1(3);
  p2p1[0] = p2idx[0] - p1idx[0];
  p2p1[1] = p2idx[1] - p1idx[1];
  p2p1[2] = p2idx[2] - p1idx[2];
  float p2p1mod =
    p2p1[0]*p2p1[0] +
    p2p1[1]*p2p1[1] +
    p2p1[2]*p2p1[2];
  vector<int> p1p0(3);
  vector<int> p2p0(3);
  int p1p0mod = 0;
  int dot_p1p0p2p1;
  float p2p1length_2 = sqrt(p2p1mod)/2;
  float d_p_indexes_origin;
  //We assume that voxelWidth aprox voxelheight aprox voxelDepth = 0.8
//   radius = (int)(radius/0.8);
//   float width = (width_microm / 0.8);
  float width = width_microm*width_scale;
  for(int y = (int)max(0.0, indexes_orig[1]-radius); 
      y < min(double(img->height), indexes_orig[1]+radius);
      y++)
    for(int x = (int)max(0.0, indexes_orig[0]-radius); 
        x < min(double(img->width), indexes_orig[0]+radius);
        x++)
              {
                //Calculates the distance between the point [x,y,z] and
                // the edge between p1 and p2
                p1p0[0] = p1idx[0]-x;
                p1p0[1] = p1idx[1]-y;
                p1p0[2] = 0;
                p1p0mod = p1p0[0]*p1p0[0] +
                          p1p0[1]*p1p0[1] +
                          p1p0[2]*p1p0[2];
                p2p0[0] = p2idx[0]-x;
                p2p0[1] = p2idx[1]-y;
                p2p0[2] = 0;
                dot_p1p0p2p1 = p1p0[0]*p2p1[0] + p1p0[1]*p2p1[1] + p1p0[2]*p2p1[2];

                dist_calc = float(p1p0mod*p2p1mod - dot_p1p0p2p1*dot_p1p0p2p1)/p2p1mod;

                // Calculates the distance from the point to the middle of the edge
                d_p_indexes_origin = sqrt( (double)(x-indexes_orig[0]) * (x-indexes_orig[0]) + 
                                           (y-indexes_orig[1]) * (y-indexes_orig[1]) );
                if((dist_calc <= width) && (d_p_indexes_origin < p2p1length_2*1.2) )
                  {
                    img->put(x,y,0);
                    // printf("x=%i y=%i\n", x, y);
                    if(theta!=NULL){
                      float theta_v = atan2((p2->coords[1]-p1->coords[1]),
                                            p2->coords[0]-p1->coords[0]);
                      theta->put(x,y,theta_v);
                    }
                    if(!(width_img==NULL)){
                      float w = (p2->coords[3] + p1->coords[3])/2;
                      width_img->put(x,y,w);
                    }
                  }
              }
}



/** Draws the neuron as voxels in the cube.*/
void Neuron::renderInCube
(
 Cube<uchar,ulong>* positive_mask,
 Cube<float, double>* theta,
 Cube<float, double>* phi,
 Cube<float, double>* scale,
 float min_width,
 float renderScale
 )
{
  positive_mask->put_all(255);
  printf("  going for segments [");
  for(int i = 0; i < axon.size(); i++){
    renderSegmentInCube(axon[i], positive_mask, theta, phi, scale,  min_width, renderScale);
  }
  for(int i = 0; i < dendrites.size(); i++){
    renderSegmentInCube(dendrites[i], positive_mask, theta, phi, scale,  min_width, renderScale);
    printf("#"); fflush(stdout);
  }
  printf("  ]\n");
}

NeuronSegment* Neuron::splitSegment(NeuronSegment* toSplit, int pointIdx)
{

  vector< NeuronPoint  > child0points;
  vector< NeuronMarker > child0markers;
  vector< NeuronSegment* > child0childs;
  string child0ending = toSplit->ending;
  NeuronColor child0color = toSplit->color;

  if(pointIdx == toSplit->points.size()-1)
    return toSplit;

  for(int i = pointIdx+1; i < toSplit->points.size(); i++){
    child0points.push_back(
                 NeuronPoint(
                             toSplit->points[i].coords[0],
                             toSplit->points[i].coords[1],
                             toSplit->points[i].coords[2],
                             toSplit->points[i].coords[3],
                             toSplit->points[i].noSenseNumber,
                             toSplit->points[i].pointNumber-pointIdx)
                           );
  }

  for(int i = 0; i < toSplit->markers.size(); i++)
    child0markers.push_back(toSplit->markers[i]);

  for(int i = 0; i < toSplit->childs.size(); i++)
    child0childs.push_back(toSplit->childs[i]);

  NeuronSegment* child0 = new NeuronSegment(toSplit,
                                           child0points,
                                           child0markers,
                                           child0childs,
                                           child0ending,
                                           child0color);

  child0->name = toSplit->name + "-1";

  NeuronSegment* child1 = new NeuronSegment(toSplit,
                                            "incomplete",
                                            child0color);

  child1->points.push_back(  NeuronPoint(
                             toSplit->points[pointIdx].coords[0],
                             toSplit->points[pointIdx].coords[1],
                             toSplit->points[pointIdx].coords[2],
                             toSplit->points[pointIdx].coords[3],
                             toSplit->points[pointIdx].noSenseNumber,
                             toSplit->points[pointIdx].pointNumber-pointIdx)
                           );

  child1->name = toSplit->name + "-2";

  toSplit->childs.resize(0);
  toSplit->markers.resize(0);
  toSplit->points.resize(pointIdx+1);
  toSplit->childs.push_back(child0);
  toSplit->childs.push_back(child1);


  return toSplit->childs[1];
}

// void Neuron::renderEdgeInCube
// (NeuronPoint* _p1, NeuronPoint* _p2,
 // Cube<uchar,ulong>* cube,
 // Cube<float,double>* theta,
 // Cube<float,double>* phi,
 // Cube<float, double>* scale,
 // float renderScale
// )
// {
  // //This is a hack to convert the rendering from neuron coordinates to micrometers
  // //We will create the points p1 and p2 with the coordinates of _p1 and _p2 in global
  // // micrometers and not in neuron ones.
  // NeuronPoint* p1 = new NeuronPoint();
  // NeuronPoint* p2 = new NeuronPoint();

  // this->neuronToMicrometers(_p1->coords, p1->coords);
  // this->neuronToMicrometers(_p2->coords, p2->coords);

  // double seglen = sqrt(
    // (p1->coords[0]-p2->coords[0])*(p1->coords[0]-p2->coords[0]) +
    // (p1->coords[1]-p2->coords[1])*(p1->coords[1]-p2->coords[1]) +
    // (p1->coords[2]-p2->coords[2])*(p1->coords[2]-p2->coords[2]) );

  // //Calculates the medium point
  // vector< float > micrometers(3);
  // micrometers[0] = (p1->coords[0]+p2->coords[0])/2;
  // micrometers[1] = (p1->coords[1]+p2->coords[1])/2;
  // micrometers[2] = (p1->coords[2]+p2->coords[2])/2;
  // vector< int > indexes_orig(3);
  // cube->micrometersToIndexes(micrometers, indexes_orig);

  // vector<int> p1idx(3);
  // cube->micrometersToIndexes(p1->coords, p1idx);
  // vector<int> p2idx(3);
  // cube->micrometersToIndexes(p2->coords, p2idx);

  // double dist_calc = 0;
  // vector<int> p2p1(3);
  // p2p1[0] = p2idx[0] - p1idx[0];
  // p2p1[1] = p2idx[1] - p1idx[1];
  // p2p1[2] = p2idx[2] - p1idx[2];
  // int p2p1mod =
    // p2p1[0]*p2p1[0] +
    // p2p1[1]*p2p1[1] +
    // p2p1[2]*p2p1[2];
  // vector<int> p1p0(3);
  // vector<int> p2p0(3);
  // int p1p0mod = 0;
  // int dot_p1p0p2p1;
  // float p2p1length_2 = sqrt(p2p1mod)/2;
  // float d_p_indexes_origin;

  // double width_microm = renderScale*(p1->coords[3] + p2->coords[3])/2;
  // double width_indexes = 3*width_microm/(cube->voxelWidth + cube->voxelHeight + cube->voxelDepth);
  // // double radius = 5*sqrt((seglen/2)*(seglen/2) + (width_microm/2)*(width_microm/2));
  // double radius = 5*(p2p1length_2 +  width_microm);

  // //We assume that voxelWidth aprox voxelheight aprox voxelDepth = 0.8
// //   radius = (int)(radius/0.8);
// //   float width = (width_microm / 0.8);
  // float width = width_microm;
  // for(int z = (int)max(0.0, indexes_orig[2]-radius);
      // z < min(double(cube->cubeDepth), indexes_orig[2]+radius);
      // z++)
    // for(int y = (int)max(0.0, indexes_orig[1]-radius); 
        // y < min(double(cube->cubeHeight), indexes_orig[1]+radius);
        // y++)
      // for(int x = (int)max(0.0, indexes_orig[0]-radius); 
              // x < min(double(cube->cubeWidth), indexes_orig[0]+radius);
              // x++)
              // {
                // //Calculates the distance between the point [x,y,z] and
                // // the edge between p1 and p2
                // p1p0[0] = p1idx[0]-x;
                // p1p0[1] = p1idx[1]-y;
                // p1p0[2] = p1idx[2]-z;
                // p1p0mod = p1p0[0]*p1p0[0] +
                          // p1p0[1]*p1p0[1] +
                          // p1p0[2]*p1p0[2];
                // p2p0[0] = p2idx[0]-x;
                // p2p0[1] = p2idx[1]-y;
                // p2p0[2] = p2idx[2]-z;
                // dot_p1p0p2p1 = p1p0[0]*p2p1[0] + p1p0[1]*p2p1[1] + p1p0[2]*p2p1[2];

                // dist_calc = float(p1p0mod*p2p1mod - dot_p1p0p2p1*dot_p1p0p2p1)/p2p1mod;

                // // Calculates the distance from the point to the middle of the edge
                // d_p_indexes_origin = sqrt( (double)(x-indexes_orig[0]) * (x-indexes_orig[0]) + 
                                           // (y-indexes_orig[1]) * (y-indexes_orig[1]) );
                // if((dist_calc <= width)  &&
                   // (d_p_indexes_origin < p2p1length_2*1.2) )
                  // {
                  // cube->put(x,y,z,0);
                  // if(theta!=NULL){
                    // float theta_v = atan2((p2->coords[1]-p1->coords[1]),
                                          // p2->coords[0]-p1->coords[0]);
                    // theta->put(x,y,z,theta_v);
                  // }
                  // if(phi!=NULL){
                    // float r_phi = sqrt(
                                       // (p2->coords[0]-p1->coords[0])*
                                       // (p2->coords[0]-p1->coords[0]) +
                                       // (p2->coords[1]-p1->coords[1])*
                                       // (p2->coords[1]-p1->coords[1]) +
                                       // (p2->coords[2]-p1->coords[2])*
                                       // (p2->coords[2]-p1->coords[2])
                                       // );
                    // float phi_v = acos((p2->coords[2]-p1->coords[2])/r_phi);
                    // phi->put(x,y,z,phi_v);
                  // }
                  // if(scale!=NULL)
                    // scale->put(x,y,z,(p1->coords[3] + p2->coords[3])/2);
                // }
              // }
// }

void Neuron::renderEdgeInCube
(NeuronPoint* _p1, NeuronPoint* _p2,
 Cube<uchar,ulong>* cube,
 Cube<float,double>* theta,
 Cube<float,double>* phi,
 Cube<float, double>* scale,
 float renderScale
)
{
  //This is a hack to convert the rendering from neuron coordinates to micrometers
  //We will create the points p1 and p2 with the coordinates of _p1 and _p2 in global
  // micrometers and not in neuron ones.
  //names: p0 -> point to add or not, pm-> point in the middle

  NeuronPoint* p1 = new NeuronPoint();
  NeuronPoint* p2 = new NeuronPoint();

  this->neuronToMicrometers(_p1->coords, p1->coords);
  this->neuronToMicrometers(_p2->coords, p2->coords);

  double seglen = sqrt(
    (p1->coords[0]-p2->coords[0])*(p1->coords[0]-p2->coords[0]) +
    (p1->coords[1]-p2->coords[1])*(p1->coords[1]-p2->coords[1]) +
    (p1->coords[2]-p2->coords[2])*(p1->coords[2]-p2->coords[2]) );

  //Calculates the medium point
  vector< float > pm(3);
  pm[0] = (p1->coords[0]+p2->coords[0])/2;
  pm[1] = (p1->coords[1]+p2->coords[1])/2;
  pm[2] = (p1->coords[2]+p2->coords[2])/2;

  vector< int > pmi(3);
  cube->micrometersToIndexes(pm, pmi);
  vector<int> p1i(3);
  cube->micrometersToIndexes(p1->coords, p1i);
  vector<int> p2i(3);
  cube->micrometersToIndexes(p2->coords, p2i);

  //Width in micrometers of the segment
  float width  = (p1->coords[3] + p2->coords[3])/2;
  //The search radius in micrometers, we assume isotropy in micrometers
  float radius = 1.5*renderScale*sqrt(seglen*seglen/4 + width*width);
  int radius_x = ceil(double(radius)/cube->voxelWidth);
  int radius_y = ceil(double(radius)/cube->voxelHeight);
  int radius_z = ceil(double(radius)/cube->voxelDepth);

  cube->put_value_in_line(0, p1i[0],p1i[1],p1i[2],
                          p2i[0],p2i[1],p2i[2]);

  cube->put_value_in_ellipsoid(0, p1i[0],p1i[1],p1i[2],
                               ceil(renderScale*p1->coords[3]/cube->voxelWidth),
                               ceil(renderScale*p1->coords[3]/cube->voxelHeight),
                               ceil(renderScale*p1->coords[3]/cube->voxelDepth));
  cube->put_value_in_ellipsoid(0, p2i[0],p2i[1],p2i[2],
                               ceil(renderScale*p2->coords[3]/cube->voxelWidth),
                               ceil(renderScale*p2->coords[3]/cube->voxelHeight),
                               ceil(renderScale*p2->coords[3]/cube->voxelDepth));

  vector< int > p0i(3);
  vector< float > p0(3);
  vector< float > pmp0(3);
  vector< float > pmp2u(3);
  vector< float > p0v(3);
  pmp2u[0] = p2->coords[0] - pm[0];
  pmp2u[1] = p2->coords[1] - pm[1];
  pmp2u[2] = p2->coords[2] - pm[2];
  float pmp2v = sqrt(pmp2u[0]*pmp2u[0] + pmp2u[1]*pmp2u[1] +
                     pmp2u[2]*pmp2u[2]);
  pmp2u[0] = pmp2u[0]/pmp2v;   pmp2u[1] = pmp2u[1]/pmp2v;   pmp2u[2] = pmp2u[2]/pmp2v;
  float pmp0xpmp2u;
  float vertDist;

  for(int z = (int)max(0, pmi[2]-radius_z);
      z < min((int)cube->cubeDepth, pmi[2]+radius_z);
      z++){
    for(int y = (int)max(0, pmi[1]-radius_y);
        y < min((int)cube->cubeHeight, pmi[1]+radius_y);
        y++){
      for(int x = (int)max(0, pmi[0]-radius_x);
          x < min((int)cube->cubeWidth, pmi[0]+radius_x);
          x++)
        {
          p0i[0]=x;p0i[1]=y;p0i[2]=z;
          cube->indexesToMicrometers(p0i, p0);
          pmp0[0] = p0[0]-pm[0];
          pmp0[1] = p0[1]-pm[1];
          pmp0[2] = p0[2]-pm[2];
          //Computes the projecton of the point in the line of the p2pm
          pmp0xpmp2u = pmp0[0]*pmp2u[0] + pmp0[1]*pmp2u[1] + pmp0[2]*pmp2u[2];
          //If it is too far in the projection of the line, do nothing
          if(fabs(pmp0xpmp2u) > 1.2*seglen/2)
            continue;
          //Computes the vertical distance between the point and the line
          p0v[0] = pmp0[0] - pmp0xpmp2u*pmp2u[0];
          p0v[1] = pmp0[1] - pmp0xpmp2u*pmp2u[1];
          p0v[2] = pmp0[2] - pmp0xpmp2u*pmp2u[2];
          vertDist = sqrt(p0v[0]*p0v[0] + p0v[1]*p0v[1] + p0v[2]*p0v[2]);
          if( vertDist > width*renderScale)
            continue;
          cube->put(x,y,z,0);
          if(theta!=NULL){
            float theta_v = atan2((p2->coords[1]-p1->coords[1]),
                                  p2->coords[0]-p1->coords[0]);
            theta->put(x,y,z,theta_v);
          }
          if(phi!=NULL){
            float r_phi = sqrt(
                               (p2->coords[0]-p1->coords[0])*
                               (p2->coords[0]-p1->coords[0]) +
                               (p2->coords[1]-p1->coords[1])*
                               (p2->coords[1]-p1->coords[1]) +
                               (p2->coords[2]-p1->coords[2])*
                               (p2->coords[2]-p1->coords[2])
                               );
            float phi_v = acos((p2->coords[2]-p1->coords[2])/r_phi);
            phi->put(x,y,z,phi_v);
          }
          if(scale!=NULL)
            scale->put(x,y,z,(p1->coords[3] + p2->coords[3])/2);
        }
    }
  }
}



void Neuron::renderSegmentInCube
(NeuronSegment* segment, Cube<uchar,ulong>* cube,
 Cube<float, double>* theta, Cube<float, double>* phi,
 Cube<float, double>* scale, float min_width, float renderScale
)
{

//   if(segment->name == "a-00-2-2-2-2-2"){
//     printf("Here in %s\n", segment->name.c_str());
//   }

  if(segment->parent != NULL){
    NeuronPoint* p1 = &segment->parent->points[segment->parent->points.size()-1];
    NeuronPoint* p2 = &segment->points[0];
    if((p1->coords[3] > min_width) &&
       (p2->coords[3] > min_width) )
      renderEdgeInCube(p1, p2, cube, theta, phi, scale, renderScale);
  }

  for(int i = 1; i < segment->points.size(); i++){
    if((segment->points[i-1].coords[3] > min_width) &&
       (segment->points[i].coords[3]   > min_width) )
      renderEdgeInCube(&segment->points[i-1], &segment->points[i], cube,
                       theta, phi, scale, renderScale);
  }
  for(int i = 0; i < segment->childs.size(); i++)
    renderSegmentInCube(segment->childs[i], cube, theta, phi,
                        scale,  min_width, renderScale);

}


void Neuron::toCloudOld(string points_file,
                     string edges_file,
                     float width_sampled,
                     Cube<uchar, ulong>* cube)
{

  std::ofstream points_of(points_file.c_str());
  std::ofstream edges_of(edges_file.c_str());

  int point_number = 0;
  for(int i = 0; i < dendrites.size(); i++)
    toCloudOld(points_of,
            edges_of, width_sampled,
            dendrites[i], cube, point_number);
  for(int i = 0; i < axon.size(); i++)
    toCloudOld(points_of, edges_of,
            width_sampled, axon[i], cube, point_number);
  points_of.close();
  edges_of.close();
}

void Neuron::toCloudOld(std::ofstream& points_of,
                     std::ofstream& edges_of,
                     float width_sampled,
                     NeuronSegment* segment,
                     Cube<uchar, ulong>* cube,
                     int& last_point_number)
{

//   vector< float > p_microm(3);
//   vector< int > p_idx(3);
//   vector< int > nxt_p_idx(3);
//   float distance = 0;

//   for(int i = 0; i < segment->points.size()-1; i++){
//     if(segment->points[i].coords[3] < width_sampled)
//       continue;
//     this->neuronToMicrometers(segment->points[i].coords, p_microm);
//     cube->micrometersToIndexes(p_microm, p_idx);

//     for(int j = i; j < segment->points.size(); j++)
//       {
//         this->neuronToMicrometers(segment->points[j].coords, p_microm);
//         cube->micrometersToIndexes(p_microm, nxt_p_idx);
//         distance = sqrt((p_idx[0]-nxt_p_idx[0])*(p_idx[0]-nxt_p_idx[0]) +
//                         (p_idx[1]-nxt_p_idx[1])*(p_idx[1]-nxt_p_idx[1]) +
//                         (p_idx[2]-nxt_p_idx[2])*(p_idx[2]-nxt_p_idx[2]));
//         if(distance > 15){
//           edges_of << last_point_number << " ";
//           last_point_number++;
//           points_of << p_idx[0] << " " << p_idx[1] << " " << p_idx[2] << std::endl;
//           edges_of << last_point_number << " " << distance << std::endl;
//           points_of << nxt_p_idx[0] << " " << nxt_p_idx[1] << " " << nxt_p_idx[2] << std::endl;
//           last_point_number++;
//           break; // Only for the first far point
//         }
//       }
//   }

//   for(int i = 0; i < segment->childs.size(); i++)
//     toCloud(points_of, edges_of, width_sampled,
//             segment->childs[i], cube, last_point_number);

  vector< float > p_microm(3);
  vector< float > nxt_p_microm(3);
  vector< int > p_idx(3);
  vector< int > nxt_p_idx(3);
  float distance = 0;

  for(int i = 0; i < segment->points.size()-1; i++){
    if(segment->points[i].coords[3] < width_sampled)
      continue;
    this->neuronToMicrometers(segment->points[i].coords, p_microm);
    cube->micrometersToIndexes(p_microm, p_idx);
    this->neuronToMicrometers(segment->points[i+1].coords, nxt_p_microm);
    cube->micrometersToIndexes(nxt_p_microm, nxt_p_idx);

    if ( (p_idx[0] < 0) ||
         (p_idx[0] >= cube->cubeWidth) ||
         (p_idx[1] < 0) ||
         (p_idx[1] >= cube->cubeHeight) ||
         (p_idx[2] < 0) ||
         (p_idx[2] >= cube->cubeDepth) ||
         (nxt_p_idx[0] < 0) ||
         (nxt_p_idx[0] >= cube->cubeWidth) ||
         (nxt_p_idx[1] < 0) ||
         (nxt_p_idx[1] >= cube->cubeHeight) ||
         (nxt_p_idx[2] < 0) ||
         (nxt_p_idx[2] >= cube->cubeDepth)
         )
      continue;

    //Threhold to elliminate points really close by. This is due to the awful habit of double
    //clicking in the neuron
    if( (p_idx[0]-nxt_p_idx[0])*(p_idx[0]-nxt_p_idx[0]) +
        (p_idx[1]-nxt_p_idx[1])*(p_idx[1]-nxt_p_idx[1]) +
        (p_idx[2]-nxt_p_idx[2])*(p_idx[2]-nxt_p_idx[2])  < 
        10)
      continue;

    this->neuronToMicrometers(segment->points[i+1].coords, nxt_p_microm);


    points_of << (p_idx[0]) << " "
              << (p_idx[1]) << " " 
              << (segment->points[i].coords[3]) << " "
              << atan2(-(nxt_p_microm[1] - p_microm[1]),
                       nxt_p_microm[0] - p_microm[0])*180/M_PI
              << " " << 1 << std::endl;
  }

  for(int i = 0; i < segment->childs.size(); i++)
    toCloudOld(points_of, edges_of, width_sampled,
            segment->childs[i], cube, last_point_number);
}



//Ich bin ein Berliner
Cloud_P* Neuron::toCloud(string cloudName,
                         bool saveOrientation,
                         bool saveType,
                         Cube_P* cubeLimit,
                         bool saveWidth)
{

  Cloud_P* cloud;
  if(saveOrientation && saveType && !saveWidth){
    cloud = new Cloud< Point3Dot >(cloudName);
  } else if (saveOrientation & !saveType) {
    cloud = new Cloud< Point3Do >(cloudName);
  } else if (!saveOrientation & !saveType){
    cloud = new Cloud< Point3D >(cloudName);
  } else if (saveOrientation & saveType & saveWidth) {
    cloud = new Cloud< Point3Dotw >(cloudName);
  }

  int point_number = 0;
  for(int i = 0; i < dendrites.size(); i++)
    toCloud(dendrites[i], cloud, saveOrientation,
            saveType, cubeLimit, saveWidth);
  for(int i = 0; i < axon.size(); i++)
    toCloud(dendrites[i], cloud, saveOrientation,
            saveType, cubeLimit, saveWidth);
  cloud->saveToFile(cloudName);
  return cloud;
}


void Neuron::toCloud(NeuronSegment* segment,
                     Cloud_P* cloud,
                     bool saveOrientation,
                     bool saveType,
                     Cube_P* cubeLimit,
                     bool saveWidth)
{

  vector< float > mcoords(3);
  vector< float > mcnext(3);
  float theta;
  float phi;
  float radius;

  //For the indexing of the vertices of the cube, see doc/coordinates.txt
  vector< float > cv0mic(3);
  vector< float > cv6mic(3);
  vector< int > v_0(3);
  vector< int > v_6(3);
  if( cubeLimit != NULL){
    v_0[0]=0;v_0[1]=0;v_0[2]=0;
    v_6[0]=cubeLimit->cubeWidth;
    v_6[1]=cubeLimit->cubeHeight;
    v_6[2]=cubeLimit->cubeDepth;
    cubeLimit->indexesToMicrometers(v_0, cv0mic);
    cubeLimit->indexesToMicrometers(v_6, cv6mic);
  }


  for(int i = 0; i < segment->points.size(); i++){
    neuronToMicrometers(segment->points[i].coords, mcoords);

    if(cubeLimit!= NULL){
      if( (mcoords[0] < cv0mic[0]) | (mcoords[0] > cv6mic[0]) |
          (mcoords[1] > cv0mic[1]) | (mcoords[1] < cv6mic[1]) |
          (mcoords[2] < cv0mic[2]) | (mcoords[2] > cv6mic[2]) )
        continue;
    }

    if(saveOrientation){
      if(i!= segment->points.size()-1)
        neuronToMicrometers(segment->points[i+1].coords, mcnext);
      else if (i >= 1)
        neuronToMicrometers(segment->points[i-1].coords, mcnext);
      else continue;
      theta = atan2(mcnext[1] - mcoords[1], mcnext[0]-mcoords[0]);
      radius = sqrt( (mcnext[2]-mcoords[2])*(mcnext[2]-mcoords[2]) +
                     (mcnext[1]-mcoords[1])*(mcnext[1]-mcoords[1]) +
                     (mcnext[0]-mcoords[0])*(mcnext[0]-mcoords[0]) );
      phi = acos( (mcnext[2]-mcoords[2])/radius);
    }

    if(saveOrientation && saveType && !saveWidth)
      cloud->points.push_back(new Point3Dot(mcoords[0],mcoords[1],mcoords[2],
                                            theta, phi, Point3Dot::TrainingPositive));
     if(saveOrientation && !saveType)
       cloud->points.push_back(new Point3Do(mcoords[0],mcoords[1],mcoords[2],
                                             theta, phi));
     if(!saveOrientation && !saveType)
       cloud->points.push_back(new Point3D(mcoords[0],mcoords[1],mcoords[2]));

    if(saveOrientation && saveType && saveWidth)
      cloud->points.push_back(new Point3Dotw(mcoords[0],mcoords[1],mcoords[2],
                                             theta, phi, Point3Dot::TrainingPositive,
                                             mcoords[3]));
  }

  for(int i = 0; i < segment->childs.size(); i++)
    toCloud(segment->childs[i], cloud, saveOrientation,
            saveType, cubeLimit, saveWidth);
}
