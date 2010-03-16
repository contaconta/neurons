#ifndef DOUBLESET_H_
#define DOUBLESET_H_

#include <sstream>
#include "Point3D.h"
#include "Point.h"
#include "VisibleE.h"
#include "Cube_P.h"
#include "Image.h"

template < class C=float>
class PointDs
{
 public:
  vector<C> coords;
  vector<int> indexes;

  // Save indexes
  void save(ostream &out){
    if(indexes.size()>0)
      {
        for(int i = 0; i < indexes.size()-1; i++)
          out << indexes[i] << " ";
        out << indexes[indexes.size()-1] << std::endl;
      }
  }

  // Save world coordinates
  void save_coords(ostream &out){
    if(coords.size()>0)
      {
        for(int i = 0; i < coords.size()-1; i++)
          out << coords[i] << " ";
        out << coords[coords.size()-1] << std::endl;
      }
  }

  // Load indexes
  bool load(istream &in, int ptSize=3){
   indexes.resize(ptSize);
   int start = in.tellg();
    for(int i = 0; i < ptSize; i++){
      in >> indexes[i];
      if(in.fail()){
        in.clear();
        //cout << "load failed\n";
        //in.seekg(start+1); //????????? Why that one
        return false;
      }
    }
    return true;
  }

  // Load world coordinates
  bool load_coords(istream &in, int ptSize=3){
    coords.resize(ptSize);
    int start = in.tellg();
    for(int i = 0; i < ptSize; i++){
      in >> coords[i];
      if(in.fail()){
        in.clear();
        //cout << "load failed\n";
        //in.seekg(start+1); //????????? Why that one
        return false;
      }
    }
    return true;
  }

  void draw(float width){
    glPushMatrix();
    glTranslatef(coords[0], coords[1], coords[2]);
    glutSolidSphere(width, 10, 10);
    glPopMatrix();
  }
};

template <class P=float,class T=Point3D>
class DoubleSet : public VisibleE
{
 private:
 static int id;

 void init();

 public:
 string name;

 vector< PointDs<P>* > set1;
 vector< PointDs<P>* > set2;

 DoubleSet();

 ~DoubleSet();

 void addPoint(PointDs<P>* point, int setId);

 void clear();

 void draw(int x, int y, int z,float point_radius=1.0f);
 void draw(float point_radius=1.0f);

 void save(const string& filename);

 bool load(const string& filename, Cube_P* cube=0, Image<float>* img=0);

 bool load(istream &in, Cube_P* cube=0, Image<float>* img=0);
 bool load_micrometers(istream &in, Cube_P* cube=0);

 virtual string className(){
   return "DoubleSet";
 }

};

template<class P,class T>
int DoubleSet<P,T>::id = 0;

template<class P,class T>
DoubleSet<P,T>::DoubleSet() : VisibleE(){
  init();
}

template<class P,class T>
void DoubleSet<P,T>::init(){
  std::string s;
  std::stringstream out;
  out << id;
  name = "Double_set_" + out.str();
  id++;
}

template<class P,class T>
//DoubleSet<P,T>::~DoubleSet() : ~Visible(){
DoubleSet<P,T>::~DoubleSet() {
  typename vector< PointDs<P> *>::iterator itPointDs;
  for(itPointDs = set1.begin();
      itPointDs != set1.end(); itPointDs++)
    {
      delete *itPointDs;
    }
  for(itPointDs = set2.begin();
      itPointDs != set2.end(); itPointDs++)
    {
      delete *itPointDs;
    }
}

template<class P,class T>
void DoubleSet<P,T>::clear(){
  set1.clear();
  set2.clear();
}

template<class P,class T>
void DoubleSet<P,T>::draw(int x,int y,int z,float point_radius){
  //glPushAttrib(GL_LINE_BIT);
  //glLineWidth(6.0f);
  //glBegin(GL_LINE_STRIP);
  glDisable(GL_DEPTH_TEST);
  glColor3f(1,0,0);
  typename vector< PointDs<P> *>::iterator itPointDs;
  for(itPointDs = set1.begin();
      itPointDs != set1.end(); itPointDs++)
    {
      //printf("Draw set1 %d %d\n",z,(*itPointDs)->indexes[2]);
      if(z==-1 || (*itPointDs)->indexes[2] == z)
	{
          (*itPointDs)->draw(point_radius);
        }
    }
  glColor3f(0,1,0);
  for(itPointDs = set2.begin();
      itPointDs != set2.end(); itPointDs++)
    {
      //printf("Draw set2 %d %d\n",z,(*itPointDs)->indexes[2]);
      if(z==-1 || (*itPointDs)->indexes[2] == z)
	{
          (*itPointDs)->draw(point_radius);
        }
    }
  glEnable(GL_DEPTH_TEST);
  //glEnd();
  //glPopAttrib();
}

template<class P,class T>
void DoubleSet<P,T>::draw(float point_radius){
  //glPushAttrib(GL_LINE_BIT);
  //glLineWidth(6.0f);
  //glBegin(GL_LINE_STRIP);
  glDisable(GL_DEPTH_TEST);
  glColor3f(1,0,0);
  typename vector< PointDs<P> *>::iterator itPointDs;
  for(itPointDs = set1.begin();
      itPointDs != set1.end(); itPointDs++)
    {
      //glVertex3f((*itPointDs)->coords[0],(*itPointDs)->coords[1],(*itPointDs)->coords[2]);
      (*itPointDs)->draw(point_radius);
    }
  glColor3f(0,1,0);
  for(itPointDs = set2.begin();
      itPointDs != set2.end(); itPointDs++)
    {
      //glVertex3f((*itPointDs)->coords[0],(*itPointDs)->coords[1],(*itPointDs)->coords[2]);
      (*itPointDs)->draw(point_radius);
    }
  glEnable(GL_DEPTH_TEST);
  //glEnd();
  //glPopAttrib();
}

template<class P,class T>
  bool DoubleSet<P,T>::load(const string& filename, Cube_P* cube, Image<float>* img)
{
  ifstream in(filename.c_str());
  bool bRes = load(in,cube,img);
  in.close();
  return bRes;
}

template<class P,class T>
  bool DoubleSet<P,T>::load(istream &in, Cube_P* cube, Image<float>* img)
{
  float x,y,z;
  int start = in.tellg();
  string s;
  in >> s;
  int orig = s.find("<Cloud");
  if(orig == string::npos){
    printf("Cloud::error load called when there is no beginning of Cloud, instead: %s\n",
           s.c_str());
    in.seekg(start);
    return false;
  }
  in >> s;
  T* tp = new T;
  orig = s.find(tp->className()+">");
  
  if(orig == string::npos){
    printf("Cloud::error load called when there is no type of the class %s\n", tp->className().c_str());
    in.seekg(start);
    delete tp;
    return false;
  }
  delete tp;

  if(!VisibleE::load(in))
    return false;
  
  // Load Set1
  //cout << "s1:" << s << endl;
  in >> s;
  //cout << "s2:" << s << endl;
  orig = s.find("<Set1>");
  if(orig == string::npos){
    printf("Cloud::error load didn't find Set1\n");
    in.seekg(start);
    return false;
  }
  PointDs<P>* pt = new PointDs<P>();
  while(pt->load(in)){

    // DEBUG
    printf("p ");
    for(int i = 0;i<pt->indexes.size();i++)
      printf("%f ", (float)pt->indexes[i]);
    printf("\n");

    if(cube!=0 && cube->dummy == false)
      {
        cube->indexesToMicrometers3(pt->indexes[0],
                                    pt->indexes[1],
                                    pt->indexes[2],
                                    x,y,z);

        printf("Cube found 1 %f %f %f\n",x,y,z);

        pt->coords.push_back(x);
        pt->coords.push_back(y);
        pt->coords.push_back(z);
      }
    else
      {
        if(img)
          {
            img->indexesToMicrometers(pt->indexes[0],
                                      pt->indexes[1],
                                      x,y);
            z = pt->indexes[2];

            printf("Image found 1 %f %f %f\n",x,y,z);
            
            pt->coords.push_back(x);
            pt->coords.push_back(y);
            pt->coords.push_back(z);
          }
      }

    set1.push_back(pt);
    pt = new PointDs<P>();
  }
  delete pt;

  // Load Set2
  //cout << "s1:" << s << endl;
  in >> s;
  //cout << "s2:" << s << endl;
  in >> s;
  //cout << "s3:" << s << endl;
  orig = s.find("<Set2>");
  if(orig == string::npos){
    printf("Cloud::error load didn't find Set2\n");
    in.seekg(start);
    return false;
  }
  pt = new PointDs<P>();
  while(pt->load(in)){

    // DEBUG
    printf("p ");
    for(int i = 0;i<pt->indexes.size();i++)
      printf("%f ", (float)pt->indexes[i]);
    printf("\n");

    if(cube!=0 && cube->dummy == false)
      {
        cube->indexesToMicrometers3(pt->indexes[0],
                                    pt->indexes[1],
                                    pt->indexes[2],
                                    x,y,z);

        printf("Cube found 2 %f %f %f\n",x,y,z);

        pt->coords.push_back(x);
        pt->coords.push_back(y);
        pt->coords.push_back(z);
      }
    else
      {
        if(img)
          {
            // DEBUG
            cube->indexesToMicrometers3(pt->indexes[0],
                                        pt->indexes[1],
                                        pt->indexes[2],
                                        x,y,z);
            printf("Cube found 2 %f %f %f\n",x,y,z);

            img->indexesToMicrometers(pt->indexes[0],
                                      pt->indexes[1],
                                      x,y);
            z = pt->indexes[2];

            printf("Image found 2 %f %f %f\n",x,y,z);
            
            pt->coords.push_back(x);
            pt->coords.push_back(y);
            pt->coords.push_back(z);
          }
      }

    set2.push_back(pt);    
    pt = new PointDs<P>();
  }
  delete pt;

  //cout << "s1:" << s << endl;
  in >> s;
  //cout << "s2:" << s << endl;
  in >> s;
  //cout << "s3:" << s << endl;
  if(s.find("</Cloud>")==string::npos){
    printf("Cloud::error load can not find </Cloud>\n");
    in.seekg(start);
    return false;
  }
  return true;
}

template<class P,class T>
bool DoubleSet<P,T>::load_micrometers(istream &in, Cube_P* cube)
{
  int x,y,z;
  int start = in.tellg();
  string s;
  in >> s;
  int orig = s.find("<Cloud");
  if(orig == string::npos){
    printf("Cloud::error load called when there is no beginning of Cloud, instead: %s\n",
           s.c_str());
    in.seekg(start);
    return false;
  }
  in >> s;
  T* tp = new T;
  orig = s.find(tp->className()+">");
  
  if(orig == string::npos){
    printf("Cloud::error load called when there is no type of the class %s\n", tp->className().c_str());
    in.seekg(start);
    delete tp;
    return false;
  }
  delete tp;

  if(!VisibleE::load(in))
    return false;
  
  // Load Set1
  in >> s;
  orig = s.find("<Set1>");
  if(orig == string::npos){
    printf("Cloud::error load didn't find Set1\n");
    in.seekg(start);
    return false;
  }
  PointDs<P>* pt = new PointDs<P>();
  while(pt->load(in)){

    // DEBUG
    printf("p.z : %f\n",  pt->coords[pt->coords.size()-1]);

    if(cube)
      {        
        cube->micrometersToIndexes3(pt->coords[0],
                                    pt->coords[1],
                                    pt->coords[2],
                                    x,y,z);

        printf("Cube found 1 %d %d %d\n",x,y,z);

        pt->indexes.push_back(x);
        pt->indexes.push_back(y);
        pt->indexes.push_back(z);
      }

    set1.push_back(pt);
    pt = new PointDs<P>();
  }
  delete pt;

  // Load Set2
  in >> s;
  in >> s;
  orig = s.find("<Set2>");
  if(orig == string::npos){
    printf("Cloud::error load didn't find Set2\n");
    in.seekg(start);
    return false;
  }
  pt = new PointDs<P>();
  //in >> s;
  while(pt->load(in)){

    // DEBUG
    printf("p.z : %f\n",  pt->coords[pt->coords.size()-1]);

    if(cube)
      {
        cube->micrometersToIndexes3(pt->coords[0],
                                    pt->coords[1],
                                    pt->coords[2],
                                    x,y,z);

        printf("Cube found 2 %d %d %d\n",x,y,z);

        pt->indexes.push_back(x);
        pt->indexes.push_back(y);
        pt->indexes.push_back(z);
      }

    set2.push_back(pt);    
    pt = new PointDs<P>();
  }
  delete pt;

  in >> s;
  in >> s;
  if(s.find("</Cloud>")==string::npos){
    printf("Cloud::error load can not find </Cloud>\n");
    in.seekg(start);
    return false;
  }
  return true;
}

template<class P,class T>
void DoubleSet<P,T>::save(const string& filename){

  ofstream out(filename.c_str());

  if(!out.good())
    {
      printf("Error while creating file %s\n", filename.c_str());
      return;
    }
  
  T* tp = new T;
  out << "<Cloud " << tp->className() << ">" << std::endl;
  delete tp;
  VisibleE::save(out);

  typename vector< PointDs<P> *>::iterator itPointDs;
  out << "<Set1>" << std::endl;
  for(itPointDs = set1.begin();
      itPointDs != set1.end(); itPointDs++)
    {
      (*itPointDs)->save(out);
    }
  out << "</Set1>" << std::endl;
  out << "<Set2>" << std::endl;
  for(itPointDs = set2.begin();
      itPointDs != set2.end(); itPointDs++)
    {
      (*itPointDs)->save(out);
    }
  out << "</Set2>" << std::endl;
  out << "</Cloud>" << std::endl;
  out.close();
}

template<class P,class T>
void DoubleSet<P,T>::addPoint(PointDs<P>* point, int setId)
{
  if(setId == 1)
    set1.push_back(point);
  else if(setId == 2)
    set2.push_back(point);
}

#endif //DOUBLESET_H_
