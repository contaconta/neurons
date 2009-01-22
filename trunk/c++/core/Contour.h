#ifndef CONTOUR_H_
#define CONTOUR_H_

#include <sstream>
#include "Point.h"
#include "VisibleE.h"

template < class P=Point>
class Contour : public VisibleE
{
private:
    static int contour_id;

    void init();

public:
    string contour_name;

    vector< Point* >* points;

    Contour();

    Contour(vector< Point* >* _points);

    ~Contour();

    void addPoint(Point* point);

    void clear();

    void draw();

    void save(const string& filename);

    bool load(istream &in);

    static string className(){
        return "Contour";
    }

};

template< class P>
int Contour<P>::contour_id = 0;

template< class P>
Contour<P>::Contour() : VisibleE(){
    init();
    points = new vector<Point*>;
}

template< class P>
Contour<P>::Contour(vector< Point* >* _points) : VisibleE(){
    init();
    points = _points;
}

template< class P>
void Contour<P>::init(){
    std::string s;
    std::stringstream out;
    out << contour_id;
    contour_name = "contour " + out.str();
    contour_id++;
}

template< class P>
//Contour<P>::~Contour() : ~Visible(){
Contour<P>::~Contour() {
    for(vector< Point* >::iterator itPoints = points->begin();
        itPoints != points->end(); itPoints++)
    {
        delete *itPoints;
    }
    delete points;
}

template< class P>
void Contour<P>::clear(){
    points->clear();
}

template< class P>
void Contour<P>::draw(){
    glColor3f(1,0,0);
    glPushAttrib(GL_LINE_BIT);
    glLineWidth(6.0f);
    glBegin(GL_LINE_STRIP);
    for(vector< Point* >::iterator itPoints = points->begin();
        itPoints != points->end(); itPoints++)
    {
        glVertex3f((*itPoints)->coords[0],(*itPoints)->coords[1],(*itPoints)->coords[2]);
    }
    glEnd();
    glPopAttrib();
}

template< class P>
void Contour<P>::save(const string& filename){

    if(points->size()==0)
        return;

    std::ofstream writer(filename.c_str());

    if(!writer.good())
    {
        printf("Error while creating file %s\n", filename.c_str());
        return;
    }

    for(vector< Point* >::iterator itPoints = points->begin();
        itPoints != points->end(); itPoints++)
    {
        writer << **itPoints <<  std::endl;
    }

    writer.close();
}

template< class P>
bool Contour<P>::load(istream &in){
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
void Contour<P>::addPoint(Point* point)
{
    points->push_back(point);
}

#endif //CONTOUR_H_
