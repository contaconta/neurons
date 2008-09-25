/** Class VisibleE
 *  defines a visible object with extended attributes, such as color 
 */

#ifndef VISIBLEE_H_
#define VISIBLEE_H_

#include "Visible.h"

class VisibleE : public Visible
{
public:

  /** Options for drawing the object.*/
  bool v_draw_projection; // true for maximum intensity, false for minimum
  double v_r; // Color of the object
  double v_g;
  double v_b;
  double v_radius; //In case the object has a radius
  bool   v_enable_depth_test;
  bool   v_blend;

  /** In case we need to create a list to speed things up.*/
  int v_glList;
  bool v_saveVisibleAttributes; //If we want to save the vissible attributes.

  VisibleE() : Visible(){
    v_draw_projection = false;
    v_r = 1.0;
    v_g = 0.0;
    v_b = 0.0;
    v_radius = 1;
    v_glList = 0;
    v_saveVisibleAttributes = true;
    v_enable_depth_test = false;
    v_blend = true;
  }

  void save(ostream &out)
  {
    if(v_saveVisibleAttributes){
      out << "<VisibleE>" << std::endl;
      out << "v_r " << v_r << std::endl;
      out << "v_g " << v_g << std::endl;
      out << "v_b " << v_b << std::endl;
      out << "v_radius " << v_radius << std::endl;
      out << "v_enable_depth_test " << v_enable_depth_test << std::endl;
      out << "v_blend " << v_blend << std::endl;
      out << "</VisibleE>" << std::endl;
    }
  }

  bool load(istream& in)
  {
    if(v_saveVisibleAttributes){
      string s;
      string val;
      int start = in.tellg();
      in >> s;
      if(s!="<VisibleE>")
        return false;
      in >> s;
      while(s!="</VisibleE>"){
        in >> val;
        // std::cout << s << " " << val << std::endl;
        stringstream iss(val);
        if(s=="v_r")
          iss >> v_r;
        else if(s=="v_g")
          iss >> v_g;
        else if(s=="v_b")
          iss >> v_b;
        else if(s=="v_radius")
          iss >> v_radius;
        else if(s=="v_enable_depth_test")
          iss >> v_enable_depth_test;
        else if(s=="v_blend")
          iss >> v_blend;
        else{
          printf("Vissible:error: parameter %s not known. Exiting the parsing\n", s.c_str());
          in.seekg(start);
          return false;
        }
        in >> s;
      }
    } //if

    return true;
  }

  virtual void draw(){
    glColor3f(v_r, v_g, v_b);
    glLineWidth(v_radius);
    if(v_enable_depth_test)
      glEnable(GL_DEPTH_TEST);
    else
      glDisable(GL_DEPTH_TEST);
  }

};

#endif
