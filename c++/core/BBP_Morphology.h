#ifndef BBP_MORPHOLOGY_H_
#define BBP_MORPHOLOGY_H_

#ifdef WITH_BBP

#include "neseg.h"
#include <bbp.h>
#include <iostream>
#include <BBP/Model/Microcircuit/Morphology.h>
#include <BBP/Model/Microcircuit/Morphology_RW.h>
#include <BBP/Model/Microcircuit/Readers/Morphology_Reader.h>
#include <BBP/Model/Microcircuit/Containers/Morphologies.h>


class BBP_Morphology : public VisibleE
{
public:

  bbp::rw::Morphology* morph;

  //Creates a dummy morphology for debug purposes
  BBP_Morphology(){

    morph = new bbp::rw::Morphology();
    bbp::rw::Section axon = morph->create_branch(bbp::AXON, 2.5f,
                                                 1.0f, 1.0, 1.0f, 1.5f);

    axon.push_back(2.0f, 2.0, 2.0f, 1.0f);
    axon.push_back(3.0f, 3.0, 3.0f, 1.0f);
    axon.push_back(4.0f, 4.0, 4.0f, 1.0f);

    bbp::rw::Section dendrite = morph->create_branch(bbp::DENDRITE, 3.5f,
                                                  -1.0f, -1.0, -1.0f, 1.5f);

    dendrite.push_back(-2.0f, -2.0, -2.0f, 1.0f);
    dendrite.push_back(-3.0f, -3.0, -3.0f, 1.0f);
    dendrite.push_back(-4.0f, -4.0, -4.0f, 1.0f);

    size_t count = 0;
    for (bbp::rw::Section::iterator i = dendrite.begin(); i != dendrite.end(); ++i)
      {
        if (count == 1)
          {
            morph->fork_branch(
                                   i, 0.75f, 10.0f, 10.0f, 10.0f, 0.5f);
            break;
          }
        ++count;
      }

    bbp::rw::Section apical_dendrite =
      morph->create_branch(bbp::APICAL_DENDRITE, 3.5f,
                               +2.0f, -2.0, -2.0f, 2.5f);

    apical_dendrite.push_back(2.0f, -2.0, -2.0f, 4.0f);
    apical_dendrite.push_back(3.0f, -3.0, -3.0f, 5.0f);
    apical_dendrite.push_back(4.0f, -4.0, -4.0f, 6.0f);

    count = 0;
    for (bbp::rw::Section::iterator j = apical_dendrite.begin();
         j != apical_dendrite.end(); ++j)
      {
        if (count == 1)
          {
            bbp::rw::Section new_section = 
              morph->fork_branch(j, 0.75f, -10.0f, 10.0f, 10.0f, 0.5f);
            morph->mark_tuft(new_section.parent());
            morph->mark_cut_point(new_section);
            break;
          }
        ++count;
      }
  }

  BBP_Morphology(bbp::rw::Morphology* morph){
    this->morph = morph;
  }

  BBP_Morphology(string filename){
    bbp::Morphologies morphologies;
    string dir = getDirectoryFromPath(filename);
    string name = getNameFromPathWithoutExtension(filename);

    printf("Creating a BBP_Morphology from: %s/%s\n", dir.c_str(), name.c_str());

    bbp::Morphology_Reader_Ptr reader =
      bbp::Morphology_Reader::create_reader(dir.c_str());
    std::set < std::string > names;
    names.insert(name);
    reader->load (morphologies, names);
    morph = new bbp::rw::Morphology(*(morphologies.begin()));
  }


  void drawSection(bbp::rw::Morphology::iterator section_it){
    // section->print();
    for (bbp::rw::Section::const_iterator segment = section_it->begin(); 
         segment != section_it->end(); ++segment)
      {
        bbp::Vector_3D<float> coords = segment->begin().center();
        // std::cout << coords.x() << " " << coords.y() << " " << coords.z() << std::endl;
        float diameter = segment->begin().diameter();
        glPushMatrix();
        glTranslatef(coords.x(), coords.y(), coords.z());
        // glutSolidSphere(diameter, 10, 10);
        glutSolidSphere(0.5, 10, 10);
        glPopMatrix();

        if(segment != section_it->end()){
          bbp::rw::Section::const_iterator temp = segment;
          ++temp;
          bbp::Vector_3D<float> coords2 = temp->begin().center();
          glBegin(GL_LINES);
          glVertex3f(coords.x(), coords.y(), coords.z());
          glVertex3f(coords2.x(),coords2.y(),coords2.z());
          glEnd();
          glPushMatrix();
          glTranslatef(coords2.x(), coords2.y(), coords2.z());
          // glutSolidSphere(diameter, 10, 10);
          glutSolidSphere(0.5, 10, 10);
          glPopMatrix();

        }

      }
  }


  void draw(){

    VisibleE::draw();
    //Does everything in a opengl list
    if(v_glList == 0){
      // Reduces the number of points to 2000
      v_glList = glGenLists(1);
      glNewList(v_glList, GL_COMPILE);
      for (bbp::rw::Morphology::iterator section = morph->begin();
           section != morph->end(); ++section)
        {
          // section->print);
          drawSection(section);
        }
      glEndList();
    }
    else{
      glCallList(v_glList);
    }
  }




};

#endif //WITH_BBP




#endif
