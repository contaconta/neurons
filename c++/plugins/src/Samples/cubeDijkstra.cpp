#include <gtk/gtk.h>
#include <gmodule.h>
#include <stdio.h>
#include <Object.h>
#include "Cube_P.h"
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include "CubeDijkstra.h"
#include "Point3D.h"
// #include "../../../viewer/src/globalsE.h"

extern "C"
{

  Cube_P* localCube;
  // Enumeration for the modes
  enum actions_cd {CD_SELECTINITPOINT, CD_SELECTENDPOINT, CD_NONE};
  actions_cd action;
  Point3D* initPoint;
  Point3D* endPoint;

  G_MODULE_EXPORT const bool plugin_init()
  {
    printf("init cubeDijkstra\n");
    action = CD_NONE;
    return true;
  }

  G_MODULE_EXPORT const bool plugin_run(vector<Object*>& objects)
  {
    printf("Plugin: run\n");
    for(vector<Object*>::iterator itObject = objects.begin();
        itObject != objects.end(); itObject++)
      {
        string objType = (*itObject)->className();
        printf("Object class = %s\n", objType.c_str());
        if((*itObject)->className()=="Cube")
          {
            localCube = dynamic_cast<Cube_P*>((*itObject));
            printf("There is a Cube in here\n");
            // printf("Cube : %d\n",cube->cubeWidth);
          }
      }

    return true;
  }

  G_MODULE_EXPORT const bool plugin_key_press_event
  (GtkWidget *widget, GdkEventKey* event, gpointer user_data)
  {
    if(event->keyval == 'g'){
      switch(action){
      case CD_NONE:
        printf("The action is CD_NONE\n");
        action = CD_SELECTINITPOINT;
        break;
      case CD_SELECTINITPOINT:
        printf("The action is CD_SELECTINITPOINT\n");
        action = CD_SELECTENDPOINT;
        break;
      case CD_SELECTENDPOINT:
        printf("The action is CD_SELECTENDPOINT\n");
        action = CD_NONE;
        break;
      }
    }
  }

  G_MODULE_EXPORT const bool plugin_unproject_mouse
  (int x, int y)
  {
    printf("Plugin: The position of the mouse is %i %i\n", x, y);
    switch(action){
    case CD_NONE:
      printf("UnprojectMouse: The action is CD_NONE\n");
      break;
    case CD_SELECTINITPOINT:
      printf("UnprojectMouse: The action is CD_SELECTINITPOINT\n");
      action = CD_SELECTENDPOINT;
      break;
    case CD_SELECTENDPOINT:
      printf("UnprojectMouse: The action is CD_SELECTENDPOINT\n");
      action = CD_NONE;
      break;
    }
  }

  G_MODULE_EXPORT const bool plugin_expose
  (GtkWidget *widget, GdkEventKey* event, gpointer user_data)
  {
    glDisable(GL_DEPTH_TEST);
    glColor3f(0.0,0.0,1.0);
    glPushMatrix();
    glutWireSphere(10,10,10);
    glPopMatrix();
    glEnable(GL_DEPTH_TEST);

  }

  G_MODULE_EXPORT const bool plugin_quit()
  {
    printf("Plugin: Exit\n");
    return true;
  }



}
