#include <gtk/gtk.h>
#include <gmodule.h>
#include <stdio.h>
#include <Object.h>
#include "Cube_P.h"
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include "Neuron.h"
// #include "../../../viewer/src/globalsE.h"

extern "C"
{

  Neuron* localNeuron;

  G_MODULE_EXPORT const bool plugin_init()
  {
    printf("init ascEdit\n");
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
        if((*itObject)->className()=="Neuron")
          {
            localNeuron = dynamic_cast<Neuron*>((*itObject));
            printf("There is a neuron in here\n");
            // printf("Cube : %d\n",cube->cubeWidth);
          }
      }

    return true;
  }

  G_MODULE_EXPORT const bool plugin_key_press_event
  (GtkWidget *widget, GdkEventKey* event, gpointer user_data)
  {
    printf("Plugin: The key pressed is %c\n", event->keyval);
  }

  G_MODULE_EXPORT const bool plugin_unproject_mouse
  (int x, int y)
  {
    printf("Plugin: The position of the mouse is %i %i\n", x, y);
  }

  G_MODULE_EXPORT const bool plugin_expose
  (GtkWidget *widget, GdkEventKey* event, gpointer user_data)
  {
    printf("Plugin: expose event\n");
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
