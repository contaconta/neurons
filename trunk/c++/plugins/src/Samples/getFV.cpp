#include <gtk/gtk.h>
#include <gmodule.h>
#include <stdio.h>
#include <Object.h>
#include "Cube_P.h"
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include "Image.h"
#include <sstream>

extern "C"
{

  Image<float>* localImage;
  //char fileName[] = "FIBSLICE0002_u10_all_feature_vectors";

  void printFV(int x,int y)
  {
    //string fileName = "/localhome/aurelien/Sources/EM/svm_test/Testingglcm/FIBSLICE0002_u10_all_feature_vectors";
    string fileName = "/localhome/aurelien/Sources/neurons/c++/GT001a_all_feature_vectors";
    const int patchSize = 10;

    if(x<patchSize || y<patchSize || x>localImage->width-patchSize || y>localImage->height-patchSize)
      {
        printf("No corresponding vector for (%d,%d)\n",x,y);
        return;
      }
    else
      {
        x-=patchSize;
        y-=patchSize;
      }

    //printf("filename: %s %d %d\n", fileName.c_str(),x,y);
    ifstream reader(fileName.c_str());
    if(!reader.good())
      {
        printf("Error : can not find the file: %s\n", fileName.c_str());
        return;
      }

    //int iLine_Number = y*localImage->width + x;
    int iLine_Number = x*localImage->height + y;

    string s;
    for (int i=0; i<=iLine_Number; i++) // loop 'till the desired line
      getline(reader, s);

    getline(reader, s);

    stringstream outX;
    outX << x;
    stringstream outY;
    outY << y;

    cout << "FV (" << outX.str() << "," << outY.str() << ") : " << s.c_str()[0] << endl;    
  }

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
        if((*itObject)->className()=="Image")
          {
            localImage = dynamic_cast< Image < float> * >((*itObject));
            printf("Image received\n");
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
    //printf("Plugin: The position of the mouse is %i %i\n", x, y); 
    printFV(x,y);   

    /*
    GLdouble wx, wy, wz;
    get_world_coordinates(wx, wy, wz, true);

    printf("World Coordinates: %f %f %f\n", wx, wy, wz);
    vector< float > world(3);
    world[0] = wx;
    world[1] = wy;
    world[2] = wz;

    if(localImage!=NULL){
      vector< int > indexes(3);
      localImage->micrometersToIndexes(world, indexes);
      printf("Indexes: %i %i %i\n", indexes[0], indexes[1], indexes[2]);
      printFV(indexes[0],indexes[1]);
    }
    */
  }

  G_MODULE_EXPORT const bool plugin_expose
  (GtkWidget *widget, GdkEventKey* event, gpointer user_data)
  {
    /*
    printf("Plugin: expose event\n");
    glDisable(GL_DEPTH_TEST);
    glColor3f(0.0,0.0,1.0);
    glPushMatrix();
    glutWireSphere(10,10,10);
    glPopMatrix();
    glEnable(GL_DEPTH_TEST);
    */
  }

  G_MODULE_EXPORT const bool plugin_quit()
  {
    printf("Plugin: Exit\n");
    return true;
  }

}
