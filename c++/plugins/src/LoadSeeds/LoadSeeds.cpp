#include <gtk/gtk.h>
#include <gmodule.h>
#include <stdio.h>
#include <Object.h>
#include "Cube_P.h"
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include "Point3D.h"
#include "Cloud.h"
#include <pthread.h>
#include "DoubleSet.h"
#include "utils.h"

extern "C"
{

  G_MODULE_IMPORT void get_world_coordinates(double &wx, double &wy, double &wz,
                                             int x, int y);

  Cube_P* localCube;
  // Enumeration for the modes
  enum actions_cd    {CD_SELECTINITPOINT, CD_SELECTENDPOINT, CD_NONE};
  actions_cd         action;
  pthread_t          thread;

  struct sPoint3d
  {
    int x;
    int y;
    int z;
  } *aPoint3d;

  // Function to be done in the thread
  static void *thread_func(void *vptr_args)
  {

  }

  G_MODULE_EXPORT const bool plugin_init()
  {
    printf("init LoadSeeds\n");
    action = CD_NONE;      
    return true;
  }

  G_MODULE_EXPORT const bool plugin_run(vector<Object*>& objects)
  {
    printf("Plugin: run\n");
    int ix,iy,iz;
    DoubleSet<float>* ds = 0;

    for(vector<Object*>::iterator itObject = objects.begin();
        itObject != objects.end(); itObject++)
      {
        string objType = (*itObject)->className();
        printf("Object class = %s\n", objType.c_str());

        if((*itObject)->className()=="Cube")
          {
            localCube = dynamic_cast<Cube_P*>((*itObject));
            printf("There is a Cube in here\n");
          }
        else if((*itObject)->className()=="DoubleSet")
          {
            printf("There is a DoubleSet in here\n");
            ds = dynamic_cast< DoubleSet<float>* > ((*itObject));
          }
      }

    if(ds== 0 || localCube == 0)
      {
        printf("Error : no DoubleSet or Cube\n");
        return false;
      }
    
    // Load files
    string dir = "/home/alboot/Sources/EM/Superpixels/predict/";
    vector<string> files;
    get_files_in_dir(dir, files);

    const int sampleSpace = 8; // FIXME : should be a param
    const int NB_LABELS = 3;
    int label;
    double pb[NB_LABELS];

    // -2*sampleSpace as we don't have the borders of the image
    int width = localCube->cubeWidth - (sampleSpace*2);
    int height = localCube->cubeHeight - (sampleSpace*2);
    
    printf("w %d h %d\n",width, height);

    float mx,my,mz;
    int z = 0;
    for(vector<string>::iterator itFiles = files.begin();
        itFiles != files.end(); itFiles++)
      {
        printf("File %s",itFiles->c_str());
        if(getExtension(*itFiles)!="predict")
          continue;

        string filename = dir + *itFiles;
        ifstream ifs(filename.c_str());

        if(ifs)
          {
            // jump first line
            char buffer[256];
            ifs.getline(buffer,256);
            printf("Buffer %s\n",buffer);

            //for(int x=-width/2;x<width/2;x++)
            //  for(int y=-height/2;y<height/2;y++)
            for(int x=0;x<width;x++)
              for(int y=0;y<height;y++)
                {
                  if(ifs.fail())
                    break;

                  ifs >> label;
                  for(int i=0;i<NB_LABELS;i++)
                    {
                      ifs >> pb[i];
                      //printf("pb %f", pb[i]);
                    }

                  if(x%8 != 0 || y%8 != 0)
                    continue;

                  localCube->indexesToMicrometers3(x,y,z,
                                                   mx,my,mz);

                  //if(x < 10 && y < 10)
                  //  printf("%d %d %d %d %d %d %d\n",x,y,z,ix,iy,iz,label);

                  PointDs<float>* pt = new PointDs<float>;
                  pt->indexes.push_back(x);
                  pt->indexes.push_back(y);
                  pt->indexes.push_back(z);
                  pt->coords.push_back(mx);
                  pt->coords.push_back(my);
                  pt->coords.push_back(mz);
                  if(label == -1)
                    ds->addPoint(pt,1);
                  else if(label == 1)
                    ds->addPoint(pt,2);
                  else if(label != 2)
                    printf("Error : unknown label\n");
                }
          }
        z++;
      }

    printf("DoubleSet Loaded\n");
    /*
    if (pthread_create(&thread, NULL, thread_func, NULL) != 0)
      return false;
    else
      return true;
    */
    return true;
  }

  G_MODULE_EXPORT const bool plugin_key_press_event
  (GtkWidget *widget, GdkEventKey* event, gpointer user_data)
  {
    if(event->keyval == 'i'){
        printf("The action is CD_SELECTINITPOINT\n");
        action = CD_SELECTINITPOINT;
    }
    if(event->keyval == 'o'){
      printf("The action is CD_SELECTENDPOINT\n");
        action = CD_SELECTENDPOINT;
    }
  }


  G_MODULE_EXPORT const bool plugin_unproject_mouse
  (int x, int y)
  {
    double wx, wy, wz;
    get_world_coordinates(wx, wy, wz, x, y);
    // printf("The world coordinates are %f %f %f\n", wx, wy, wz);
    // printf("Plugin: The position of the mouse is %i %i\n", x, y);
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
    //aPoint3d = (sPoint3d*)user_data;
  }

  G_MODULE_EXPORT const bool plugin_quit()
  {
    printf("Plugin: Exit\n");
    return true;
  }



}
