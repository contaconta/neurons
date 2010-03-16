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
#include "GraphCut.h"
#include "DoubleSet.h"
#include "Configuration.h"

extern "C"
{

  G_MODULE_IMPORT void get_world_coordinates(double &wx, double &wy, double &wz,
                                             int x, int y);

  Cube_P* localCube;
  // Enumeration for the modes
  enum actions_cd    {CD_SELECTINITPOINT, CD_SELECTENDPOINT, CD_NONE};
  actions_cd         action;
  pthread_t          thread;
  //vector< GraphCut<Point3D>* > lGraphCuts;
  GraphCut<float>* graphCut;
  Image<float>* img;

  struct sPoint3d
  {
    int x;
    int y;
    int z;
  } *aPoint3d;

  // Function to be done in the thread
  static void *thread_func(void *vptr_args)
  {
    int layer_xy = -1; // TODO : FIXME
    printf("[Graph-cuts] Thread started...\n");
    if(localCube->dummy)
      {
        if(img)
          {
            // Reload image in case it is a color image that was loaded as a gray scale image by the Image class
            string fullname = img->directory + img->name;

            Configuration* config = Configuration::Instance();
            int useColorImage = 0;
            if(config)
              {
                config->retrieveIfExists("graphcuts_useColorImage",&useColorImage);
                printf("[Graph-cuts] useColorImage %d\n",useColorImage);
              }

            IplImage* iimg;
            if(useColorImage == 0)
              iimg = cvLoadImage(fullname.c_str(),0);
            else
              iimg = cvLoadImage(fullname.c_str());

            if(iimg)
              {
                graphCut->run_maxflow(iimg,1.0,1.0);
                cvReleaseImage(&iimg);
              }
            else
              printf("[Graph-cuts] Error while loading %s\n",fullname.c_str());
          }
      }
    else
      {
        if(localCube->type == "uchar"){
          graphCut->run_maxflow((Cube<uchar,ulong>*)localCube, layer_xy);
        }
        else if(localCube->type == "float"){
          graphCut->run_maxflow((Cube<float,double>*)localCube, layer_xy);
        }
        else
          printf("[Graph-cuts] Unknown cube type %s\n",localCube->type.c_str());
      }
    printf("[Graph-cuts] Thread is over...\n");
  }

  G_MODULE_EXPORT const bool plugin_init()
  {
    //printf("[Graph-cuts] initializing\n");
    action = CD_NONE;
    img = 0;
    localCube = 0;
    return true;
  }

  G_MODULE_EXPORT const bool plugin_run(vector<Object*>& objects)
  {
    //printf("[Graph-cuts] run\n");
    int x,y,z;
    graphCut = new GraphCut<float>(0);
    for(vector<Object*>::iterator itObject = objects.begin();
        itObject != objects.end(); itObject++)
      {
        string objType = (*itObject)->className();
        printf("Object class = %s\n", objType.c_str());
        if((*itObject)->className()=="Cube")
          {
            localCube = dynamic_cast<Cube_P*>((*itObject));
            printf("There is a Cube in here\n");
            graphCut->setCube(localCube);
          }
        else if((*itObject)->className()=="DoubleSet")
          {
            DoubleSet<float>* ds = dynamic_cast<DoubleSet<float>*>((*itObject));
            /*
            DoubleSet<int>* ds_indexes = new DoubleSet<int>;
            for(int i=0;i<ds_micrometers->set1.size();i++)
              {
                localCube->micrometersToIndexes3(ds_micrometers->set1[i]->coords[0],
                                                 ds_micrometers->set1[i]->coords[1],
                                                 ds_micrometers->set1[i]->coords[2],
                                                 x,y,z);
                PointDs<int>* pt = new PointDs<int>;
                pt->coords.push_back(x);
                pt->coords.push_back(y);
                pt->coords.push_back(z);
                ds_indexes->addPoint(pt,1);
              }
            for(int i=0;i<ds_micrometers->set2.size();i++)
              {
                localCube->micrometersToIndexes3(ds_micrometers->set2[i]->coords[0],
                                                 ds_micrometers->set2[i]->coords[1],
                                                 ds_micrometers->set2[i]->coords[2],
                                                 x,y,z);
                PointDs<int>* pt = new PointDs<int>;
                pt->coords.push_back(x);
                pt->coords.push_back(y);
                pt->coords.push_back(z);
                ds_indexes->addPoint(pt,2);
              }
            */
            graphCut->set_points = ds;
            printf("DoubleSet %d\n", (int)ds->set1.size());
          }
        else if((*itObject)->className()=="Image")
          {
            printf("Image\n");
            img = dynamic_cast<Image<float>*>((*itObject));
          }
      }

    if (pthread_create(&thread, NULL, thread_func, NULL) != 0)
      return false;
    else
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
    if(event->keyval == 'p'){
      printf("The action is CD_CALCULATE\n");
      if (pthread_create(&thread, NULL, thread_func, NULL) != 0)
        {
          return false;
        }
      action = CD_NONE;
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
    aPoint3d = (sPoint3d*)user_data;
    
    // FIXME : Bug when passing arguments ?
    //printf("Expose %d %d %d\n",aPoint3d->x,aPoint3d->y,aPoint3d->z);
    //aPoint3d->x = 1;
    //aPoint3d->y = -1;
    //aPoint3d->z = -1;

    //printf("Expose 2 %d %d\n", graphCut, graphCut->set_points);


    if(!graphCut->running_maxflow)
      {
        glPushMatrix();

        graphCut->draw(-1,-1,-1);
        //graphCut->draw(aPoint3d->x,aPoint3d->y,aPoint3d->z);
        //graphCut->drawSegmentation(aPoint3d->x,aPoint3d->y,aPoint3d->z);

        /*
          for(vector< GraphCut<Point3D>* >::iterator itGraphCut = lGraphCuts.begin();
          itGraphCut != lGraphCuts.end(); itGraphCut++)
          {
          (*itGraphCut)->draw(aPoint3d.x,aPoint3d.y,aPoint3d.z);
          (*itGraphCut)->drawSegmentation(aPoint3d.x,aPoint3d.y,aPoint3d.z);
          }
        */

        glPopMatrix();
      }
  }

  G_MODULE_EXPORT const bool plugin_quit()
  {
    printf("Plugin: Exit\n");
    return true;
  }



}
