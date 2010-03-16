#include <gtk/gtk.h>
#include <gmodule.h>
#include <stdio.h>
#include <Object.h>
#include "Cube_P.h"
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include "CubeLiveWire.h"
#include "Point3D.h"
#include "Cloud.h"
#include <pthread.h>
#include <signal.h>

extern "C"
{

  G_MODULE_IMPORT void get_world_coordinates(double &wx, double &wy, double &wz,
                                             int x, int y);

  Cube_P* localCube;
  // Enumeration for the modes
  enum actions_cd    {CD_SELECTINITPOINT, CD_SELECTENDPOINT, CD_NONE};
  actions_cd         action;
  Point3D*           initPoint;
  Point3D*           endPoint;
  CubeLiveWire*      cubeLiveWire;
  // Cloud<Point3D>*    shortestPath;
  Graph<Point3D, EdgeW<Point3D> >*    shortestPath;
  vector< Graph<Point3D, EdgeW<Point3D> >* > savedPaths;
  Cloud<Point3D>*    boundary;
  pthread_t          thread;
  pthread_mutex_t    mutexBoundary;
  bool               drawBoundary;
  int                last_x;
  int                last_y;
  vector< int >      iInit(3);
  vector<int >       iEnd(3);
  vector< Cloud< Point3D>* > clouds;


  // Function to be done in the thread
  static void *thread_func(void *vptr_args)
  {
    localCube->micrometersToIndexes(initPoint->coords, iInit);
    printf("  computing the shortest path from %i %i %i and everywhere\n",
           iInit[0], iInit[1], iInit[2]);
    cubeLiveWire->computeDistances(iInit[0],iInit[1],iInit[2]);
  }

  G_MODULE_EXPORT const bool plugin_init()
  {
    printf("init cubeLiveWire\n");
    initPoint = NULL;
    endPoint  = NULL;
    cubeLiveWire == NULL;
    action = CD_NONE;
    return true;
  }

  G_MODULE_EXPORT const bool plugin_run(vector<Object*>& objects)
  {
    action = CD_NONE;
    // initPoint = new Point3D(-15.558399, 16.319525, -18.224996);
    // endPoint  = new Point3D(-15.308799,  7.543466, -17.414998);
    initPoint = NULL;
    endPoint  = NULL;
    drawBoundary  = true;
    boundary  = new Cloud<Point3D>();
    // Cube<float, double>* cubeAguet = new Cube<float, double>
      // ("/media/neurons/cut2/aguet_4.00_2.00.nfo");
    // Cube<float, double>* cubeAguetTheta = new Cube<float, double>
      // ("/media/neurons/cut2/aguet_4.00_2.00_theta.nfo");
    // Cube<float, double>* cubeAguetPhi = new Cube<float, double>
      // ("/media/neurons/cut2/aguet_4.00_2.00_phi.nfo");
    // DistanceDijkstraColorAngle* djkc = new DistanceDijkstraColorAngle
      // (cubeAguet, cubeAguetTheta, cubeAguetPhi);
    Cube<uchar, ulong>* cubeD = new Cube<uchar, ulong>("/media/neurons/cut2/cut2.nfo");

    DistanceDijkstraColor* djkc = new DistanceDijkstraColor
      (cubeD);

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
            // DistanceDijkstraColor* djkc = new DistanceDijkstraColor
              // (localCube);
            cubeLiveWire = new CubeLiveWire(localCube, djkc);
            // exit(0);
            // printf("Cube : %d\n",cube->cubeWidth);
          }
      }
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
      // printf("The action is CD_CALCULATE\n");
      // }
      action = CD_NONE;
    }
  }


  G_MODULE_EXPORT const bool plugin_unproject_mouse
  (int x, int y)
  {
    double wx, wy, wz;
    get_world_coordinates(wx, wy, wz, x, y);
    printf("The world coordinates are %f %f %f\n", wx, wy, wz);
    printf("Plugin: The position of the mouse is %i %i\n", x, y);
    switch(action){
    case CD_NONE:
      printf("UnprojectMouse: The action is CD_NONE\n");
      break;
    case CD_SELECTINITPOINT:
      printf("UnprojectMouse: The action is CD_SELECTINITPOINT\n");
      initPoint = new Point3D(wx, wy, wz);
      action = CD_SELECTENDPOINT;
      if ((initPoint != NULL) ){
        printf("  computing the path... \n");
        if(thread != NULL){
          printf("Killing the thread\n");
          if(pthread_cancel(thread)!=0){
            printf("  error killing the thread\n");
          } else {
            printf("  thread killed succesfully\n");
          }
        }
        if (pthread_create(&thread, NULL, thread_func, NULL) != 0)
          {
            return false;
          }
      }
      action = CD_NONE;
      break;
    case CD_SELECTENDPOINT:
      printf("UnprojectMouse: The action is CD_SELECTENDPOINT\n");
      endPoint = new Point3D(wx, wy, wz);
      vector<int > iInit(3); //indexes initial point
      vector<int > iEnd(3);  //indexes end point
      localCube->micrometersToIndexes(initPoint->coords, iInit);
      localCube->micrometersToIndexes(endPoint->coords, iEnd);
      printf("  computing the shortest path between %i %i %i and %i %i %i\n",
             iInit[0], iInit[1], iInit[2],
             iEnd[0],  iEnd[1],  iEnd[2]);
      shortestPath = cubeLiveWire->findShortestPathG(iInit[0], iInit[1], iInit[2],
                                                    iEnd[0],  iEnd[1],  iEnd[2]
                                                    );
      if(shortestPath!=NULL){
        savedPaths.push_back(shortestPath);
      }
      action = CD_NONE;
      break;
    }
  }

  G_MODULE_EXPORT const bool plugin_expose
  (GtkWidget *widget, GdkEventKey* event, gpointer user_data)
  {
    // printf("init: %i, end %i, boundary: %i\n", initPoint, endPoint, boundary);
    glDisable(GL_DEPTH_TEST);
    double wx, wy, wz;
    get_world_coordinates(wx, wy, wz, last_x, last_y);
    vector<int >    pt(3); //indexes initial point
    vector< float > wCoords(3);  //world coordinates
    wCoords[0] = wx; wCoords[1] = wy; wCoords[2] = wz;
    localCube->micrometersToIndexes(wCoords, pt);

    if( (pt[0] >= 0) && (pt[0] < localCube->cubeWidth) &&
        (pt[1] >= 0) && (pt[1] < localCube->cubeHeight) &&
        (pt[2] >= 0) && (pt[2] < localCube->cubeDepth) ){
      if(cubeLiveWire->visited[pt[2]][pt[1]][pt[0]] == true){
        glColor3f(1.0,0.0,0.0);
      } else {
        glColor3f(0.0,0.0,1.0);
      }
      glPushMatrix();
      glTranslatef(wx,wy,wz);
      glutSolidSphere(1.0,10.0,10.0);
      glPopMatrix();
    }

    if(initPoint != NULL){
      glColor3f(0.0,1.0,0.0);
      initPoint->draw();
    }
    if(endPoint != NULL){
      glColor3f(1.0,0.0,0.0);
      endPoint->draw();
    }
    if(shortestPath!=NULL){
      shortestPath->cloud->v_r = 1.0;
      shortestPath->cloud->v_g = 1.0;
      shortestPath->cloud->v_radius = 0.05;
      shortestPath->draw();
    }

    for(int i = 0; i < savedPaths.size(); i++){
      savedPaths[i]->cloud->v_r = 1.0;
      savedPaths[i]->cloud->v_g = 1.0;
      savedPaths[i]->cloud->v_b = 0.0;
      savedPaths[i]->cloud->v_radius = 0.05;
      savedPaths[i]->draw();
    }

    if(drawBoundary && (boundary!=NULL) && (!cubeLiveWire->pathFound)){
        pthread_mutex_lock(&mutexBoundary);
        boundary->draw();
        pthread_mutex_unlock(&mutexBoundary);
    }
    glEnable(GL_DEPTH_TEST);
  }

  G_MODULE_EXPORT const bool plugin_quit()
  {
    printf("Plugin: Exit\n");
    return true;
  }

  G_MODULE_EXPORT const bool plugin_motion_notify
  (GtkWidget *widget, GdkEventKey* event, gpointer user_data)
  {
    int x, y;
    GdkModifierType state;
    gdk_window_get_pointer (event->window, &x, &y, &state);
    last_x = x;
    last_y = y;

    double wx, wy, wz;
    get_world_coordinates(wx, wy, wz, last_x, last_y);
    vector< int   > iEnd(3); //indexes initial point
    vector< float > wCoords(3);  //world coordinates
    wCoords[0] = wx; wCoords[1] = wy; wCoords[2] = wz;
    localCube->micrometersToIndexes(wCoords, iEnd);

    shortestPath = cubeLiveWire->findShortestPathG(iInit[0], iInit[1], iInit[2],
                                                  iEnd[0],  iEnd[1],  iEnd[2]);
  }



}
