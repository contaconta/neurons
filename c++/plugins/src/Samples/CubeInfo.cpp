#include <gmodule.h>
#include <stdio.h>
#include <Object.h>
#include "Cube_P.h"

extern "C"
{
  G_MODULE_EXPORT const bool plugin_init()
  {

    printf("init CubeInfo\n");
    return true;
  }

  G_MODULE_EXPORT const bool plugin_run(vector<Object*>& objects)
  {
    printf("Run CubeInfo\n");
    for(vector<Object*>::iterator itObject = objects.begin();
        itObject != objects.end(); itObject++)
      {
        if((*itObject)->className()=="Cube")
          {
            Cube_P* cube = (Cube_P*)*itObject;
            printf("Cube : %d\n",cube->cubeWidth);
          }
      }

    return true;
  }

  G_MODULE_EXPORT const bool plugin_quit()
  {

    printf("Exit\n");
    return true;
  }
}
