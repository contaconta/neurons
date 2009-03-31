#include <gmodule.h>
#include <stdio.h>
#include <Object.h>
#include "Image.h"

extern "C"
{
  G_MODULE_EXPORT const bool plugin_init()
  {

    printf("init Image plugin\n");
    return true;
  }

  G_MODULE_EXPORT const bool plugin_run(vector<Object*>& objects)
  {
    printf("Image plugin running\n");
    for(vector<Object*>::iterator itObject = objects.begin();
        itObject != objects.end(); itObject++)
      {
        if((*itObject)->className()=="Image")
          {
            Image< float >* img = (Image<float>*)*itObject;
            printf("img - width: %d, height: %d\n",img->width, img->height);

            for(int i=0;i<img->width/4;i++)
              for(int j=0;j<img->height/4;j++)
                {
                  img->pixels[j][i]=0;
                  //img->texels[i*img->height+j]=0;
                }
            img->reloadTexture();
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
