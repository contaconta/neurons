#include <gmodule.h>
#include <stdio.h>
#include <vector>
#include "Object.h"
using namespace std;

extern "C"
{
  G_MODULE_EXPORT const bool plugin_init()
  {

    printf("init plugin\n");
    return true;
  }

  G_MODULE_EXPORT const bool plugin_run(vector<Object*>& objects)
  {

    printf("Hello world !!!\n");
    return true;
  }

  G_MODULE_EXPORT const bool plugin_quit()
  {

    printf("Exit\n");
    return true;
  }
}
