#include "ActorSet.h"

ActorSet::ActorSet()
{
  actors.resize(0);
  actors.push_back(new Axis());
}

void ActorSet::addActorFromPath(string path){

  printf("Here = %s actors size=%i\n", path.c_str(), actors.size());

  string ext = getExtension(path);

  if( ext=="nfo" || ext=="nfc" ||
      ext=="tif" || ext=="TIF" ||
      ext=="tiff" || ext=="TIFF")
    {
      Cube_P* cube = CubeFactory::load(path);
      cube->load_texture_brick(0,0);
      cube->v_r = 1.0;
      cube->v_g = 1.0;
      cube->v_b = 1.0;
      cube->v_draw_projection = false;
      actors.push_back(cube);
    }
  if(ext == "gr" || ext=="GR"){
    actors.push_back(GraphFactory::load(path));
  }
  if(ext == "cl" || ext=="CL"){
    actors.push_back(CloudFactory::load(path));
  }
  if(ext == "asc" || ext=="ASC"){
    actors.push_back(new Neuron(path));
  }
  if(ext == "swc" || ext=="SWC"){
    actors.push_back(new SWC(path));
  }
  if ((ext == "jpg") || (ext == "png"))  {
    actors.push_back(new Image<float>(path,0));
  }


}
