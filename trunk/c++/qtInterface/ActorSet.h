#ifndef ACTORSET_H_
#define ACTORSET_H_

#include "VisibleE.h"
#include "CubeFactory.h"
#include "Axis.h"
#include "CloudFactory.h"
#include "GraphFactory.h"
#include "Neuron.h"
#include "Image.h"
#include "SWC.h"

class ActorSet
{

public:

  vector< VisibleE* > actors;

  ActorSet();

  void addActorFromPath(string path);

};




#endif



