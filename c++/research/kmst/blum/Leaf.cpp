/***************************************************************************
                          Leaf.cpp  -  description
                             -------------------
    begin                : Tue Sept 25 2001
    copyright            : (C) 2001 by Christian Blum
    email                : cblum@ulb.ac.be
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "Leaf.h"

Leaf::Leaf(Edge* anEdge, Vertex* aVertex){

  lEdge = anEdge;
  lVertex = aVertex;
}

Leaf::~Leaf(){
}

Vertex* Leaf::getVertex() {

  return lVertex;
}

Edge* Leaf::getEdge() {

  return lEdge;
}

Leaf* Leaf::copy() {

  Leaf* copy = new Leaf(lEdge,lVertex);
  return copy;
}
