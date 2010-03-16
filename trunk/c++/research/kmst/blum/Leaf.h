/***************************************************************************
                          Leaf.h  -  description
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

#ifndef LEAF_H
#define LEAF_H

#include "config.h"


#include "Edge.h"
#include "Vertex.h"

/**
  *@author Christian Blum
  */

class Leaf {
public: 
	Leaf(Edge* anEdge, Vertex* aVertex);
	~Leaf();
	
	Edge* lEdge;
	Vertex* lVertex;

	Vertex* getVertex();
	Edge* getEdge();
	Leaf* copy();
};

#endif

