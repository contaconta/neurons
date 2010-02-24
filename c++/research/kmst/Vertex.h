/***************************************************************************
                          Vertex.h  -  description
                             -------------------
    begin                : Sun Nov 26 2000
    copyright            : (C) 2000 by Christian Blum
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

#ifndef VERTEX_H
#define VERTEX_H

#include <map>
#include <list>
#include <iostream>
#include <fstream>
#include "Edge.h"

/**
  *@author Christian Blum
  */

class Edge;

class Vertex {
public: 

	Vertex();
	Vertex(int anID);
	~Vertex();
	
	int    id();

	void setId(int anID);

public:

	int _id;
};

#endif

