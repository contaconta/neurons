/***************************************************************************
                          Vertex.cpp  -  description
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

#include "config.h"

#include "Vertex.h"

/* method to create a vertex */
/* Pay attention: With this method the id of a vertex is set to 0. However, all the vertices of a graph are required to have different ids */

Vertex::Vertex(){

	_id = 0;
}

/* method to create a vertex by providing an id */

Vertex::Vertex(int anID) {

	_id = anID;
}

/* destrutor for a vertex */

Vertex::~Vertex(){
}

/* method to request the id of a vertex */

int Vertex::id() {

	return _id;
}

/* method to set the id of a vertex */

void Vertex::setId(int anID) {

	_id = anID;
}
