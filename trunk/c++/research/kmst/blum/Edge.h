/***************************************************************************
                          Edge.h  -  description
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

#ifndef EDGE_H
#define EDGE_H

#include "config.h"

#include "Vertex.h"
#include <iostream>
#include <fstream>
#include <stdio.h>


/**
  *@author Christian Blum
  */

class Vertex;

class Edge {
public:

	Edge(Vertex* fromVertex, Vertex* toVertex, double aWeight=0.0);
	Edge();
	~Edge();
	
	bool    contains(Vertex* aVertex);
	Vertex* otherVertex(Vertex* aVertex);
	Vertex* fromVertex();
	Vertex* toVertex();
	double  weight();
	void    setWeight(double aWeight);
        void    setID(int id);
        int     id();

	friend ostream&  operator<< (ostream& os, Edge& e);

public:

	Vertex* fVertex;
	Vertex* tVertex;
	double  weightValue;
	int     usage;
	int     _id;
};

#endif
