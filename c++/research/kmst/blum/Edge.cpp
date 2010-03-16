/***************************************************************************
                          Edge.cpp  -  description
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

#include "Edge.h"

/* method to create an edge by providing the two vertices that are the two endpoints of the edge, and the edge weight */

Edge::Edge(Vertex* fromVertex, Vertex* toVertex, double aWeight){

	fVertex = fromVertex;
	tVertex = toVertex;
	weightValue = aWeight;
	usage = 0;
}

/* method to create an edge with two new vertices */
/* Pay attention: this method is rarely used. Usually the vertices are created, and then the edges with the method above */

Edge::Edge(){

	fVertex = new Vertex();
	tVertex = new Vertex();
	weightValue = 0.0;
	usage = 0;
}

/* destructor for an edge */
/* pay attention: the vertices that are the two endpoints of an edge still exist after destroying the edge */

Edge::~Edge(){
}

/* method to request if an edge contains 'aVertex' as one of the endpoints */

bool Edge::contains(Vertex* aVertex) {

	return ((fVertex == aVertex) || (tVertex == aVertex));
}

/* method to request the second endpoint of an edge by providing the other endpoint */

Vertex* Edge::otherVertex(Vertex* aVertex) {

	Vertex* v;
	if (fVertex == aVertex) v = tVertex;
	else v = fVertex;

	return v;
}

/* method to request the first endpoint of an edge */

Vertex* Edge::fromVertex() {

	return fVertex;
}

/* method to request the second endpoint of an edge */

Vertex* Edge::toVertex() {

	return tVertex;
}

/* method to request the edge-weight of an edge */

double Edge::weight() {

  return weightValue;
}

/* method to set the edge-weight of an edge */

void Edge::setWeight(double aWeight) {

  weightValue = aWeight;
}

/* method to print an edge */

ostream&  operator<< (ostream& os, Edge& e)
{
	os << "(" << e.fromVertex()->id() << ",";
	os << e.toVertex()->id() << ")";
	return os;
}

/* method to set the ID of an edge */

void Edge::setID(int id) {

  _id = id;
}

/* method to request the ID of an edge */
/* Pay attention: when creating a graph, all the edges should have different ids */

int Edge::id() {

  return _id;
}
