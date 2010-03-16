/***************************************************************************
                          utilstuff.h  -  description
                             -------------------
    begin                : Tue Dec 5 2000
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

#ifndef UTILSTUFF_H
#define UTILSTUFF_H

#include "config.h"

#include <list>
#include <vector>
#include "Vertex.h"
#include "Edge.h"
#include "UndirectedGraph.h"

bool vertex_list_contains(list<Vertex*>* aList, Vertex* aVertex);
bool edge_list_contains(list<Edge*>* aList, Edge* anEdge);
bool vertice_lists_equal(list<Vertex*>* listA, list<Vertex*>* listB);
bool isConnectedTree(UndirectedGraph* aT);
double weightOfSolution(UndirectedGraph* aTree);
int* IntVectorAlloc(int n);

#endif
