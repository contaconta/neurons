/***************************************************************************
                          utilstuff.cpp  -  description
                             -------------------
    begin                : Tue Dec 5 2000
    copyright            : (C) 2000 by Christian Blum
    email                : chr_blum@hotmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "utilstuff.h"
#include <list>
#include "Vertex.h"
#include "Edge.h"
#include "UndirectedGraph.h"

bool vertex_list_contains(list<Vertex*>* aList, Vertex* aVertex) {

  bool result = false;
  for (list<Vertex*>::iterator i = (*aList).begin(); i != (*aList).end(); i++) {
    if ((*i) == aVertex) {
      result = true;
      break;
    }
  }
  return result;
}

bool edge_list_contains(list<Edge*>* aList, Edge* anEdge) {
  
  bool result = false;
  for (list<Edge*>::iterator i = (*aList).begin(); i != (*aList).end(); i++) {
    if ((*i) == anEdge) {
      result = true;
      break;
    }
  }
  return result;
}

bool vertice_lists_equal(list<Vertex*>* listA, list<Vertex*>* listB) {

  bool result = true;
  if ((listA->size()) != (listB->size())) {
    result = false;
  }
  else {
    list<Vertex*>::iterator iter = (*listA).begin();
    while (iter != (*listA).end()) {
      result = vertex_list_contains(listB,(*iter));
      iter++;
      if (result == false) {
	iter = (*listA).end();
      }
    }
  }
  return result;
}

bool isConnectedTree(UndirectedGraph* aT) {

  bool result = true;
  for (list<Edge*>::iterator i = ((*aT).edges).begin(); i != ((*aT).edges).end(); i++) {
    Vertex* fv = (*i)->fromVertex();
    Vertex* tv = (*i)->toVertex();
    if (!((aT->contains(fv)) && (aT->contains(tv)))) {
      result = false;
    }
  }
  unsigned int ds;
  if (result == true) {
    ds = 0;
    for (list<Vertex*>::iterator v = ((*aT).vertices).begin(); v != ((*aT).vertices).end(); v++) {
      for (list<Edge*>::iterator e = ((*aT).edges).begin(); e != ((*aT).edges).end(); e++) {
	if ((*e)->contains(*v)) {
	  ds = ds + 1;
	}
      }
    }
  }
  if (ds != ((((*aT).vertices).size() - 1) * 2)) {
    result = false;
  }
  return result;
}

double weightOfSolution(UndirectedGraph* aTree) {

  double weight = 0.0;
  for (list<Edge*>::iterator i = ((*aTree).edges).begin(); i != ((*aTree).edges).end(); i++) {
    weight = weight + (*i)->weight();
  }
  return weight;
}

int* IntVectorAlloc(int n) {

  int* v = (int*) malloc( n * sizeof(int) );
  return v;
}

