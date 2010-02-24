/***************************************************************************
                          LocalSearch.cpp  -  description
                             -------------------
    begin                : Mon Dec 4 2000
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

#include "LocalSearch.h"
#include <string>
#include "UndirectedGraph.h"
#include "Vertex.h"
#include "Edge.h"
#include "GreedyHeuristic.h"
#include "utilstuff.h"
#include "Leaf.h"

LocalSearch::LocalSearch(){
}

LocalSearch::LocalSearch(UndirectedGraph* aGraph, UndirectedGraph* aTree) {

  tree = aTree;
  graph = aGraph;
}

LocalSearch::~LocalSearch(){
}

void LocalSearch::setTree(UndirectedGraph* aTree) {

  tree = aTree;
}

void LocalSearch::setGraph(UndirectedGraph* aGraph) {

  graph = aGraph;
}

void LocalSearch::run(string type) {

  double weight = weightOfSolution(tree);
  if (type != "none") {
    bool sn_changed = true;
    generateSortedNeighborhood();
    generateSortedLeafs();
    LSMove* bestMove = NULL;
    while (sn_changed == true) {
      sn_changed = false;
      if (bestMove != NULL) {
	delete(bestMove);
      }
      if (type == "first_improvement") {
	bestMove = getFirstMove();
      }
      else {
	bestMove = getBestMove();
      }
      
      if (bestMove != NULL) {
	if (bestMove->weight_diff < 0.0) {
	  
	  tree->age = 0;
	  tree->addVertex((bestMove->in)->lVertex);
	  tree->addEdge((bestMove->in)->lEdge);
	  tree->remove((bestMove->out)->lEdge);
	  tree->remove((bestMove->out)->lVertex);
	  
	  adaptLeafs(bestMove);
	  adaptNeighborhood(bestMove);
	  
	  delete(bestMove->in);
	  delete(bestMove->out);
	  
	  weight = weight + bestMove->weight_diff;
	  sn_changed = true;
	}
      }
    }
    if (bestMove != NULL) {
      delete(bestMove);
    }
    for (list<Leaf*>::iterator aL = leafs.begin(); aL != leafs.end(); aL++) {
      delete(*aL);
    }
    leafs.clear();
    for (list<Leaf*>::iterator aL = neighborhood.begin(); aL != neighborhood.end(); aL++) {
      delete(*aL);
    }
    neighborhood.clear();
    tree->setWeight(weight);
  }
  tree = NULL;
}


LSMove* LocalSearch::getBestMove() {

  LSMove* bm = NULL;
  double weight_diff = 0.0;
  bool started = false;
  Leaf* inl = NULL;
  Leaf* outl = NULL;
  for (list<Leaf*>::iterator anIn = neighborhood.begin(); anIn != neighborhood.end(); anIn++) {
    for (list<Leaf*>::iterator anOut = leafs.begin(); anOut != leafs.end(); anOut++) {
      if (((*anIn)->lEdge)->otherVertex((*anIn)->lVertex) != (*anOut)->lVertex) {
	if (started == false) {
	  started = true;
	  inl = *anIn;
	  outl = *anOut;
	  weight_diff = ((*anIn)->lEdge)->weight() - ((*anOut)->lEdge)->weight();
	}
	else {
	  double help = ((*anIn)->lEdge)->weight() - ((*anOut)->lEdge)->weight();
	  if (help < weight_diff) {
	    inl = *anIn;
	    outl = *anOut;
	    weight_diff = help;
	  }
	}
      }
    }
  }
  if ((inl != NULL) && (outl != NULL)) {
    bm = new LSMove(inl,outl);
  }
  return bm;
}

LSMove* LocalSearch::getFirstMove() {

  LSMove* bm = NULL;
  double weight_diff = 0.0;
  bool started = false;
  Leaf* inl = NULL;
  Leaf* outl = NULL;
  bool stop = false;
  for (list<Leaf*>::iterator anIn = neighborhood.begin(); anIn != neighborhood.end(); anIn++) {
    if (stop) {
      break;
    }
    for (list<Leaf*>::iterator anOut = leafs.begin(); anOut != leafs.end(); anOut++) {
      if (((*anIn)->lEdge)->otherVertex((*anIn)->lVertex) != (*anOut)->lVertex) {
	if (started == false) {
	  started = true;
	  inl = *anIn;
	  outl = *anOut;
	  weight_diff = ((*anIn)->lEdge)->weight() - ((*anOut)->lEdge)->weight();
	  if (weight_diff < 0.0) {
	    stop = true;
	    break;
	  }
	}
	else {
	  double help = ((*anIn)->lEdge)->weight() - ((*anOut)->lEdge)->weight();
	  if (help < weight_diff) {
	    inl = *anIn;
	    outl = *anOut;
	    weight_diff = help;
	    if (weight_diff < 0.0) {
	      stop = true;
	      break;
	    }
	  }
	}
      }
    }
  }
  if ((inl != NULL) && (outl != NULL)) {
    bm = new LSMove(inl,outl);
  }
  return bm;
}

void LocalSearch::generateSortedNeighborhood() {

  for (list<Leaf*>::iterator aL = neighborhood.begin(); aL != neighborhood.end(); aL++) {
    delete(*aL);
  }
  neighborhood.clear();
  Leaf* theLeaf = NULL;
  for (list<Edge*>::iterator i = ((*graph).edges).begin(); i != ((*graph).edges).end(); i++) {
    if (!(tree->contains(*i))) {
      bool doit = false;
      if (tree->contains((*i)->fromVertex()) && (!tree->contains((*i)->toVertex()))) {
	theLeaf = new Leaf(*i,(*i)->toVertex());
	doit = true;
      }
      if (tree->contains((*i)->toVertex()) && (!tree->contains((*i)->fromVertex()))) {
	theLeaf = new Leaf(*i,(*i)->fromVertex());
	doit = true;
      }
      if (doit) {
	bool inserted = false;
	list<Leaf*>::iterator aLeaf;
	for (aLeaf = neighborhood.begin(); aLeaf != neighborhood.end(); aLeaf++) {
	  if (((*theLeaf).getEdge())->weight() >= ((*aLeaf)->getEdge())->weight()) {
	    break;
	    inserted = true;
	  }
	}
	if (inserted == true) {
	  neighborhood.insert(aLeaf,theLeaf);
	}
	else {
	  neighborhood.push_back(theLeaf);
	}
      }
    }
  }
}

void LocalSearch::generateSortedLeafs() {

  for (list<Leaf*>::iterator aL = leafs.begin(); aL != leafs.end(); aL++) {
    delete(*aL);
  }
  leafs.clear();
  for (list<Vertex*>::iterator iv = ((*tree).vertices).begin(); iv != ((*tree).vertices).end(); iv++) {
    if (((*tree).incidentEdges(*iv))->size() == 1) {
      Edge* le = *(((*tree).incidentEdges(*iv))->begin());
      Leaf* newLeaf = new Leaf(le,*iv);
      bool inserted = false;
      list<Leaf*>::iterator aLeaf;
      for (aLeaf = leafs.begin(); aLeaf != leafs.end(); aLeaf++) {
	Edge* cle = (*aLeaf)->getEdge();
	if (le->weight() >= cle->weight()) {
	  break;
	  inserted = true;
	}
      }
      if (inserted == true) {
	leafs.insert(aLeaf,newLeaf);
      }
      else {
	leafs.push_back(newLeaf);
      }
    }
  }
}

void LocalSearch::adaptLeafs(LSMove* aMove) {

  leafs.remove(aMove->out);

  Leaf* toRemove = NULL;
  for (list<Leaf*>::iterator aLeaf = leafs.begin(); aLeaf != leafs.end(); aLeaf++) {
    Vertex* other = ((aMove->in)->getEdge())->otherVertex((aMove->in)->getVertex());
    if ((*aLeaf)->getVertex() == other) {
      toRemove = *aLeaf;
      break;
    }
  }
  leafs.remove(toRemove);
  delete(toRemove);

  Vertex* ov = ((aMove->out)->getEdge())->otherVertex((aMove->out)->getVertex());
  if (((*tree).incidentEdges(ov))->size() == 1) {
    Edge* le = *(((*tree).incidentEdges(ov))->begin());
    if ((*tree).isLeave(ov)) {
      bool inserted = false;
      list<Leaf*>::iterator aLeaf;
      for (aLeaf = leafs.begin(); aLeaf != leafs.end(); aLeaf++) {
	Edge* cle = (*aLeaf)->getEdge();
	if (le->weight() >= cle->weight()) {
	  break;
	  inserted = true;
	}
      }
      if (inserted == true) {
	leafs.insert(aLeaf,new Leaf(le,ov));
      }
      else {
	leafs.push_back(new Leaf(le,ov));
      }
    }
  }
  
  Leaf* newLeaf = (aMove->in)->copy();
  Edge* le = newLeaf->getEdge();
  bool inserted = false;
  list<Leaf*>::iterator aLeaf;
  for (aLeaf = leafs.begin(); aLeaf != leafs.end(); aLeaf++) {
    Edge* cle = (*aLeaf)->getEdge();
    if (le->weight() >= cle->weight()) {
      break;
      inserted = true;
    }
  }
  if (inserted == true) {
    leafs.insert(aLeaf,newLeaf);
  }
  else {
    leafs.push_back(newLeaf);
  }
}

void LocalSearch::adaptNeighborhood(LSMove* aMove) {

  neighborhood.remove(aMove->in);

  list<Leaf*> toRemove;
  for (list<Leaf*>::iterator aLeaf = neighborhood.begin(); aLeaf != neighborhood.end(); aLeaf++) {
    Edge* moveEdge = (*aLeaf)->getEdge();
    Vertex* other = moveEdge->otherVertex((*aLeaf)->getVertex());
    if ((*tree).contains(moveEdge->fromVertex()) && (*tree).contains(moveEdge->toVertex())) {
      toRemove.push_back(*aLeaf);
    }
    else {
      if ((!(*tree).contains(moveEdge->fromVertex())) && (!(*tree).contains(moveEdge->toVertex()))) {
	toRemove.push_back(*aLeaf);
      }
      else {
	if (other == (aMove->out)->getVertex()) {
	  toRemove.push_back(*aLeaf);
	}
      }
    }
  }
  for (list<Leaf*>::iterator aLeaf = toRemove.begin(); aLeaf != toRemove.end(); aLeaf++) {
    neighborhood.remove(*aLeaf);
    delete(*aLeaf);
  }
  toRemove.clear();
  
  list<Edge*>* incidents = (*graph).incidentEdges((aMove->in)->getVertex());
  for (list<Edge*>::iterator anEdge = (*incidents).begin(); anEdge != (*incidents).end(); anEdge++) {
    if (!((*tree).contains(*anEdge))) {
      Vertex* nv = (*anEdge)->otherVertex((aMove->in)->getVertex());
      if (!((*tree).contains(nv))) {
	Leaf* nn = new Leaf(*anEdge,nv);
	bool inserted = false;
	list<Leaf*>::iterator aN;
	for (aN = neighborhood.begin(); aN != neighborhood.end(); aN++) {
	  Edge* cle = (*aN)->getEdge();
	  if ((*anEdge)->weight() >= cle->weight()) {
	    break;
	    inserted = true;
	  }
	}
	if (inserted == true) {
	  neighborhood.insert(aN,nn);
	}
	else {
	  neighborhood.push_back(nn);
	}
      }
    }
  }

  Vertex* inVertex = (aMove->out)->getVertex();
  incidents = (*graph).incidentEdges(inVertex);
  for (list<Edge*>::iterator anEdge = (*incidents).begin(); anEdge != (*incidents).end(); anEdge++) {
    if ((*tree).contains((*anEdge)->otherVertex(inVertex))) {
      Leaf* nn = new Leaf(*anEdge,inVertex);
      bool inserted = false;
      list<Leaf*>::iterator aN;
      for (aN = neighborhood.begin(); aN != neighborhood.end(); aN++) {
	Edge* cle = (*aN)->getEdge();
	if ((*anEdge)->weight() >= cle->weight()) {
	  break;
	  inserted = true;
	}
      }
      if (inserted == true) {
	neighborhood.insert(aN,nn);
      }
      else {
	neighborhood.push_back(nn);
      }
    }
  }
}
