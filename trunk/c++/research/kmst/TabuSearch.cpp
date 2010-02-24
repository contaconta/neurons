/***************************************************************************
                          TabuSearch.cpp  -  description
                             -------------------
    begin                : Mon Oct 7 2002
    copyright            : (C) 2002 by Christian Blum
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

#include "TabuSearch.h"
#include <string>
#include "UndirectedGraph.h"
#include "Vertex.h"
#include "Edge.h"
#include "GreedyHeuristic.h"
#include "utilstuff.h"
#include "Leaf.h"

TabuSearch::TabuSearch(){
}

TabuSearch::TabuSearch(UndirectedGraph* aGraph, UndirectedGraph* aTree) {

  tree = aTree;
  graph = aGraph;
  currentSol = new UndirectedGraph();
  currentSol->copy(tree);
}

TabuSearch::~TabuSearch(){

  delete(currentSol);
}

void TabuSearch::setTree(UndirectedGraph* aTree) {

  tree = aTree;
}

void TabuSearch::setGraph(UndirectedGraph* aGraph) {

  graph = aGraph;
}

void TabuSearch::run(string movetype, int maxiter) {

  initializeTabuLists();
  initializeLeafsAndNeighborhood();

  LSMove* bestMove = NULL;

  int nic = 1;
  int init_min = 0;
  int cardinality = (*tree).edges.size();
  if (((*graph).vertices.size() - cardinality) < cardinality) {
    init_min = (*graph).vertices.size() - cardinality;
  }
  else {
    init_min = cardinality;
  }
  int init_length;
  if (init_min < ((int)(((double)(*graph).vertices.size()) / 5.0))) {
    init_length = init_min;
  }
  else {
    init_length = (int)(((double)(*graph).vertices.size()) / 5.0);
  }
  in_length = init_length;
  out_length = init_length;
  int max_length = (int)(((double)(*graph).vertices.size()) / 3.0);
  int increment = ((int)((max_length - in_length) / 4.0)) + 1;
  int max_unimpr_iters = increment;
  if (max_unimpr_iters < 100) {
    max_unimpr_iters = 200;
  }

  int iter = 1;
  
  while (iter <= maxiter) {
	  
    if ((nic % max_unimpr_iters) == 0) {
      if (in_length + increment > max_length) {
	in_length = init_length;
	out_length = init_length;
	cutTabuLists();
      }
      else {
	in_length = in_length + increment;
	out_length = out_length + increment;
      }
    }
    
    if (bestMove != NULL) {
      delete(bestMove);
    }
    if (movetype == "first_improvement") {
      bestMove = getFirstMove(tree->weight(),currentSol->weight());
    }
    else {
      bestMove = getBestMove(tree->weight(),currentSol->weight());
    }
	  
    if (bestMove != NULL) {
      
      currentSol->addVertex((bestMove->in)->lVertex);
      currentSol->addEdge((bestMove->in)->lEdge);
      currentSol->remove((bestMove->out)->lEdge);
      currentSol->remove((bestMove->out)->lVertex);
	      
      adaptLeafs(bestMove);
      adaptNeighborhood(bestMove);
      adaptTabuLists((bestMove->in)->getEdge(),(bestMove->out)->getEdge());
	      
      delete(bestMove->in);
      delete(bestMove->out);
	      
      currentSol->setWeight(currentSol->weight() + (bestMove->weight_diff));
    }
    
    if (currentSol->weight() < tree->weight()) {
      tree->copy(currentSol);
      nic = 1;
      in_length = init_length;
      out_length = init_length;
      cutTabuLists();
    }
    else {
      nic = nic + 1;
    }
    
    iter = iter + 1;
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
  in_list_map.clear();
  out_list_map.clear();
  in_list.clear();
  out_list.clear();
  tree = NULL;
}

void TabuSearch::initializeTabuLists() {

  in_list.clear();
  out_list.clear();
  for (list<Edge*>::iterator e = ((*graph).edges).begin(); e != ((*graph).edges).end(); e++) {
    in_list_map[*e] = 0;
    out_list_map[*e] = 0;
  }
}

void TabuSearch::cutTabuLists() {

  while (in_list.size() > in_length) {
    Edge* fEdge = in_list.front();
    in_list_map[fEdge] = 0;
    in_list.pop_front();
  }
  while (out_list.size() > out_length) {
    Edge* fEdge = out_list.front();
    out_list_map[fEdge] = 0;
    out_list.pop_front();
  }
}


void TabuSearch::adaptTabuLists(Edge* inEdge, Edge* outEdge) {

  if (out_list_map[inEdge] == 0) {
    if (out_list.size() == out_length) {
      Edge* anEdge = out_list.front();
      out_list.pop_front();
      out_list_map[anEdge] = 0;
    }
    out_list.push_back(inEdge);
    out_list_map[inEdge] = 1;
  }
  else {
    out_list.remove(inEdge);
    out_list.push_back(inEdge);
  }
  if (in_list_map[outEdge] == 0) {
    if (in_list.size() == in_length) {
      Edge* anEdge = in_list.front();
      in_list.pop_front();
      in_list_map[anEdge] = 0;
    }
    in_list.push_back(outEdge);
    in_list_map[outEdge] = 1;
  }
  else {
    in_list.remove(outEdge);
    in_list.push_back(outEdge);
  }
}

bool TabuSearch::isTabu(Edge* inEdge, Edge* outEdge) {

  bool return_val = false;
  if ((in_list_map[inEdge] == 1) && (out_list_map[outEdge] == 1)) {
    return_val = true;
  }
  return return_val;
}

LSMove* TabuSearch::getBestMove(double bestValue, double currentValue) {

  LSMove* bm = NULL;
  double weight_diff = 0.0;
  bool started = false;
  Leaf* inl = NULL;
  Leaf* outl = NULL;
  for (list<Leaf*>::iterator anIn = neighborhood.begin(); anIn != neighborhood.end(); anIn++) {
    for (list<Leaf*>::iterator anOut = leafs.begin(); anOut != leafs.end(); anOut++) {
      if (((*anIn)->lEdge)->otherVertex((*anIn)->lVertex) != (*anOut)->lVertex) {
	bool istabu = isTabu((*anIn)->lEdge,(*anOut)->lEdge);
	double help = ((*anIn)->lEdge)->weight() - ((*anOut)->lEdge)->weight();
	if ((istabu && ((currentValue + help) < bestValue)) || (!istabu)) {
	  if (started == false) {
	    started = true;
	    inl = *anIn;
	    outl = *anOut;
	    weight_diff = help;
	  }
	  else {
	    if (help < weight_diff) {
	      inl = *anIn;
	      outl = *anOut;
	      weight_diff = help;
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

LSMove* TabuSearch::getFirstMove(double bestValue, double currentValue) {

  LSMove* bm = NULL;
  double weight_diff = 0.0;
  bool started = false;
  Leaf* inl = NULL;
  Leaf* outl = NULL;
  bool stop = false;
  for (list<Leaf*>::iterator anIn = neighborhood.begin(); (anIn != neighborhood.end()) && (!stop); anIn++) {
    for (list<Leaf*>::iterator anOut = leafs.begin(); (anOut != leafs.end()) && (!stop); anOut++) {
      if (((*anIn)->lEdge)->otherVertex((*anIn)->lVertex) != (*anOut)->lVertex) {
	bool istabu = isTabu((*anIn)->lEdge,(*anOut)->lEdge);
	double help = ((*anIn)->lEdge)->weight() - ((*anOut)->lEdge)->weight();
	if ((istabu && ((currentValue + help) < bestValue)) || (!istabu)) {
	  if (started == false) {
	    started = true;
	    inl = *anIn;
	    outl = *anOut;
	    weight_diff = help;
	    if (weight_diff < 0.0) {
	      stop = true;
	    }
	  }
	  else {
	    if (help < weight_diff) {
	      inl = *anIn;
	      outl = *anOut;
	      weight_diff = help;
	      if (weight_diff < 0.0) {
		stop = true;
	      }
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

void TabuSearch::generateSortedNeighborhood() {

  for (list<Leaf*>::iterator aL = neighborhood.begin(); aL != neighborhood.end(); aL++) {
    delete(*aL);
  }
  neighborhood.clear();
  Leaf* theLeaf = NULL;
  for (list<Edge*>::iterator i = ((*graph).edges).begin(); i != ((*graph).edges).end(); i++) {
    if (!(currentSol->contains(*i))) {
      bool doit = false;
      if (currentSol->contains((*i)->fromVertex()) && (!currentSol->contains((*i)->toVertex()))) {
	theLeaf = new Leaf(*i,(*i)->toVertex());
	doit = true;
      }
      if (currentSol->contains((*i)->toVertex()) && (!currentSol->contains((*i)->fromVertex()))) {
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

void TabuSearch::generateSortedLeafs() {

  for (list<Leaf*>::iterator aL = leafs.begin(); aL != leafs.end(); aL++) {
    delete(*aL);
  }
  leafs.clear();
  for (list<Vertex*>::iterator iv = ((*currentSol).vertices).begin(); iv != ((*currentSol).vertices).end(); iv++) {
    if (((*currentSol).incidentEdges(*iv))->size() == 1) {
      Edge* le = *(((*currentSol).incidentEdges(*iv))->begin());
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

list<Leaf*> TabuSearch::generateLeafs() {

  list<Leaf*> shrinkLeafs;
  for (list<Vertex*>::iterator iv = ((*currentSol).vertices).begin(); iv != ((*currentSol).vertices).end(); iv++) {
    if (((*currentSol).incidentEdges(*iv))->size() == 1) {
      Edge* le = *(((*currentSol).incidentEdges(*iv))->begin());
      Leaf* newLeaf = new Leaf(le,*iv);
      shrinkLeafs.push_back(newLeaf);
    }
  }
  return shrinkLeafs;
}

void TabuSearch::initializeLeafsAndNeighborhood() {

  generateSortedNeighborhood();
  generateSortedLeafs();
}

void TabuSearch::adaptLeafs(LSMove* aMove) {

  leafs.remove(aMove->out);

  Leaf* toRemove = NULL;
  for (list<Leaf*>::iterator aLeaf = leafs.begin(); aLeaf != leafs.end(); aLeaf++) {
    Vertex* other = ((aMove->in)->getEdge())->otherVertex((aMove->in)->getVertex());
    if ((*aLeaf)->getVertex() == other) {
      toRemove = *aLeaf;
      break;
    }
  }
  if (toRemove != NULL) {
    leafs.remove(toRemove);
    delete(toRemove);
  }

  Vertex* ov = ((aMove->out)->getEdge())->otherVertex((aMove->out)->getVertex());
  if (((*currentSol).incidentEdges(ov))->size() == 1) {
    Edge* le = *(((*currentSol).incidentEdges(ov))->begin());
    if ((*currentSol).isLeave(ov)) {
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

void TabuSearch::adaptNeighborhood(LSMove* aMove) {

  neighborhood.remove(aMove->in);

  list<Leaf*> toRemove;
  for (list<Leaf*>::iterator aLeaf = neighborhood.begin(); aLeaf != neighborhood.end(); aLeaf++) {
    Edge* moveEdge = (*aLeaf)->getEdge();
    Vertex* other = moveEdge->otherVertex((*aLeaf)->getVertex());
    if ((*currentSol).contains(moveEdge->fromVertex()) && (*currentSol).contains(moveEdge->toVertex())) {
      toRemove.push_back(*aLeaf);
    }
    else {
      if ((!(*currentSol).contains(moveEdge->fromVertex())) && (!(*currentSol).contains(moveEdge->toVertex()))) {
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
    if (!((*currentSol).contains(*anEdge))) {
      Vertex* nv = (*anEdge)->otherVertex((aMove->in)->getVertex());
      if (!((*currentSol).contains(nv))) {
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
    if ((*currentSol).contains((*anEdge)->otherVertex(inVertex))) {
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

