/***************************************************************************
                          GreedyHeuristic.cpp  -  description
                             -------------------
    begin                : Wed Dec 12 2001
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

#include "GreedyHeuristic.h"
#include "UndirectedGraph.h"
#include "Vertex.h"
#include "Edge.h"
#include "utilstuff.h"
#include "LocalSearch.h"
#include "LocalSearchB.h"

#include <stdio.h>

GreedyHeuristic::GreedyHeuristic(){

	rg = new Random((unsigned) time(&t));
}

GreedyHeuristic::GreedyHeuristic(UndirectedGraph* aGraph) {

	graph = aGraph;
	rg = new Random((unsigned) time(&t));
}

GreedyHeuristic::~GreedyHeuristic(){

	delete(rg);
}

void GreedyHeuristic::getVertexBasedGreedyHeuristicResult(UndirectedGraph* aTree, int cardinality, string ls_type) {

  UndirectedGraph* greedyTree = new UndirectedGraph();
  bool started = false;
  for (list<Vertex*>::iterator aV = ((*graph).vertices).begin(); aV != ((*graph).vertices).end(); aV++) {
    greedyTree->clear();
    greedyTree->addVertex(*aV);
    generateNeighborhoodFor(*aV);
    for (int k = 0; k < cardinality; k++) {
      Edge* kctn = getMinNeighbor();
      Vertex* nn = determineNeighborNode(kctn,greedyTree);
      greedyTree->addVertex(nn);
      greedyTree->addEdge(kctn);
      adaptNeighborhoodFor(kctn,nn,greedyTree);
    }
    if (!(ls_type == "no")) {
      
      /* application of local search */
      if (ls_type == "leafs") {
	LocalSearch lsm(graph,greedyTree);
	lsm.run(ls_type);
      }
      else {
	if (ls_type == "cycles_leafs") {
	  LocalSearchB lsm;
	  lsm.run(graph,greedyTree);
	}
      }
      /* end local search */
      /*
      if (!isConnectedTree(greedyTree)) {
	cout << "non-connected tree" << endl;
      }
      */

    }
    greedyTree->setWeight(weightOfSolution(greedyTree));
    if (started == false) {
      started = true;
      aTree->copy(greedyTree);
    }
    else {
      if ((greedyTree->weight()) < (aTree->weight())) {
	aTree->copy(greedyTree);
      }
    }
  }
  delete(greedyTree);
}

void GreedyHeuristic::getGreedyHeuristicResult(UndirectedGraph* aTree, int cardinality, string ls_type) {

  UndirectedGraph* greedyTree = new UndirectedGraph();
  bool started = false;
  for (list<Edge*>::iterator anEdge = ((*graph).edges).begin(); anEdge != ((*graph).edges).end(); anEdge++) {
    greedyTree->clear();
    greedyTree->addVertex((*anEdge)->fromVertex());
    greedyTree->addVertex((*anEdge)->toVertex());
    greedyTree->addEdge(*anEdge);
    generateNeighborhoodFor(*anEdge);
    for (int k = 1; k < cardinality; k++) {
      Edge* kctn = getMinNeighbor();
      Vertex* nn = determineNeighborNode(kctn,greedyTree);
      greedyTree->addVertex(nn);
      greedyTree->addEdge(kctn);
      adaptNeighborhoodFor(kctn,nn,greedyTree);
    }
    if (!(ls_type == "no")) {
      
      /* application of local search */
      if (ls_type == "leafs") {
	LocalSearch lsm(graph,greedyTree);
	lsm.run(ls_type);
      }
      else {
	if (ls_type == "cycles_leafs") {
	  //cout << *greedyTree << endl;
	  LocalSearchB lsm;
	  lsm.run(graph,greedyTree);
	}
      }
      /* end local search */
      /*
      if (!isConnectedTree(greedyTree)) {
	cout << "non-connected tree" << endl;
      }
      */

    }
    greedyTree->setWeight(weightOfSolution(greedyTree));
    if (started == false) {
      started = true;
      aTree->copy(greedyTree);
    }
    else {
      if ((greedyTree->weight()) < (aTree->weight())) {
	aTree->copy(greedyTree);
      }
    }
  }
  delete(greedyTree);
}

void GreedyHeuristic::getRandomTree(UndirectedGraph* aTree, int cardinality, Edge* startE) {

  aTree->addVertex(startE->fromVertex());
  aTree->addVertex(startE->toVertex());
  aTree->addEdge(startE);
  generateNeighborhoodFor(startE);
  for (int k = 1; k < cardinality; k++) {
    Edge* newEdge = getRandomNeighbor();
    Vertex* newVertex = NULL;
    if (aTree->contains(newEdge->fromVertex())) {
      newVertex = newEdge->toVertex();
    }
    else {
      newVertex = newEdge->fromVertex();
    }
    aTree->addVertex(newVertex);
    aTree->addEdge(newEdge);
    adaptNeighborhoodFor(newEdge,newVertex,aTree);
  }
  aTree->setWeight(weightOfSolution(aTree));
}

void GreedyHeuristic::getACOTree(UndirectedGraph* aTree, map<Edge*,double>* pheromone, int cardinality) {

  Edge* startE = getFirstACOEdge(pheromone);
  aTree->addVertex(startE->fromVertex());
  aTree->addVertex(startE->toVertex());
  aTree->addEdge(startE);
  generateNeighborhoodFor(startE);
  for (int k = 1; k < cardinality; k++) {
    Edge* newEdge = getACONeighbor(pheromone);
    Vertex* newVertex = NULL;
    if (aTree->contains(newEdge->fromVertex())) {
      newVertex = newEdge->toVertex();
    }
    else {
      newVertex = newEdge->fromVertex();
    }
    aTree->addVertex(newVertex);
    aTree->addEdge(newEdge);
    adaptNeighborhoodFor(newEdge,newVertex,aTree);
  }
  aTree->setWeight(weightOfSolution(aTree));
}

void GreedyHeuristic::getSemiRandomTree(UndirectedGraph* aTree, int cardinality, Edge* startE) {

  aTree->addVertex(startE->fromVertex());
  aTree->addVertex(startE->toVertex());
  aTree->addEdge(startE);
  generateNeighborhoodFor(startE);
  for (int k = 1; k < cardinality; k++) {
    double rand = rg->next();
    Edge* newEdge = NULL;
    if (rand < 0.2) {
      newEdge = getRandomNeighbor();
    }
    else {
      newEdge = getMinNeighbor();
    }
    Vertex* newVertex = NULL;
    if (aTree->contains(newEdge->fromVertex())) {
      newVertex = newEdge->toVertex();
    }
    else {
      newVertex = newEdge->fromVertex();
    }
    aTree->addVertex(newVertex);
    aTree->addEdge(newEdge);
    adaptNeighborhoodFor(newEdge,newVertex,aTree);
  }
  aTree->setWeight(weightOfSolution(aTree));
}

void GreedyHeuristic::getGreedyTree(UndirectedGraph* greedyTree,int cardinality) {

  Edge* startE = getRandomEdge();
  greedyTree->addVertex(startE->fromVertex());
  greedyTree->addVertex(startE->toVertex());
  greedyTree->addEdge(startE);
  generateNeighborhoodFor(startE);
  for (int k = 1; k < cardinality; k++) {
    Edge* kctn = getMinNeighbor();
    Vertex* nn = determineNeighborNode(kctn,greedyTree);
    greedyTree->addVertex(nn);
    greedyTree->addEdge(kctn);
    adaptNeighborhoodFor(kctn,nn,greedyTree);
  }
  greedyTree->setWeight(weightOfSolution(greedyTree));  
}

UndirectedGraph* GreedyHeuristic::getUCTree(UndirectedGraph* t1, UndirectedGraph* t2, list<Edge*>* iset, UndirectedGraph* ug) {

  int cardinality = ((*t1).vertices).size() - 1;
  UndirectedGraph* greedyTree = new UndirectedGraph();
  Edge* minE = getMinEdge(iset);
  greedyTree->addVertex(minE->fromVertex());
  greedyTree->addVertex(minE->toVertex());
  greedyTree->addEdge(minE);
  generateUCNeighborhoodFor(ug,minE);
  for (int k = 1; k < cardinality; k++) {
    Edge* newEdge = getUCNeighbor(iset);
    Vertex* newVertex = NULL;
    if (greedyTree->contains(newEdge->fromVertex())) {
      newVertex = newEdge->toVertex();
    }
    else {
      newVertex = newEdge->fromVertex();
    }
    greedyTree->addVertex(newVertex);
    greedyTree->addEdge(newEdge);
    adaptUCNeighborhoodFor(newEdge,newVertex,greedyTree,ug);
  }
  greedyTree->setWeight(weightOfSolution(greedyTree));
  return greedyTree;
}

UndirectedGraph* GreedyHeuristic::getICTree(UndirectedGraph* t1, UndirectedGraph* t2, list<Edge*>* iset, UndirectedGraph* ug) {

  int cardinality = ((*t1).vertices).size() - 1;
  UndirectedGraph* greedyTree = new UndirectedGraph();
  //cout << "iset size: " << iset->size() << endl;
  Edge* minE = getMinEdge(iset);
  greedyTree->addVertex(minE->fromVertex());
  greedyTree->addVertex(minE->toVertex());
  greedyTree->addEdge(minE);
  generateUCNeighborhoodFor(ug,minE);
  for (int k = 2; k < ((*ug).vertices).size(); k++) {
    Edge* newEdge = getICNeighbor(iset);
    Vertex* newVertex = NULL;
    if (greedyTree->contains(newEdge->fromVertex())) {
      newVertex = newEdge->toVertex();
    }
    else {
      newVertex = newEdge->fromVertex();
    }
    greedyTree->addVertex(newVertex);
    greedyTree->addEdge(newEdge);
    adaptUCNeighborhoodFor(newEdge,newVertex,greedyTree,ug);
  }
  if ((greedyTree->vertices).size() > (cardinality + 1)) {
    shrinkTree(greedyTree,cardinality);
  }
  greedyTree->setWeight(weightOfSolution(greedyTree));
  return greedyTree;
}

UndirectedGraph* GreedyHeuristic::uniteOnCommonBase(UndirectedGraph* t1, UndirectedGraph* t2, list<Edge*>* is) {

  UndirectedGraph* ugh = new UndirectedGraph();
  for (list<Vertex*>::iterator v = ((*t1).vertices).begin(); v != ((*t1).vertices).end(); v++) {
    ugh->addVertex(*v);
  }
  for (list<Vertex*>::iterator v = ((*t2).vertices).begin(); v != ((*t2).vertices).end(); v++) {
    if (!(ugh->contains(*v))) {
      ugh->addVertex(*v);
    }
  }
  for (list<Edge*>::iterator e = ((*t1).edges).begin(); e != ((*t1).edges).end(); e++) {
    ugh->addEdge(*e);
  }
  for (list<Edge*>::iterator e = ((*t2).edges).begin(); e != ((*t2).edges).end(); e++) {
    if (!(ugh->contains(*e))) {
      ugh->addEdge(*e);
    }
    else {
      is->push_back(*e);
    }
  }
  return ugh;
}

void GreedyHeuristic::adaptUCNeighborhoodFor(Edge* neig, Vertex* nv, UndirectedGraph* grTree, UndirectedGraph* ugh) {

  neighborhood.remove(neig);
  list<Edge*> rem;
  for (list<Edge*>::iterator anEdge = neighborhood.begin(); anEdge != neighborhood.end(); anEdge++) {
    if (grTree->contains((*anEdge)->fromVertex()) && grTree->contains((*anEdge)->toVertex())) {
      rem.push_back(*anEdge);
    }
  }
  for (list<Edge*>::iterator anEdge = rem.begin(); anEdge != rem.end(); anEdge++) {
    neighborhood.remove(*anEdge);
  }
  list<Edge*>* incEdges = (*ugh).incidentEdges(nv);
  for (list<Edge*>::iterator anEdge = incEdges->begin(); anEdge != incEdges->end(); anEdge++) {
    if (!(grTree->contains((*anEdge)->fromVertex()) && grTree->contains((*anEdge)->toVertex()))) {
      neighborhood.push_back(*anEdge);
    }
  }
}

void GreedyHeuristic::adaptNeighborhoodFor(Edge* neig, Vertex* nv, UndirectedGraph* grTree) {

  neighborhood.remove(neig);
  list<Edge*> rem;
  for (list<Edge*>::iterator anEdge = neighborhood.begin(); anEdge != neighborhood.end(); anEdge++) {
    if (grTree->contains((*anEdge)->fromVertex()) && grTree->contains((*anEdge)->toVertex())) {
      rem.push_back(*anEdge);
    }
  }
  for (list<Edge*>::iterator anEdge = rem.begin(); anEdge != rem.end(); anEdge++) {
    neighborhood.remove(*anEdge);
  }
  list<Edge*>* incEdges = (*graph).incidentEdges(nv);
  for (list<Edge*>::iterator anEdge = incEdges->begin(); anEdge != incEdges->end(); anEdge++) {
    if (!(grTree->contains((*anEdge)->fromVertex()) && grTree->contains((*anEdge)->toVertex()))) {
      neighborhood.push_back(*anEdge);
    }
  }
}

Vertex* GreedyHeuristic::determineNeighborNode(Edge* neig, UndirectedGraph* grTree) {

  Vertex* result;
  if (grTree->contains(neig->fromVertex())) {
    result = neig->toVertex();
  }
  else {
    result = neig->fromVertex();
  }
  return result;
}

Edge* GreedyHeuristic::getMinNeighbor() {

  double minW;
  Edge* result;
  bool started = false;
  for (list<Edge*>::iterator i = neighborhood.begin(); i != neighborhood.end(); i++) {
    if (started == false) {
      started = true;
      result = (*i);
      minW = result->weight();
    }
    else {
      if (((*i)->weight()) < minW) {
	result = (*i);
	minW = result->weight();
      }
    }
  }
  return result;	
}

Edge* GreedyHeuristic::getUCNeighbor(list<Edge*>* is) {

  double minW;
  Edge* result;
  bool started = false;
  bool is_from_intersection = false;
  for (list<Edge*>::iterator i = neighborhood.begin(); i != neighborhood.end(); i++) {
    if (started == false) {
      started = true;
      result = (*i);
      minW = result->weight();
      if (edge_list_contains(is,(*i))) {
	is_from_intersection = true;
      }
    }
    else {
      if (edge_list_contains(is,(*i))) {
	if (is_from_intersection == false) {
	  is_from_intersection = true;
	  result = (*i);
	  minW = result->weight();
	}
	else {
	  if (((*i)->weight()) < minW) {
	    result = (*i);
	    minW = result->weight();
	  }
	}
      }
      else {
	if (((*i)->weight()) < minW) {
	  result = (*i);
	  minW = result->weight();
	}
      }
    }
  }
  return result;	
}

Edge* GreedyHeuristic::getICNeighbor(list<Edge*>* is) {

  double minW;
  Edge* result;
  bool started = false;
  bool is_from_intersectioncomp = false;
  for (list<Edge*>::iterator i = neighborhood.begin(); i != neighborhood.end(); i++) {
    if (started == false) {
      started = true;
      result = (*i);
      minW = result->weight();
      if (!(edge_list_contains(is,(*i)))) {
	is_from_intersectioncomp = true;
      }
    }
    else {
      if (is_from_intersectioncomp == false) {
	if (!(edge_list_contains(is,(*i)))) {
	  is_from_intersectioncomp = true;
	  result = (*i);
	  minW = result->weight();
	}
	else {
	  if (((*i)->weight()) < minW) {
	    result = (*i);
	    minW = result->weight();
	  }
	}
      }
      else {
	if (!(edge_list_contains(is,(*i)))) {
	  if (((*i)->weight()) < minW) {
	    result = (*i);
	    minW = result->weight();
	  }
	}
      }
    }
  }
  return result;	
}

Edge* GreedyHeuristic::getRandomNeighbor() {

  int size = neighborhood.size();
  int rand = random(0,size-1);
  vector<Edge*> vec;
  for (list<Edge*>::iterator it = neighborhood.begin(); it != neighborhood.end(); it++) {
    vec.push_back(*it);
  }
  Edge* retVal = vec[rand];
  vec.clear();
  return retVal;
}

Edge* GreedyHeuristic::getACONeighbor(map<Edge*,double>* pheromone) {

  double rand = rg->next();
  double sum = 0.0;
  list<Edge*>::iterator i;
  for (i = neighborhood.begin(); i != neighborhood.end(); i++) {
    sum = sum + (*pheromone)[*i];
  }
  double wheel = 0.0;
  i = neighborhood.begin();
  while ((wheel < rand) && (i != neighborhood.end())) {
    wheel = wheel + ((*pheromone)[*i] / sum);
    i++;
  }
  i--;
  return (*i);
}

Edge* GreedyHeuristic::getFirstACOEdge(map<Edge*,double>* pheromone) {

  double rand = rg->next();
  double sum = 0.0;
  list<Edge*>::iterator i;
  for (i = ((*graph).edges).begin(); i != ((*graph).edges).end(); i++) {
    sum = sum + (*pheromone)[*i];
  }
  double wheel = 0.0;
  i = ((*graph).edges).begin();
  while ((wheel < rand) && (i != ((*graph).edges).end())) {
    wheel = wheel + ((*pheromone)[*i] / sum);
    i++;
  }
  i--;
  return (*i);
}

Edge* GreedyHeuristic::getRandomEdge() {
  
  return (*graph).chooseEdge();
}

void GreedyHeuristic::generateUCNeighborhoodFor(UndirectedGraph* ugh, Edge* anEdge) {

  neighborhood.clear();
  list<Edge*>* ie1 = (*ugh).incidentEdges(anEdge->fromVertex());
  list<Edge*>* ie2 = (*ugh).incidentEdges(anEdge->toVertex());
  for (list<Edge*>::iterator i = (*ie1).begin(); i != (*ie1).end(); i++) {
    if (anEdge != *i) {
      neighborhood.push_back(*i);
    }
  }
  for (list<Edge*>::iterator i = (*ie2).begin(); i != (*ie2).end(); i++) {
    if (anEdge != *i) {
      neighborhood.push_back(*i);
    }
  }
}

void GreedyHeuristic::generateNeighborhoodFor(Vertex* aVertex) {

  neighborhood.clear();
  list<Edge*>* ie1 = (*graph).incidentEdges(aVertex);
  for (list<Edge*>::iterator i = (*ie1).begin(); i != (*ie1).end(); i++) {
    neighborhood.push_back(*i);
  }
}

void GreedyHeuristic::generateNeighborhoodFor(Edge* anEdge) {

  neighborhood.clear();
  list<Edge*>* ie1 = (*graph).incidentEdges(anEdge->fromVertex());
  list<Edge*>* ie2 = (*graph).incidentEdges(anEdge->toVertex());
  for (list<Edge*>::iterator i = (*ie1).begin(); i != (*ie1).end(); i++) {
    if (anEdge != *i) {
      neighborhood.push_back(*i);
    }
  }
  for (list<Edge*>::iterator i = (*ie2).begin(); i != (*ie2).end(); i++) {
    if (anEdge != *i) {
      neighborhood.push_back(*i);
    }
  }
}

void GreedyHeuristic::getLeafs(UndirectedGraph* aTree, list<Vertex*>* leafs) {

  for (list<Vertex*>::iterator iv = ((*aTree).vertices).begin(); iv != ((*aTree).vertices).end(); iv++) {
    if (((*aTree).incidentEdges(*iv))->size() == 1) {
      (*leafs).push_back(*iv);
    }
  }
}

void GreedyHeuristic::getSortedLeafs(UndirectedGraph* aTree, list<Vertex*>* leafs) {

  for (list<Vertex*>::iterator iv = ((*aTree).vertices).begin(); iv != ((*aTree).vertices).end(); iv++) {
    if (((*aTree).incidentEdges(*iv))->size() == 1) {
      Edge* le = *(((*aTree).incidentEdges(*iv))->begin());
      bool inserted = false;
      list<Vertex*>::iterator aLeaf;
      for (aLeaf = (*leafs).begin(); aLeaf != (*leafs).end(); aLeaf++) {
	Edge* cle = *(((*aTree).incidentEdges(*aLeaf))->begin());
	if (le->weight() >= cle->weight()) {
	  break;
	  inserted = true;
	}
      }
      if (inserted == true) {
	(*leafs).insert(aLeaf,*iv);
      }
      else {
	(*leafs).push_back(*iv);
      }
    }
  }
}

void GreedyHeuristic::shrinkTree(UndirectedGraph* aTree, int cardinality) {

  list<Vertex*>* leafs = new list<Vertex*>;
  getSortedLeafs(aTree,leafs);
  bool stop = false;
  while (!stop) {
    Vertex* rv = *((*leafs).begin());
    Edge* re = *(((*aTree).incidentEdges(rv))->begin());
    (*aTree).remove(re);
    (*aTree).remove(rv);
    (*leafs).pop_front();
    if (((*aTree).vertices).size() == (cardinality + 1)) {
      stop = true;
    }
    else {
      Vertex* ov = re->otherVertex(rv);
      if (((*aTree).incidentEdges(ov))->size() == 1) {
	Edge* le = *(((*aTree).incidentEdges(ov))->begin());
	if ((*aTree).isLeave(ov)) {
	  bool inserted = false;
	  list<Vertex*>::iterator aLeaf;
	  for (aLeaf = (*leafs).begin(); aLeaf != (*leafs).end(); aLeaf++) {
	    Edge* cle = *(((*aTree).incidentEdges(*aLeaf))->begin());
	    if (le->weight() >= cle->weight()) {
	      break;
	      inserted = true;
	    }
	  }
	  if (inserted == true) {
	    (*leafs).insert(aLeaf,ov);
	  }
	  else {
	    (*leafs).push_back(ov);
	  }
	}
      }
    }
  }  
  delete(leafs);
}

void GreedyHeuristic::removeMaxLeaf(UndirectedGraph* aTree) {

  double maxW;
  Vertex* vresult;
  Edge* eresult;
  Edge* he;
  bool started = false;
  for (list<Vertex*>::iterator iv = ((*aTree).vertices).begin(); iv != ((*aTree).vertices).end(); iv++) {
    int counter = 0;
    for (list<Edge*>::iterator ie = ((*aTree).edges).begin(); ie != ((*aTree).edges).end(); ie++) {
      if ((*ie)->contains(*iv)) {
	counter = counter + 1;
	he = (*ie);
      }
    }
    if (counter == 1) {
      if (started == false) {
	started = true;
	vresult = (*iv);
	eresult = he;
	maxW = eresult->weight();
      }
      else {
	if ((he->weight()) > maxW) {
	  vresult = (*iv);
	  eresult = he;
	  maxW = eresult->weight();
	}
      }
    }
  }
  (*aTree).remove(eresult);
  (*aTree).remove(vresult);
}

Edge* GreedyHeuristic::getMinEdge(list<Edge*>* nb) {

  double minW;
  Edge* result;
  bool started = false;
  for (list<Edge*>::iterator i = (*nb).begin(); i != (*nb).end(); i++) {
    if (started == false) {
      started = true;
      result = (*i);
      minW = result->weight();
    }
    else {
      if (((*i)->weight()) < minW) {
	result = (*i);
	minW = result->weight();
      }
    }
  }
  return result;
}

int GreedyHeuristic::random(int start, int end) {

  int ret_val = (int)(((double)start) + (( ((double)end) - ((double)start) + 1 ) * rg->next()));
  return ret_val;
}
