/***************************************************************************
                          KCT_Ant.cpp  -  description
                             -------------------
    begin                : Mon Nov 27 2000
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

#include "KCT_Ant.h"
#include "Random.h"
#include "UndirectedGraph.h"
#include "LocalSearch.h"
#include "TabuSearch.h"
#include <list>
#include <vector>
#include <math.h>
#include "utilstuff.h"

KCT_Ant::KCT_Ant() {
}

KCT_Ant::~KCT_Ant(){

  if (current_solution != NULL) {
    delete(current_solution);
  }
  if (current_daemon_solution != NULL) {
    delete(current_daemon_solution);
  }
  if (best_found_solution != NULL) {
    delete(best_found_solution);
  }
}

void KCT_Ant::setRandomGenerator(Random* rg) {

  r_number = rg;
}

void KCT_Ant::setBaseGraph(UndirectedGraph* bGraph)  {

  baseGraph = bGraph;
}

void KCT_Ant::initialize(UndirectedGraph* bGraph, Random* rg, map<Edge*,double>* ph, string ls_spec) {

  setBaseGraph(bGraph);
  setRandomGenerator(rg);
  pheromone = ph;
  best_found_value = 0.0;
  current_value = 0.0;
  best_found_flag = false;
  current_solution = new UndirectedGraph();
  best_found_solution = new UndirectedGraph();
  current_daemon_solution = new UndirectedGraph();
  ls = ls_spec;
  improved = false;
}

void KCT_Ant::reset() {

  neighborhood.clear();
  improved = false;
  if (current_solution != NULL) {
    delete(current_solution);
  }
  current_solution = new UndirectedGraph();
  if (current_daemon_solution != NULL) {
    delete(current_daemon_solution);
  }
  current_daemon_solution = new UndirectedGraph();
}

void KCT_Ant::restart_reset() {

  neighborhood.clear();
  improved = false;
  if (current_solution != NULL) {
    delete(current_solution);
  }
  current_solution = new UndirectedGraph();
  if (current_daemon_solution != NULL) {
    delete(current_daemon_solution);
  }
  current_daemon_solution = new UndirectedGraph();  
}


void KCT_Ant::step(bool deterministic) {

  if (deterministic) {
    if (((*current_solution).vertices).size() == 0) {
      Edge* newEdge = getFirstEdge();
      current_solution->addVertex(newEdge->fromVertex());
      current_solution->addVertex(newEdge->toVertex());
      current_solution->addEdge(newEdge);
      generateNeighborhoodFor(newEdge);
    }
    else {
      Edge* newEdge = getDeterministicNextEdge();
      Vertex* newVertex = NULL;
      if (current_solution->contains(newEdge->fromVertex())) {
	newVertex = newEdge->toVertex();
      }
      else {
	newVertex = newEdge->fromVertex();
      }
      current_solution->addVertex(newVertex);
      current_solution->addEdge(newEdge);
      adaptNeighborhoodFor(newEdge,newVertex);
    }    
  }
  else {
    if (((*current_solution).vertices).size() == 0) {
      Edge* newEdge = getFirstEdge();
      current_solution->addVertex(newEdge->fromVertex());
      current_solution->addVertex(newEdge->toVertex());
      current_solution->addEdge(newEdge);
      generateNeighborhoodFor(newEdge);
    }
    else {
      Edge* newEdge = getNextEdge();
      Vertex* newVertex = NULL;
      if (current_solution->contains(newEdge->fromVertex())) {
	newVertex = newEdge->toVertex();
      }
      else {
	newVertex = newEdge->fromVertex();
      }
      current_solution->addVertex(newVertex);
      current_solution->addEdge(newEdge);
      adaptNeighborhoodFor(newEdge,newVertex);
    }
  }
}

void KCT_Ant::step(double cf) {

  double rand = r_number->next();
  if (rand < cf) {
    if (((*current_solution).vertices).size() == 0) {
      Edge* newEdge = getFirstEdge();
      current_solution->addVertex(newEdge->fromVertex());
      current_solution->addVertex(newEdge->toVertex());
      current_solution->addEdge(newEdge);
      generateNeighborhoodFor(newEdge);
    }
    else {
      Edge* newEdge = getDeterministicNextEdge();
      Vertex* newVertex = NULL;
      if (current_solution->contains(newEdge->fromVertex())) {
	newVertex = newEdge->toVertex();
      }
      else {
	newVertex = newEdge->fromVertex();
      }
      current_solution->addVertex(newVertex);
      current_solution->addEdge(newEdge);
      adaptNeighborhoodFor(newEdge,newVertex);
    }    
  }
  else {
    if (((*current_solution).vertices).size() == 0) {
      Edge* newEdge = getFirstEdge();
      current_solution->addVertex(newEdge->fromVertex());
      current_solution->addVertex(newEdge->toVertex());
      current_solution->addEdge(newEdge);
      generateNeighborhoodFor(newEdge);
    }
    else {
      Edge* newEdge = getNextEdge();
      Vertex* newVertex = NULL;
      if (current_solution->contains(newEdge->fromVertex())) {
	newVertex = newEdge->toVertex();
      }
      else {
	newVertex = newEdge->fromVertex();
      }
      current_solution->addVertex(newVertex);
      current_solution->addEdge(newEdge);
      adaptNeighborhoodFor(newEdge,newVertex);
    }
  }
}

Edge* KCT_Ant::getFirstEdge() {

  double rand = r_number->next();
  double sum = 0.0;
  list<Edge*>::iterator i;
  for (i = ((*baseGraph).edges).begin(); i != ((*baseGraph).edges).end(); i++) {
    sum = sum + (*pheromone)[*i];
  }
  double wheel = 0.0;
  i = ((*baseGraph).edges).begin();
  while ((wheel < rand) && (i != ((*baseGraph).edges).end())) {
    wheel = wheel + ((*pheromone)[*i] / sum);
    i++;
  }
  i--;
  return (*i);
}

Edge* KCT_Ant::getNextEdge() {

  double rand = r_number->next();
  double baseSum = 0.0;
  list<Edge*>::iterator i;
  for (i = neighborhood.begin(); i != neighborhood.end(); i++) {
    baseSum = baseSum + ((*pheromone)[*i] * (1 / ((*i)->weight())));
  }
  double wheel = 0.0;
  i = neighborhood.begin();
  while ((wheel < rand) && (i != neighborhood.end())) {
    double help = ((*pheromone)[*i] * (1 / ((*i)->weight())));
    wheel = wheel + (help / baseSum);
    i++;
  }
  i--;
  return (*i);
}

Edge* KCT_Ant::getDeterministicNextEdge() {

  Edge* result = NULL;
  double max = 0.0;
  for (list<Edge*>::iterator i = neighborhood.begin(); i != neighborhood.end(); i++) {
    double help = (*pheromone)[*i] * (1.0 / ((*i)->weight()));
    if (result == NULL) {
      max = help;
      result = *i;
    }
    else {
      if (help > max) {
	max = help;
	result = *i;
      }
    }
  }
  return result;
}

bool KCT_Ant::isAlreadyNeighbor(Edge* anEdge) {

  bool return_val = false;
  for (list<Edge*>::iterator i = neighborhood.begin(); i != neighborhood.end(); i++) {
    if ((*i) == anEdge) {
      return_val = true;
      break;
    }
  }
  return return_val;
}

void KCT_Ant::generateNeighborhoodFor(Edge* e) {
  
  neighborhood.clear();
  list<Edge*>* ie1 = (*baseGraph).incidentEdges(e->fromVertex());
  list<Edge*>* ie2 = (*baseGraph).incidentEdges(e->toVertex());
  for (list<Edge*>::iterator i = (*ie1).begin(); i != (*ie1).end(); i++) {
    if (e != *i) {
      neighborhood.push_back(*i);
    }
  }
  for (list<Edge*>::iterator i = (*ie2).begin(); i != (*ie2).end(); i++) {
    if (e != *i) {
      neighborhood.push_back(*i);
    }
  }
}

void KCT_Ant::adaptNeighborhoodFor(Edge* neig, Vertex* nv) {
  
  neighborhood.remove(neig);
  list<Edge*> rem;
  for (list<Edge*>::iterator anEdge = neighborhood.begin(); anEdge != neighborhood.end(); anEdge++) {
    if (current_solution->contains((*anEdge)->fromVertex()) && current_solution->contains((*anEdge)->toVertex())) {
      rem.push_back(*anEdge);
    }
  }
  for (list<Edge*>::iterator anEdge = rem.begin(); anEdge != rem.end(); anEdge++) {
    neighborhood.remove(*anEdge);
  }
  list<Edge*>* incEdges = (*baseGraph).incidentEdges(nv);
  for (list<Edge*>::iterator anEdge = incEdges->begin(); anEdge != incEdges->end(); anEdge++) {
    if (!(current_solution->contains((*anEdge)->fromVertex()) && current_solution->contains((*anEdge)->toVertex()))) {
      neighborhood.push_back(*anEdge);
    }
  }
}

double KCT_Ant::weightOfSolution(UndirectedGraph* aGraph) {

  double weight = 0.0;
  for (list<Edge*>::iterator i = ((*aGraph).edges).begin(); i != ((*aGraph).edges).end(); i++) {
    weight = weight + (*i)->weight();
  }
  return weight;
}

void KCT_Ant::evaluate() {

  current_value = weightOfSolution(current_solution);
  current_solution->setWeight(current_value);

  if (ls != "none") {
    
    current_daemon_solution->copy(current_solution);
    /* application of local search */
    if (ls == "leafs") {
      LocalSearch lsm(baseGraph,current_daemon_solution);
      lsm.run(ls);
    }
    else {
      if (ls == "tsleafs") {
	TabuSearch tso(baseGraph,current_daemon_solution);
	tso.run("first_improvement",1*((*current_solution).edges.size()));
      }
    }
    /* end local search */
	
    current_daemon_value = current_daemon_solution->weight();
  }
  else {
    current_daemon_solution->copy(current_solution);
    current_daemon_value = current_daemon_solution->weight();
  }

  if (best_found_flag == false) {
    best_found_flag = true;
    best_found_solution->copy(current_daemon_solution);
    best_found_value = current_daemon_value;
    improved = true;
  }
  else {
    if (current_daemon_value < best_found_value) {
      best_found_solution->copy(current_daemon_solution);
      best_found_value = current_daemon_value;
      improved = true;
    }
  }
}

double KCT_Ant::getBestFoundValue() {

  return best_found_value;
}

double KCT_Ant::getCurrentValue() {
  
  return current_value;
}

double KCT_Ant::getCurrentDaemonValue() {

  return current_daemon_value;
}

UndirectedGraph* KCT_Ant::getCurrentSolution() {

  return current_solution;
}

UndirectedGraph* KCT_Ant::getCurrentDaemonSolution() {

  return current_daemon_solution;
}

UndirectedGraph* KCT_Ant::getBestFoundSolution() {

  return best_found_solution;
}

void KCT_Ant::eliteAction() {

  //cout << "elite_action" << endl;
  TabuSearch tso(baseGraph,current_daemon_solution);
  tso.run("first_improvement",2*((*current_solution).edges.size()));
  current_daemon_value = current_daemon_solution->weight();
}
