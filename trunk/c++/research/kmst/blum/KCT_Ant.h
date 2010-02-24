/***************************************************************************
                          KCT_Ant.h  -  description
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

#ifndef KCT_ANT_H
#define KCT_ANT_H

#include "config.h"

#include "UndirectedGraph.h"
#include "Random.h"
#include <list>
#include <vector>
#include <string>
#include <map>

/**
  *@author Christian Blum
  */

class KCT_Ant {
public: 
	KCT_Ant();
	~KCT_Ant();
	
	UndirectedGraph* baseGraph;
	double best_found_value;
	double current_value;
	double current_daemon_value;
	
	string ls;
	string phupd;
	
	UndirectedGraph* best_found_solution;
	bool best_found_flag;
	UndirectedGraph* current_solution;
	UndirectedGraph* current_daemon_solution;
	list<Edge*> neighborhood;
	Random* r_number;
	bool improved;
	map<Edge*,double>* pheromone;
	
	void setRandomGenerator(Random* rg);
	void setBaseGraph(UndirectedGraph* bGraph);
	void initialize(UndirectedGraph* bGraph, Random* rg, map<Edge*,double>* ph, string ls_spec);
	void reset();
        void restart_reset();
	void step(bool deterministic);
	void step(double cf);
	void evaluate();
	double getBestFoundValue();
	UndirectedGraph* getCurrentSolution();
	UndirectedGraph* getCurrentDaemonSolution();
	double getCurrentValue();
	double getCurrentDaemonValue();
	UndirectedGraph* getBestFoundSolution();
	void eliteAction();

private:
	Edge* getFirstEdge();
	Edge* getNextEdge();
	Edge* getDeterministicNextEdge();
	bool isAlreadyNeighbor(Edge* anEdge);
	void generateNeighborhoodFor(Edge* e);
	void adaptNeighborhoodFor(Edge* neig,Vertex* nv);
	double weightOfSolution(UndirectedGraph* aGraph);

};

#endif
