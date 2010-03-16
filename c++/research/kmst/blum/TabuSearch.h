/***************************************************************************
                          TabuSearch.h  -  description
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

#ifndef TABUSEARCH_H
#define TABUSEARCH_H

#include "config.h"

#include <string>
#include "UndirectedGraph.h"
#include "Vertex.h"
#include "Edge.h"
#include "Leaf.h"
#include "LSMove.h"

/**
  *@author Christian Blum
  */

class TabuSearch {
public:
	TabuSearch();
	TabuSearch(UndirectedGraph* aGraph, UndirectedGraph* aTree);
	~TabuSearch();
	
	UndirectedGraph* tree;
	UndirectedGraph* currentSol;
	UndirectedGraph* graph;
	list<Leaf*> neighborhood;
	list<Leaf*> leafs;

	map<Edge*,int> in_list_map;
	map<Edge*,int> out_list_map;
	list<Edge*> in_list;
	list<Edge*> out_list;
	int in_length;
	int out_length;

	/* either 'first_improvement' or 'best_improvement' */
	string moveType;
	
	void setTree(UndirectedGraph* aTree);
	void setGraph(UndirectedGraph* aGraph);
	void run(string movetype, int maxiter);

private:

	void initializeTabuLists();
	void cutTabuLists();
	void adaptTabuLists(Edge* inEdge, Edge* outEdge);
	bool isTabu(Edge* inEdge, Edge* outEdge);
	LSMove* getBestMove(double bestValue, double currentValue);
	LSMove* getFirstMove(double bestValue, double currentValue);
	void generateSortedNeighborhood();
	void generateSortedLeafs();
	list<Leaf*> generateLeafs();
	void initializeLeafsAndNeighborhood();
	void adaptLeafs(LSMove* aMove);
	void adaptNeighborhood(LSMove* aMove);
	
};
		
#endif
