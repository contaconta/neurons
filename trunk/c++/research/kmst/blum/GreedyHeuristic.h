/***************************************************************************
                          GreedyHeuristic.h  -  description
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

#ifndef GREEDYHEURISTIC_H
#define GREEDYHEURISTIC_H

#include "config.h"

#include <list>
#include <time.h>
#include "UndirectedGraph.h"
#include "Vertex.h"
#include "Edge.h"
#include "Random.h"

/**
  *@author Christian Blum
  */

class GreedyHeuristic {
public: 
	GreedyHeuristic();
	GreedyHeuristic(UndirectedGraph* aGraph);
	~GreedyHeuristic();
	
	UndirectedGraph* graph;
	list<Edge*> neighborhood;
	Random* rg;
	time_t t;
	
	void getVertexBasedGreedyHeuristicResult(UndirectedGraph* aTree, int cardinality, string ls_type);
	void getGreedyHeuristicResult(UndirectedGraph* aTree, int cardinality, string ls_type);
	void getRandomTree(UndirectedGraph* aTree, int cardinality, Edge* startE);
	void getACOTree(UndirectedGraph* aTree, map<Edge*,double>* pheromone, int cardinality);
	void getSemiRandomTree(UndirectedGraph* aTree, int cardinality, Edge* startE);
	void getGreedyTree(UndirectedGraph* aTree,int cardinality);
	UndirectedGraph* getUCTree(UndirectedGraph* t1, UndirectedGraph* t2, list<Edge*>* iset, UndirectedGraph* ug);
	UndirectedGraph* getICTree(UndirectedGraph* t1, UndirectedGraph* t2, list<Edge*>* iset, UndirectedGraph* ug);
	void getLeafs(UndirectedGraph* aTree, list<Vertex*>* leafs);
	
 private:
	
	UndirectedGraph* uniteOnCommonBase(UndirectedGraph* t1, UndirectedGraph* t2, list<Edge*>* is);
	void adaptUCNeighborhoodFor(Edge* neig, Vertex* nv, UndirectedGraph* grTree, UndirectedGraph* ugh);
	void adaptNeighborhoodFor(Edge* neig, Vertex* nv, UndirectedGraph* grTree);
	Vertex* determineNeighborNode(Edge* neig, UndirectedGraph* grTree);
	Edge* getMinNeighbor();
	Edge* getUCNeighbor(list<Edge*>* is);
	Edge* getICNeighbor(list<Edge*>* is);
	Edge* getRandomNeighbor();
	Edge* getACONeighbor(map<Edge*,double>* pheromone);
	Edge* getFirstACOEdge(map<Edge*,double>* pheromone);
	Edge* getRandomEdge();
	void generateUCNeighborhoodFor(UndirectedGraph* ugh, Edge* anEdge);
	void generateNeighborhoodFor(Vertex* aVertex);
	void generateNeighborhoodFor(Edge* anEdge);
	void removeMaxLeaf(UndirectedGraph* aTree);
	Edge* getMinEdge(list<Edge*>* nb);
	int random(int start, int end);
	void getSortedLeafs(UndirectedGraph* aTree, list<Vertex*>* leafs);
	void shrinkTree(UndirectedGraph* aTree, int cardinality);
};
		
#endif
