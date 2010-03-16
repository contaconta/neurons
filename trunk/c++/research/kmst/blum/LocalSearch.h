/***************************************************************************
                          LocalSearch.h  -  description
                             -------------------
    begin                : Mon Dec 4 2000
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

#ifndef LOCALSEARCH_H
#define LOCALSEARCH_H

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

class LocalSearch {
public:
	LocalSearch();
	LocalSearch(UndirectedGraph* aGraph, UndirectedGraph* aTree);
	~LocalSearch();
	
	UndirectedGraph* tree;
	UndirectedGraph* graph;
	list<Leaf*> neighborhood;
	list<Leaf*> leafs;
	
	void setTree(UndirectedGraph* aTree);
	void setGraph(UndirectedGraph* aGraph);
	void run(string tp);

private:

	LSMove* getBestMove();
	LSMove* getFirstMove();
	void generateSortedNeighborhood();
	void generateSortedLeafs();
	void adaptLeafs(LSMove* aMove);
	void adaptNeighborhood(LSMove* aMove);
};
		
#endif
