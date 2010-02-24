/***************************************************************************
                          UndirectedGraph.h  -  description
                             -------------------
    begin                : Sat Nov 25 2000
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

#ifndef UNDIRECTEDGRAPH_H
#define UNDIRECTEDGRAPH_H

#include "config.h"

#include <list>
#include <string>
#include <map>
#include "Edge.h"
#include "Vertex.h"
#include "Random.h"


class UndirectedGraph {
public:

  /* for a description of all the methods see 'UndirectedGraph.cpp' */

	UndirectedGraph();
	UndirectedGraph(string fileName);
	UndirectedGraph(UndirectedGraph* copyGraph);
	~UndirectedGraph();
	
	void    addEdge(Edge* anEdge);
	void    addVertex(Vertex* aVertex);
	void    remove(Vertex* aVertex);
	void    remove(Edge* anEdge);
	bool    contains(Vertex* aVertex);
	bool    contains(Edge* anEdge);
	void    clear();
	void    copy(UndirectedGraph* copyGraph);
	int     numberOfEdges();
	int     numberOfVertices();
	int     degree(Vertex* aVertex);
	Vertex* vertex(int id);
	Edge*   edge(Vertex* fromVertex, Vertex* toVertex);
	bool    isAdjacent(Vertex* oneVertex, Vertex* anotherVertex);
	double  weight();
	Edge*   chooseEdge();
	Vertex* chooseVertex();
	bool    isLeave(Vertex* aVertex);
	bool    isLeave(Edge* anEdge);
	
	/* the following are the internal data structures of the graph. */
        /* they give information about adjacencies and incidences */

	list<Vertex*>* adjacentVertices(Vertex* aVertex);	
	list<Edge*>*   incidentEdges(Vertex* aVertex);
	list<Vertex*>* copyOfAdjacentVertices(Vertex* aVertex);	
	list<Edge*>*   copyOfIncidentEdges(Vertex* aVertex);
	
	void setToGridGraph();
	void setToGeneralGraph();
	void setName(string graphName);
	void setWeight(double w);
	
	friend ostream&  operator<< (ostream& os, UndirectedGraph& g);
	void Write2File(string filename);

 public:
 
	bool    isGridGraph;
	bool    isGeneralGraph;
	string  name;
	double  extraInfo;
	Random* random;
	int age;

	/* the two basic data structures of a graph are a list of vertices */
	/* and a list of edges */

	list<Vertex*> vertices;
	list<Edge*>   edges;
	
	/* the following two maps are used in order to retrieve vertices */
        /* and edges very quickly from the graph by providing their id, */
        /* which is they key of these maps */

	map<int,Vertex*> mapOfVertices;
        map<int,Edge*> mapOfEdges;
	map<Vertex*,list<Vertex*> > adjacencies;
	map<Vertex*,list<Edge*> >   incidences;

};

#endif
