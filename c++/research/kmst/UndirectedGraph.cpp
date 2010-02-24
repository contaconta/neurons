 /***************************************************************************
                          UndirectedGraph.cpp  -  description
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

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "UndirectedGraph.h"


/* all the method are shortly desribed in the following */
/* for a good example on how to create a graph, have a look into method 'UndirectedGraph(string fileName)' that is creating a graph by reading from a file */


/* method for initialization to an empty graph */

UndirectedGraph::UndirectedGraph() {
	
  time_t t; 	
  isGridGraph=false;
  isGeneralGraph=true;
  name="default";
  extraInfo=1.0;
  random=new Random((unsigned) time(&t));
  random->next();
  vertices=list<Vertex*>(0);
  edges=list<Edge*>(0);
  age = 0;
}

/* method for initialization from a file where the input format is the one from the KCTLIB */

UndirectedGraph::UndirectedGraph(string fileName) {
	time_t t; 
	
	isGridGraph=false;
	isGeneralGraph=true;
	name=fileName;
	extraInfo = 1.0;
	random=new Random((unsigned) time(&t));
	random->next();
	vertices=list<Vertex*>(0);
	edges=list<Edge*>(0);

	long int nOfVertices;
	long int nOfEdges;
	
	FILE* inputFile=fopen(fileName.c_str(), "r");

	if (fscanf(inputFile, "%ld", &nOfVertices) < 0) {
		printf("error reading number of vertices in data file\n");
		exit(1);
	}  
	if (fscanf(inputFile, "%ld", &nOfEdges) < 0) {
		printf("error reading number of edges in data file\n");
		exit(1);
	}
	for (int i=0; i < nOfVertices; i++) {
		Vertex* v;
		v=new Vertex(i+1);
		addVertex(v);
	}
	

	#ifdef REPORT_GRAPH
	cout << endl << "Graph file: " << fileName    << endl;
	cout << "Number of vertices: " << nOfVertices << endl;
	cout << "Number of edges:    " << nOfEdges    << endl << endl;
	#endif

	double edgeWeight;
	long int node1, node2;

	for (int i=0; i < nOfEdges; i++) {
		Edge* e;
		if (fscanf(inputFile, "%ld", &node1) < 0) {
		  printf("error reading fromNode for edge in data file\n");
		  exit(1);
		}
		if (fscanf(inputFile, "%ld", &node2) < 0) {
		  printf("error reading toNode for edge in data file\n");
		  exit(1);
		}
		if (fscanf(inputFile, "%lf", &edgeWeight) < 0) {
		  printf("error reading node weight in data file\n");
		  exit(1);
		}    
		Vertex* v1=vertex(node1);
		Vertex* v2=vertex(node2);
		e=new Edge(v1,v2,edgeWeight);
		e->setID(i+1);

		#ifdef REPORT_GRAPH
		cout << "(" << node1 << "," << node2 << ")\t" << edgeWeight << endl;
		#endif

		addEdge(e);
	}

	fclose(inputFile);

	#ifdef REPORT_GRAPH
	cout << endl << endl;
	#endif

	for(list<Vertex*>::iterator i=vertices.begin(); i!=vertices.end(); i++)
	{
		if (degree(*i)==0) cout << "VERTEX " << (*i)->id() << " with DEGREE 0 !!" << endl;
	}
	age = 0;
}

/* method for initialization as the copy of another graph */

UndirectedGraph::UndirectedGraph(UndirectedGraph* copyGraph) {

	isGridGraph = copyGraph->isGridGraph;
	isGeneralGraph = copyGraph->isGeneralGraph;
	name = copyGraph->name;
	extraInfo = copyGraph->extraInfo;
	age = 0;
	time_t t;
	random=new Random((unsigned) time(&t));
	random->next();
	
	vertices=list<Vertex*>(0);
	edges=list<Edge*>(0);
	mapOfVertices.clear();
	mapOfEdges.clear();
	adjacencies.clear();
	incidences.clear();

	for(list<Vertex*>::iterator i=copyGraph->vertices.begin(); i!=copyGraph->vertices.end(); i++)
	{
		Vertex* v = (*i);
		addVertex(v);
	}
	
	for(list<Edge*>::iterator l=copyGraph->edges.begin(); l!=copyGraph->edges.end(); l++)
	{
		addEdge(*l);
	}	
}

/* method for destruction */

UndirectedGraph::~UndirectedGraph() {

	edges.clear();
	vertices.clear();
	mapOfVertices.clear();
        mapOfEdges.clear();
	adjacencies.clear();
	incidences.clear();
	delete(random);
}

/* method for adding an edge. When adding an edge, the data structures 'adjacencies' and 'incidences' are updated */
/* Pay attention: When adding an edge, the vertices that are the endpoints of that edge must be already added to the graph */

void UndirectedGraph::addEdge(Edge* anEdge) {

	edges.push_back(anEdge);

	Vertex* v1=anEdge->fromVertex();
	Vertex* v2=anEdge->toVertex();
	
	// The vertices are supposed to be already in the graph

	adjacencies[v1].push_back(v2);
	adjacencies[v2].push_back(v1);
	incidences[v1].push_back(anEdge);
	incidences[v2].push_back(anEdge);
	mapOfEdges[anEdge->id()] = anEdge;
}

/* method for adding a vertex */

void UndirectedGraph::addVertex(Vertex* aVertex) {

	vertices.push_back(aVertex);
	mapOfVertices[aVertex->id()]=aVertex;
}

/* method for removing an edge. When removing an edge, the data structures 'adjacencies' and 'incidences' are updated */

void UndirectedGraph::remove(Edge* anEdge) {

	edges.remove(anEdge);
	mapOfEdges.erase(anEdge->id());

	Vertex* v1=anEdge->fromVertex();
	Vertex* v2=anEdge->toVertex();
	
	adjacencies[v1].remove(v2);
	adjacencies[v2].remove(v1);
	incidences[v1].remove(anEdge);
	incidences[v2].remove(anEdge);
}

/* method for removing a vertex. Data structures are updated accordingly */

void UndirectedGraph::remove(Vertex* aVertex) {

	vertices.remove(aVertex);
	mapOfVertices.erase(aVertex->id());
	incidences[aVertex].clear();
	incidences.erase(aVertex);	

	// Remove the vertex from all the list of adjacencies of its neighbors
	
	for (list<Vertex*>::iterator v=adjacencies[aVertex].begin(); v != adjacencies[aVertex].end(); v++) 
	{
		adjacencies[(*v)].remove(aVertex);
	}	
	adjacencies.erase(aVertex);
}

/* method to find out if the graph contains a certain vertex */

bool UndirectedGraph::contains(Vertex* aVertex) {

  return (mapOfVertices.count(aVertex->id())!=0);
}

/* method to find out if the graph contains a certain edge */

bool UndirectedGraph::contains(Edge* anEdge) {

  return (mapOfEdges.count(anEdge->id())!=0);
}

/* method to clear the graph */

void UndirectedGraph::clear() {

	vertices.clear();
	edges.clear();
	mapOfVertices.clear();
	mapOfEdges.clear();
	adjacencies.clear();
	incidences.clear();
}

/* method to copy a graph 'copyGraph' */

void UndirectedGraph::copy(UndirectedGraph* copyGraph) {

  clear();
  isGridGraph = copyGraph->isGridGraph;
  isGeneralGraph = copyGraph->isGeneralGraph;
  name = copyGraph->name;
  extraInfo = copyGraph->extraInfo;
  age = 0;
  
  for(list<Vertex*>::iterator i=copyGraph->vertices.begin(); i!=copyGraph->vertices.end(); i++)
    {
      Vertex* v = (*i);
      addVertex(v);
    }
  
  for(list<Edge*>::iterator l=copyGraph->edges.begin(); l!=copyGraph->edges.end(); l++)
    {
      addEdge(*l);
    }	
}

/* method to request the number of edges in the graph */

int UndirectedGraph::numberOfEdges() {

	return edges.size();
}

/* method to request the number of vertices in the graph */

int UndirectedGraph::numberOfVertices() {

	return vertices.size();
}

/* method to request the degree of a vertex */

int UndirectedGraph::degree(Vertex* aVertex) {

  if (contains(aVertex)) return adjacencies[aVertex].size(); 
  
  // Otherwise...

  printf("error consulting degree of vertex %d not in the graph",aVertex->id());
  cerr << *this;  
  exit(1); 
}

/* method to obtain a vertex by providing its id */

Vertex* UndirectedGraph::vertex(int id) {
	
	return mapOfVertices[id];
}

/* method to request the edge that is connecting two vertices 'fromVertex' and 'toVertex' */
/* Pay attention: We assume that there is always only one edge that is connecting two vertices */

Edge* UndirectedGraph::edge(Vertex* fromVertex, Vertex* toVertex) {
	
	Edge* e=0;
	bool found=false;
	list<Edge*>::iterator i=edges.begin();
	while (!found && (i!=edges.end())) {
	  if ((*i)->contains(fromVertex) && (*i)->contains(toVertex)) {
	    e=(*i);
	    found=true;
	  }
	  else i++;
	}
	return e;
}

/* method to request if 'anotherVertex' is adjacent to 'oneVertex' */

bool UndirectedGraph::isAdjacent(Vertex* oneVertex, Vertex* anotherVertex) {

	bool found=false;
	
	// We traverse the shortest adjacency list of the two vertices

	int degU = degree(oneVertex);
	int degV = degree(anotherVertex);
	
	Vertex* u=((degU<degV)?(oneVertex):(anotherVertex));  
	Vertex* v=((degU<degV)?(anotherVertex):(oneVertex));  

	list<Vertex*>::iterator i=adjacencies[u].begin();
	while ((i != adjacencies[u].end()) && (!found)) {
		found=((*i) == v);
		i++;
	}	

	return found;
}

/* method to request the extraInfo, which in case of the KCT problem is the weight of the graph */
/* Pay attention: the weight of the graph is not computed internally in an automatic way. This has to be done outside. */

double UndirectedGraph::weight() {

	return extraInfo;
}

/* method to choose on of the graphs edges uniformly at random */

Edge* UndirectedGraph::chooseEdge() {

	Edge* e=0;	
	
	int start=1;
	int end=numberOfEdges();
	int r=(int)(((double)start)+((((double)end)-((double)start)+1)*random->next()));
	
	int ith=0;
	for(list<Edge*>::iterator i=edges.begin(); ((i!=edges.end()) && (ith<=r)); i++) 
	{
		ith++;
		e=(*i);
	}

	return e;
}

/* method to choose on of the graphs vertices uniformly at random */

Vertex* UndirectedGraph::chooseVertex() {

	Vertex* v=0;

	int start=1;
	int end=numberOfVertices();
	int r=(int)(((double)start)+((((double)end)-((double)start)+1)*random->next()));

	int ith=0;
	for(list<Vertex*>::iterator i=vertices.begin(); ((i!=vertices.end()) && (ith<=r)); i++) {
		ith++;
		v=(*i);
	}
	
	return v;
}

/* method to request the information if a vertex is a leaf (e.g., if it has only one incident edge) */

bool UndirectedGraph::isLeave(Vertex* aVertex)
{
	return (degree(aVertex)==1);
}

/* method to request the information if an edge is a leaf */

bool UndirectedGraph::isLeave(Edge* anEdge)
{
	return(isLeave(anEdge->fromVertex()) || isLeave(anEdge->toVertex()));
}

/* method to request a list of adjacent vertices of a vertex */
/* Pay attention: this is a pointer, so DO NOT CHANGE this list outside */

list<Vertex*>* UndirectedGraph::adjacentVertices(Vertex* aVertex) {

  return &(adjacencies[aVertex]);
}

/* method to request a list of incident edges of a vertex */
/* Pay attention: this is a pointer, so DO NOT CHANGE this list outside */

list<Edge*>* UndirectedGraph::incidentEdges(Vertex* aVertex) {

  return &(incidences[aVertex]);
}

/* method to request a list of adjacent vertices of a vertex */
/* Pay attention: this is a copy so you CAN CHANGE this list outside */

list<Vertex*>* UndirectedGraph::copyOfAdjacentVertices(Vertex* aVertex) {

  list<Vertex*>* aList = new list<Vertex*>;
  for (list<Vertex*>::iterator aV = adjacencies[aVertex].begin(); aV != adjacencies[aVertex].end(); aV++) {
    aList->push_back(*aV);
  }
  return aList;
}

/* method to request a list of incident edges of a vertex */
/* Pay attention: this is a copy so you CAN CHANGE this list outside */

list<Edge*>* UndirectedGraph::copyOfIncidentEdges(Vertex* aVertex) {

  list<Edge*>* aList = new list<Edge*>;
  for (list<Edge*>::iterator aV = incidences[aVertex].begin(); aV != incidences[aVertex].end(); aV++) {
    aList->push_back(*aV);
  }
  return aList;
}

/* method to set a boolean variable that contains the information if the graph is a gridgraph to TRUE */

void UndirectedGraph::setToGridGraph() {

	isGridGraph=true;
	isGeneralGraph=false;
}

/* method to set a boolean variable that contains the information if the graph is a  general graph (in contrast to a gridgraph) to TRUE */

void UndirectedGraph::setToGeneralGraph() {

	isGridGraph=false;
	isGeneralGraph=true;
}

/* method to set the name of a graph (not necessary) */

void UndirectedGraph::setName(string graphName) {

	name=graphName;
}

/* method to set extraInfo of a graph. In the case of the KCT problem, this is the weight */

void UndirectedGraph::setWeight(double w) {

	extraInfo=w;
}

/* method to print the graph */

ostream&  operator<< (ostream& os, UndirectedGraph& g)
{
	os << endl << "Graph file: " << g.name << endl;
	os << "Number of vertices: " << g.vertices.size() << endl;
	os << "Number of edges:    " << g.edges.size() << endl << endl;

	for(list<Edge*>::iterator i=g.edges.begin(); i!=g.edges.end() ; i++)
	{
		os << "(" << (*i)->fromVertex()->id() << ",";
		os << (*i)->toVertex()->id() << ")\t";
		os << (*i)->weight() << endl;
	}
	os << endl;	
	return os;
}

// eturetken 18.09.09. Write the graph to the specified file.
////////////////////////////////////////////////////////////////////
void UndirectedGraph::Write2File(string filename)
{
	fstream fout;
	char buffer[100];
	
	fout.open(filename.c_str(),std::ios::out);
    if (!fout.is_open()) 
	{
		cout << "Error in writing the output k-mst file." << endl;
		exit(1);
    }
	
	fout << vertices.size() << endl;
	fout << edges.size() << endl;
	for (list<Edge*>::iterator e = edges.begin(); e != edges.end(); e++)
	{
		fout << ((*e)->fromVertex())->id();
		fout << "\t";
		fout << ((*e)->toVertex())->id();
		fout << "\t";
		sprintf(buffer, "%f", (*e)->weight());
		fout << buffer << endl;
	}
	fout.close();
}
////////////////////////////////////////////////////////////////////


