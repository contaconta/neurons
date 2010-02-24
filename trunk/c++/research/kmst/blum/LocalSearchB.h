// FILE: LocalSearchB.h

//#define VERBOSE
//#define PURGE

#include "config.h"

#include "UndirectedGraph.h"
#include <list>

enum MovementType  { STATIC_MOVE=-1, DYNAMIC_MOVE=1 };

class Movement {
public:

	Movement (Edge* edgeInsert, Edge* edgeRemove);
	Movement();
	~Movement();

	void assign (Edge* edgeInsert, Edge* edgeRemove, MovementType type=STATIC_MOVE, list<Edge*>* cycle=0);
	void applyTo (UndirectedGraph* ktree);
	
	friend ostream& operator<< (ostream& os, Movement& move);
	
public:

	MovementType _type;
	Edge*        _edgeRemove;
	Edge*        _edgeInsert;
	list<Edge*>* _cycle;			// Just in case of static movements

};



class LocalSearchB {
public:

	LocalSearchB();
	~LocalSearchB();

	void run(UndirectedGraph* G, UndirectedGraph* ktree);

private: // attributes

	UndirectedGraph* _G;
	UndirectedGraph* _ktree;
	
	double           _cost;
	list<Edge*>*     _incidents;
	list<Edge*>*     _leaves;
	list<Edge*>*     _remIncidents;
	list<Edge*>*     _insIncidents;
	list<Edge*>*     _remLeaves;
	list<Edge*>*     _insLeaves;

private: // methods

	list<Edge*>*     computeIncidentEdges();
	list<Edge*>*     computeLeaveEdges();
	Edge*            maxEdgeInCycleWhenAdding (Edge* e, list<Edge*>* cycle);
	Edge*            maxEdgeInLeavesWhenAdding (Edge* e);
	list<Movement*>* neighborhood ();
	double           differentialCost (Movement* move);
	bool             chooseBestNeighbor (Movement* move);
	void             computeNeighborhoodUpdate(Movement* move);
	void             adaptNeighborhood (Movement* move);
	pair<Edge*,Edge*>* adjacentsInCycle (list<Edge*>* cycle, Edge* e);
	
	friend class Movement;
};
