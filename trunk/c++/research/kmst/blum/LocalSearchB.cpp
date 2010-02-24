// FILE: LocalSearchB.cpp

#include "LocalSearchB.h"
#include <map>


//------------------------------------------------------
// CLASS LocalSearchB
//------------------------------------------------------ 


LocalSearchB::LocalSearchB ()
{
	_incidents    = new list<Edge*>(0);
	_leaves       = new list<Edge*>(0);
	_remIncidents = new list<Edge*>(0);
	_insIncidents = new list<Edge*>(0);
	_remLeaves    = new list<Edge*>(0);
	_insLeaves    = new list<Edge*>(0);
}


LocalSearchB::~LocalSearchB ()
{
}


void LocalSearchB::run (UndirectedGraph* G, UndirectedGraph* ktree)
{
	_G = G;
	_ktree = ktree;
	
	// Adaptive structures (initialization)
	
	computeIncidentEdges();

	#ifdef PURGE
	cout << endl << endl << "INCIDENTS: ";
	for(list<Edge*>::iterator i=_incidents->begin(); i!=_incidents->end(); i++)
	{
		cout << *(*i);	
	}
	cout << endl;
	#endif

	computeLeaveEdges();

	#ifdef PURGE
	cout << endl << "LEAVES: ";
	for(list<Edge*>::iterator i=_leaves->begin(); i!=_leaves->end(); i++)
	{
		cout << *(*i);	
	}
	cout << endl;
	#endif
	
	// LS Procedure
	
	Movement* move = new Movement();
	int       iter=1;
		
	#ifdef PURGE	
	cout << endl << endl;
	cout << "Iteration #" << iter << endl;
	cout << "KTREE: " << *_ktree << endl;
	#endif
	
	bool toBeContinued = chooseBestNeighbor(move);
	
	while (toBeContinued)
	{	
		#ifdef PURGE
		cout << endl << "NEXT MOVE: " << *move << endl;
		#endif 
		
		move->applyTo(_ktree);	
		adaptNeighborhood(move);

		#ifdef PURGE
		cout << endl << endl;
		cout << "Iteration #" << ++iter << endl;
		cout << "KTREE: " << *_ktree << endl;
		#endif
		
		toBeContinued = chooseBestNeighbor(move);
	}		
}
		

// PRIVATE METHODS
//----------------- 


list<Edge*>* LocalSearchB::computeIncidentEdges()
{	
	map<Edge*,bool> lookup;
	list<Edge*>*    copy = new list<Edge*>(0);
	
	_incidents->clear();
	
	for (list<Vertex*>::iterator i=_ktree->vertices.begin(); i!=_ktree->vertices.end(); i++)
	{
		list<Edge*>* inc=_G->incidentEdges(*i);

		for (list<Edge*>::iterator j=inc->begin(); j!=inc->end(); j++)
		{
			if ((!_ktree->contains(*j)) && (lookup.count(*j)==0))
			{
				_incidents->push_back(*j);
				copy->push_back(*j);
				lookup[*j]=true;
	}	}	} 
	
	return (copy);
}


list<Edge*>* LocalSearchB::computeLeaveEdges()
{
	list<Edge*>* copy = new list<Edge*>(0);
	
	_leaves->clear();
	
	for (list<Edge*>::iterator i=_ktree->edges.begin(); i!=_ktree->edges.end(); i++)
	{
		if (_ktree->isLeave(*i)) 
		{
			_leaves->push_back(*i);
			copy->push_back(*i);
	}	}
	
	return (copy);
}


bool LocalSearchB::chooseBestNeighbor (Movement* move)
{	
	bool   better_found = false;
	double best_improvement  = 0.0;		

	list<Movement*>* candidates = neighborhood();

	for (list<Movement*>::iterator i=candidates->begin(); i!=candidates->end(); i++)
	{
		Movement* m=(*i);
		
		if (m->_edgeInsert != m->_edgeRemove)
		{
			double improvement = differentialCost(m);
			
			if (improvement < best_improvement) 
			{
				better_found = true;
				if (m->_type == STATIC_MOVE) move->assign(m->_edgeInsert, m->_edgeRemove, m->_type, m->_cycle);
				else move->assign(m->_edgeInsert, m->_edgeRemove, m->_type);
				best_improvement = improvement;
	}	}	}
	
	if (better_found) computeNeighborhoodUpdate(move);
	
	for (list<Movement*>::iterator i=candidates->begin(); i!=candidates->end(); i++)
	{
		if (*i!=move) delete(*i);
	}

	delete(candidates);
	return better_found;
}


void LocalSearchB::computeNeighborhoodUpdate(Movement* move)
{
	// Calculate how the neighborhood have to be adapted

	_remIncidents->clear();
	_insIncidents->clear();
	_remLeaves->clear();
	_insLeaves->clear();

	Vertex* fromVertexIns = move->_edgeInsert->fromVertex();
	Vertex* fromVertexRem = move->_edgeRemove->fromVertex();
	Vertex* toVertexIns = move->_edgeInsert->toVertex();
	Vertex* toVertexRem = move->_edgeRemove->toVertex();
	
	Vertex* auxVertex;

	if (move->_type == DYNAMIC_MOVE)
	{
		// -------------------
		// DYNAMIC MOVEMENT
		// -------------------

		//--- Adapt leaves ---

		_remLeaves->push_back(move->_edgeRemove);
		_insLeaves->push_back(move->_edgeInsert);

		// extra case #1: The adjancent edge to the removed one, now becomes a leave

		if ((_ktree->degree(fromVertexRem)==2) || (_ktree->degree(toVertexRem)==2))
		{
			if (_ktree->degree(fromVertexRem)==2) auxVertex=fromVertexRem; else auxVertex=toVertexRem;
			
			if ((auxVertex!=fromVertexIns) && (auxVertex!=toVertexIns)) { 

				list<Edge*>* inc = _ktree->incidentEdges(auxVertex);
				list<Edge*>::iterator i = inc->begin();
				if (*i == move->_edgeRemove) i++;

				_insLeaves->push_back(*i);
			}
		}

		// extra case #2: The adjacent edge to the inserted one, now is not leave anymore

		bool fromLeave = (_ktree->contains(fromVertexIns) && (_ktree->degree(fromVertexIns)==1));
		bool toLeave   = (_ktree->contains(toVertexIns)   && (_ktree->degree(toVertexIns)==1));
		
		if (fromLeave || toLeave)
		{
			if (fromLeave) auxVertex=fromVertexIns; else auxVertex=toVertexIns;

			list<Edge*>* inc = _ktree->incidentEdges(auxVertex);
			list<Edge*>::iterator i = inc->begin();
			
			_remLeaves->push_back(*i);
		}

		//--- Adapt incident edges ---

		_remIncidents->push_back(move->_edgeInsert);
		_insIncidents->push_back(move->_edgeRemove);
		
		Vertex* newVertex;
		
		if (!_ktree->contains(fromVertexIns)) newVertex = fromVertexIns; else newVertex = toVertexIns;
		
		list<Edge*>* inc = _G->incidentEdges(newVertex);
		for(list<Edge*>::iterator i=inc->begin(); i!=inc->end(); i++)
			if ((!_ktree->contains(*i)) && (move->_edgeInsert!=*i)) _insIncidents->push_back(*i);

		Vertex* oldLeave;		
		if (_ktree->isLeave(fromVertexRem)) oldLeave = fromVertexRem; else oldLeave = toVertexRem;
		
		inc = _G->incidentEdges(oldLeave);
		for(list<Edge*>::iterator i=inc->begin(); i!=inc->end(); i++) 
			if ((move->_edgeRemove!=*i) && (!_ktree->contains((*i)->otherVertex(oldLeave)))) 
				_remIncidents->push_back(*i);

	} else {  

		// ------------------
		// STATIC MOVEMENT
		// ------------------

		//--- Adapt leaves ---

		pair<Edge*,Edge*>* adjIns = adjacentsInCycle(move->_cycle,move->_edgeInsert); 

		Edge* adjFromSideIns = adjIns->first;
		Edge* adjToSideIns   = adjIns->second;
		
		pair<Edge*,Edge*>* adjRem = adjacentsInCycle(move->_cycle,move->_edgeRemove); 
		Edge* adjFromSideRem = adjRem->first;
		Edge* adjToSideRem   = adjRem->second;
		
		Vertex* fromVertexIns  = move->_edgeInsert->fromVertex(); 
		Vertex* fromVertex2Ins = adjFromSideIns->otherVertex(fromVertexIns);

		Vertex* toVertexIns  = move->_edgeInsert->toVertex(); 
		Vertex* toVertex2Ins = adjToSideIns->otherVertex(toVertexIns);

		Vertex* fromVertexRem  = move->_edgeRemove->fromVertex(); 
		Vertex* fromVertex2Rem = adjFromSideRem->otherVertex(fromVertexRem);

		Vertex* toVertexRem  = move->_edgeRemove->toVertex(); 
		Vertex* toVertex2Rem = adjToSideRem->otherVertex(toVertexRem);

		// Edge closing cycle connects to leave edges

		if (_ktree->isLeave(adjFromSideIns) && _ktree->isLeave(adjToSideIns))
		{
			_remLeaves->push_back(adjFromSideIns);
			_remLeaves->push_back(adjToSideIns);

			if (move->_edgeRemove == adjFromSideIns)
			{
				_insLeaves->push_back(move->_edgeInsert);

				if (_ktree->degree(fromVertex2Ins)==2) 
				{
					list<Edge*>* inc = _ktree->incidentEdges(fromVertex2Ins);
					list<Edge*>::iterator i = inc->begin();
					if ((*i)==adjFromSideIns) i++;
					_insLeaves->push_back(*i);
				} 

			} else if (move->_edgeRemove == adjToSideIns) {

				_insLeaves->push_back(move->_edgeInsert);

				if (_ktree->degree(toVertex2Ins)==2) 
				{
					list<Edge*>* inc = _ktree->incidentEdges(toVertex2Ins);
					list<Edge*>::iterator i = inc->begin();
					if ((*i)==adjToSideIns) i++;
					_insLeaves->push_back(*i);
				} 

			} else {	// move->_edgeRemove non adjacent

				if (_ktree->degree(fromVertexRem)==2) _insLeaves->push_back(adjFromSideRem);
				if (_ktree->degree(toVertexRem)==2)   _insLeaves->push_back(adjToSideRem);
			}


		// Edge closing the cycle connects one edge leave and one internal edge (CASE 1)

		} else if (_ktree->isLeave(adjFromSideIns) && !(_ktree->isLeave(adjToSideIns))) {

			_remLeaves->push_back(adjFromSideIns);

			if (move->_edgeRemove == adjFromSideIns)
			{
				_insLeaves->push_back(move->_edgeInsert);

				if (_ktree->degree(fromVertex2Ins)==2) 
				{
					list<Edge*>* inc = _ktree->incidentEdges(fromVertex2Ins);
					list<Edge*>::iterator i = inc->begin();
					if ((*i)==adjFromSideIns) i++;
					_insLeaves->push_back(*i);
				}

			} else if (move->_edgeRemove == adjToSideIns) {

				if (_ktree->degree(toVertex2Ins)==2) 
				{
					list<Edge*>* inc = _ktree->incidentEdges(toVertex2Ins);
					list<Edge*>::iterator i = inc->begin();
					if ((*i)==adjToSideIns) i++;
					_insLeaves->push_back(*i);
				}
				
			} else {	// move->_edgeRemove non adjacent

				if (_ktree->degree(fromVertexRem)==2) _insLeaves->push_back(adjFromSideRem);
				if (_ktree->degree(toVertexRem)==2)   _insLeaves->push_back(adjToSideRem);			
			}


		// Edge closing the cycle connects one edge leave and one internal edge (CASE 2)

		} else if (!(_ktree->isLeave(adjFromSideIns)) && _ktree->isLeave(adjToSideIns)) {

			_remLeaves->push_back(adjToSideIns);

			if (move->_edgeRemove == adjToSideIns)
			{
				_insLeaves->push_back(move->_edgeInsert);

				if (_ktree->degree(toVertex2Ins)==2) 
				{
					list<Edge*>* inc = _ktree->incidentEdges(toVertex2Ins);
					list<Edge*>::iterator i = inc->begin();
					if ((*i)==adjToSideIns) i++;
					_insLeaves->push_back(*i);
				}

			} else if (move->_edgeRemove == adjFromSideIns) {

				if (_ktree->degree(fromVertex2Ins)==2) 
				{
					list<Edge*>* inc = _ktree->incidentEdges(fromVertex2Ins);
					list<Edge*>::iterator i = inc->begin();
					if ((*i)==adjFromSideIns) i++;
					_insLeaves->push_back(*i);
				}
				
			} else {	// move->_edgeRemove non adjacent

				if (_ktree->degree(fromVertexRem)==2) _insLeaves->push_back(adjFromSideRem);			
				if (_ktree->degree(toVertexRem)==2)   _insLeaves->push_back(adjToSideRem);
			}


		// Edge closing the cycle connects internal edges (no leaves)

		} else if (!(_ktree->isLeave(adjFromSideIns)) && !(_ktree->isLeave(adjToSideIns))) {

			if ((_ktree->degree(fromVertexRem)==2) && (adjFromSideRem!=move->_edgeInsert)) 
				_insLeaves->push_back(adjFromSideRem);
			if ((_ktree->degree(toVertexRem)==2) && (adjToSideRem!=move->_edgeInsert))
				_insLeaves->push_back(adjToSideRem);
		} 			

		//--- Adapt incident edges 

		_remIncidents->push_back(move->_edgeInsert);
		_insIncidents->push_back(move->_edgeRemove);
	}
	
	// Eliminate duplicated elements
	
	_insIncidents->sort(); _insIncidents->unique();
	_remIncidents->sort(); _remIncidents->unique();

	_insLeaves->sort(); _insLeaves->unique();
	_remLeaves->sort(); _remLeaves->unique();	

	#ifdef PURGE
	cout << endl;
	cout << "\tREM INCIDENTS: ";
	for(list<Edge*>::iterator i=_remIncidents->begin(); i!=_remIncidents->end(); i++)
	{
		cout << *(*i);	
	}
	cout << endl;

	cout << "\tINS INCIDENTS: ";
	for(list<Edge*>::iterator i=_insIncidents->begin(); i!=_insIncidents->end(); i++)
	{
		cout << *(*i);	
	}
	cout << endl;

	cout << "\tREM LEAVES: ";
	for(list<Edge*>::iterator i=_remLeaves->begin(); i!=_remLeaves->end(); i++)
	{
		cout << *(*i);	
	}
	cout << endl;

	cout << "\tINS LEAVES: ";
	for(list<Edge*>::iterator i=_insLeaves->begin(); i!=_insLeaves->end(); i++)
	{
		cout << *(*i);	
	}
	cout << endl;
	#endif
}


void LocalSearchB::adaptNeighborhood (Movement* move)
{
	// NOTE: At this point, the move has already been applied to the ktree !!!
	
	_cost = _cost + differentialCost(move);

	// Update the incident edges
	
	for(list<Edge*>::iterator i=_remIncidents->begin(); i!=_remIncidents->end(); i++)
	{
		_incidents->remove(*i);
	}
	

	for(list<Edge*>::iterator j=_insIncidents->begin(); j!=_insIncidents->end(); j++)
	{
		_incidents->push_back(*j);
	}
	
	// Update the leaves
	
	for(list<Edge*>::iterator k=_remLeaves->begin(); k!=_remLeaves->end(); k++)
	{
		_leaves->remove(*k);
	}
	
	for(list<Edge*>::iterator l=_insLeaves->begin(); l!=_insLeaves->end(); l++)
	{
		_leaves->push_back(*l);
	}
	
	_incidents->sort();
	_incidents->unique();
	_leaves->sort();
	_leaves->unique();
	
	#ifdef PURGE
	cout << endl << endl;
	cout << "INCIDENTS: ";
	for(list<Edge*>::iterator i=_incidents->begin(); i!=_incidents->end(); i++)
	{
		cout << *(*i);	
	}
	cout << endl;

	cout << "LEAVES: ";
	for(list<Edge*>::iterator i=_leaves->begin(); i!=_leaves->end(); i++)
	{
		cout << *(*i);	
	}
	cout << endl;
	#endif
}


Edge* LocalSearchB::maxEdgeInCycleWhenAdding(Edge* e, list<Edge*>* cycle)
{
	// Edges and degrees of the ktree are copied

	for (list<Edge*>::iterator i=_ktree->edges.begin(); i!=_ktree->edges.end(); i++)
		cycle->push_back(*i);

	map<Vertex*,int> degree;
	for (list<Vertex*>::iterator i=_ktree->vertices.begin(); i!=_ktree->vertices.end(); i++) 
		degree[*i]=_ktree->degree(*i);

	// The new edge is added to the copy of the ktree

	cycle->push_back(e);
	degree[e->fromVertex()]++;
	degree[e->toVertex()]++;

	bool  cycleFound=false;
	int   numDeleted;
	Edge* maxEdge;				

	while(!cycleFound)
	{
		numDeleted=0;	
		maxEdge=((cycle->size()>0)?(cycle->front()):(0));

		list<Edge*>* rem = new list<Edge*>(0);

		for(list<Edge*>::iterator i=cycle->begin(); i!=cycle->end(); i++)
		{
			// If it is a leaf-edge, it is removed (does not belong to a cycle)

			if ((degree[(*i)->fromVertex()]==1)||(degree[(*i)->toVertex()]==1))
			{
				degree[(*i)->fromVertex()]--;
				degree[(*i)->toVertex()]--;
				rem->push_back(*i);
				numDeleted++;

			} else if ((*i)->weight()>=maxEdge->weight()) maxEdge=(*i);

			cycleFound=(numDeleted==0);
		}

		for (list<Edge*>::iterator j=rem->begin(); j!=rem->end(); j++) cycle->remove(*j);
		delete(rem);
	}

	return maxEdge;
}

	
Edge* LocalSearchB::maxEdgeInLeavesWhenAdding(Edge* e)
{
	Vertex* extremeVertex;
	
	if (_ktree->contains(e->fromVertex())) extremeVertex=e->fromVertex();
	else if (_ktree->contains(e->toVertex())) extremeVertex=e->toVertex();
	
	list<Edge*>* newLeaves = new list<Edge*>(0);
	newLeaves->push_back(e);
	
	for(list<Edge*>::iterator i=_leaves->begin(); i!=_leaves->end(); i++)
	{
		bool sharedExtreme1 = (((*i)->fromVertex()==extremeVertex) && (_ktree->degree(extremeVertex)==1));
		bool sharedExtreme2 = (((*i)->toVertex()==extremeVertex)   && (_ktree->degree(extremeVertex)==1)); 

		if ((!sharedExtreme1) && (!sharedExtreme2)) newLeaves->push_back(*i);
	}
	
	/*
	bool containsU=_ktree->contains(u);
	bool containsV=_ktree->contains(v);

	// The edge (also one of its vertices) are temporally added to the _ktree

	if (!containsU) _ktree->addVertex(u); 
	if (!containsV) _ktree->addVertex(v);
	_ktree->addEdge(e);

	// The highest-weighted edge from the edge-leaves is selected

	list<Edge*>* leaves=computeLeaveEdges();
	leaves->remove(e);
	*/
	
	Edge* maxEdge=newLeaves->front();				
	for (list<Edge*>::iterator i=newLeaves->begin(); i!=newLeaves->end(); i++)
	{
		if ((*i)->weight()>=maxEdge->weight()) maxEdge=(*i);
	}

	/*
	// The _ktree is restablished to its original composition

	if (!containsU) _ktree->remove(u);
	if (!containsV) _ktree->remove(v);
	_ktree->remove(e);
	*/
	delete(newLeaves);
	return maxEdge;
}


list<Movement*>* LocalSearchB::neighborhood()
{
	list<Movement*>* candidates=new list<Movement*>(0);

	// All the possible movements are generated

	for(list<Edge*>::iterator i=_incidents->begin(); i!=_incidents->end(); i++)
	{
		#ifdef VERBOSE
		cout << endl;
		cout << "Treating incident edge (" << (*i)->fromVertex()->id() << ",";
		cout << (*i)->toVertex()->id() << ")" << endl;
		#endif

		Movement* move=new Movement(*i,*i);

		Vertex* u=(*i)->fromVertex();
		Vertex* v=(*i)->toVertex();

		bool staticMove=(_ktree->contains(u) && _ktree->contains(v));

		// Edge with both nodes incident in the ktree (static move)

		if (staticMove) {
		
			list<Edge*>* cycle = new list<Edge*>(0);
			Edge* maxEdge=maxEdgeInCycleWhenAdding(*i,cycle);

			move->assign(*i,maxEdge,STATIC_MOVE,cycle);
			candidates->push_back(move);

		// Edge with one node incident in the ktree (dynamic move)

		} else {

			Edge* maxEdge=maxEdgeInLeavesWhenAdding(*i);
			move->assign(*i,maxEdge,DYNAMIC_MOVE);
			candidates->push_back(move);		
	}	}

	candidates->sort();
	candidates->unique();
	return candidates;	
}


double LocalSearchB::differentialCost(Movement* move)
{
	return (move->_edgeInsert->weight() - move->_edgeRemove->weight());
}


pair<Edge*,Edge*>* LocalSearchB::adjacentsInCycle(list<Edge*>* cycle, Edge* e)
{
	pair<Edge*,Edge*>* adjEdges = new pair<Edge*,Edge*>;
	Vertex* from = e->fromVertex();
	Vertex* to   = e->toVertex();
	
	bool foundFrom = false;
	bool foundTo   = false;
			
	for(list<Edge*>::iterator i=cycle->begin(); ((i!=cycle->end()) && (!(foundFrom && foundTo))); i++)
	{
		if (*i!=e)
		{
			if (((*i)->fromVertex()==from)||((*i)->toVertex()==from)) 
			{
				adjEdges->first=*i;
				foundFrom = true;
			
			} else if (((*i)->fromVertex()==to)||((*i)->toVertex()==to)) {
			
				adjEdges->second=*i;
				foundTo = true;
	}	}	}

	return adjEdges;
}


//------------------------------------------------------
// CLASS Movement
//------------------------------------------------------ 


Movement::Movement (Edge* edgeInsert, Edge* edgeRemove)
{
	_type=STATIC_MOVE;
	_edgeInsert=edgeInsert;
	_edgeRemove=edgeRemove;
}


Movement::Movement ()
{
	_type=STATIC_MOVE;
}


Movement::~Movement ()
{
}


void Movement::assign (Edge* edgeInsert, Edge* edgeRemove, MovementType type, list<Edge*>* cycle)
{
	_type=type;
	_edgeInsert=edgeInsert;
	_edgeRemove=edgeRemove;
	_cycle=cycle;
}


void Movement::applyTo (UndirectedGraph* ktree)
{
	if (_edgeInsert != _edgeRemove)
	{
		// Add an edge

		Vertex* u = _edgeInsert->fromVertex();
		Vertex* v = _edgeInsert->toVertex();

		if (!ktree->contains(u)) ktree->addVertex(u);
		if (!ktree->contains(v)) ktree->addVertex(v);
		ktree->addEdge(_edgeInsert);	

		// Remove an edge

		u = _edgeRemove->fromVertex();
		v = _edgeRemove->toVertex();

		bool isLeaveU = ktree->isLeave(u);
		bool isLeaveV = ktree->isLeave(v);

		ktree->remove(_edgeRemove);
		if (isLeaveU) ktree->remove(u);
		if (isLeaveV) ktree->remove(v);
	}	
}


ostream& operator<< (ostream& os, Movement& move)
{
	os << "i" << *move._edgeInsert << "r" << *move._edgeRemove;
	if (move._type==STATIC_MOVE) os << " STATIC"; else os << " DYNAMIC";
	return os;
}

