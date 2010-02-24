TARGET = aco ec kcp gg-ew-generator ts
CXXFLAGS = -ansi -O3
TCAOBJS = Edge.o GreedyHeuristic.o LocalSearch.o Random.o Timer.o UndirectedGraph.o Vertex.o Edge.o utilstuff.o Leaf.o LSMove.o TabuSearch.o LocalSearchB.o
TSOBJS = Edge.o GreedyHeuristic.o LocalSearch.o Random.o Timer.o UndirectedGraph.o Vertex.o Edge.o utilstuff.o Leaf.o LSMove.o LocalSearchB.o
GENOBJS = UndirectedGraph.o Vertex.o Edge.o Random.o
ACOOBJS = Edge.o LocalSearch.o Random.o Timer.o UndirectedGraph.o Vertex.o Edge.o utilstuff.o KCT_Ant.o Leaf.o LSMove.o TabuSearch.o
KCPOBJS = Edge.o GreedyHeuristic.o LocalSearch.o Random.o Timer.o UndirectedGraph.o Vertex.o Edge.o utilstuff.o Leaf.o LSMove.o LocalSearchB.o

all: ${TARGET}

ec: ec_kct.cpp $(TCAOBJS)
	${CXX} ${CXXFLAGS} -o $@ $^

ts: ts_kct.cpp $(TSOBJS)
	${CXX} ${CXXFLAGS} -o $@ $^

gg-ew-generator: gg-ew-generator.cpp $(GENOBJS)
	${CXX} ${CXXFLAGS} -o $@ $^

aco: aco_kct.cpp $(ACOOBJS)
	${CXX} ${CXXFLAGS} -o $@ $^

kcp: kcp.cpp $(KCPOBJS)
	${CXX} ${CXXFLAGS} -o $@ $^

clean:
	@rm -f *~ *.o ${TARGET} core
