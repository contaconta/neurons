#!/bin/bash

# DIRECTORY=/media/neurons/C140401_B1_new/tree_1.save/
# INIT_TREE=0
# END_TREE=19
# VOLUME=/media/neurons/C140401_B1_new/volume_subsampled.nfo


DIRECTORY=/media/neurons/C240301_A1_new/tree_1.save/
INIT_TREE=0
END_TREE=30
VOLUME=/media/neurons/C240301_A1_new/volume_subsampled.nfo



for i in `seq $INIT_TREE 1 $END_TREE`;
do

NAME_ICM=`printf %s/ICM_%02i.txt $DIRECTORY $i`
NAME_EDGES=`printf %s/edges_%02i.txt $DIRECTORY $i`
NAME_POINTS=`printf %s/points_%02i.txt $DIRECTORY $i`
OUTPUT=`printf %s/tree_%i_ICM.gr $DIRECTORY $i`

echo graphFromPointsAndEdges -d -i $NAME_ICM -v $VOLUME $NAME_POINTS $NAME_EDGES $OUTPUT
graphFromPointsAndEdges -d -i $NAME_ICM -v $VOLUME $NAME_POINTS $NAME_EDGES $OUTPUT

OUTPUT=`printf %s/tree_%i.gr $DIRECTORY $i`

graphFromPointsAndEdges -v $VOLUME -d $NAME_POINTS $NAME_EDGES $OUTPUT

done