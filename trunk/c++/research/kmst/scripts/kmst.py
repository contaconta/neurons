#!/usr/bin/env python

import threading
import time
import sys
import os

class ExperimentThread(threading.Thread):
    k = 0
    graphNameFile=""
    graphName=""
    directory=""
    cubeName = ""

    def __init__(self, k, nameGraphFile, nameGraph, nameCube):
        self.k = k
        self.graphNameFile = nameGraphFile
        self.graphName = nameGraph
        self.cubeName = nameCube
        self.directory = os.path.dirname(nameGraphFile)
        threading.Thread.__init__(self, name="ComputeThread-%i" % (k))

    def run(self):
        print "Thread %i started!" % self.k
        command = "~/workspace/viva/research/kmst/blum/aco -i " + self.graphNameFile + " -cardb " + `self.k` + " -t 10 -m " + self.directory + "/out"
        print command
        os.system(command)
        command = "kMSTFileToGraph " + self.graphName + " " +  self.directory + "/out" + `self.k` + ".mst "  + " " + self.directory + "/out_" + `self.k` + ".gr"
        os.system(command)
        outName = "sol_%03i.swc" %  self.k
        command = "diademGraphToSWC " + self.directory + "/out_" + `self.k` + ".gr " + self.cubeName + " " + self.directory + "/" + outName
        os.system(command)
        print command


        # time.sleep(1)
        # print "Thread %i finished!" % self.k

############################### MAIN #######################################################
if __name__ == '__main__':

    nameGraph = sys.argv[1]
    nameCube  = sys.argv[2]
    # print nameGraph
    print os.path.dirname(nameGraph)
    nameDirectory = os.path.dirname(nameGraph)+'/kmst/'
    if (os.path.isdir(nameDirectory)!=True):
        os.mkdir(nameDirectory)
    print nameDirectory
    nameGraphFile = nameDirectory + "/" + os.path.basename(nameGraph) + ".txt"
    command = "kMSTGraphToFile " + nameGraph + " " + nameGraphFile
    os.system(command)

    allThreads = []
    for k in range(1,160):
        print "Adding thread%i" % k
        allThreads.append(ExperimentThread(k, nameGraphFile, nameGraph, nameCube));

    nMaxConcurrentThreads = 2
    nThreadsDefault = len(threading.enumerate())

    print "Threads init. NThreadsDefault %i" % nThreadsDefault

    nThreadsRun = 0
    while(nThreadsRun < len(allThreads)):
        if(len(threading.enumerate()) - nThreadsDefault) < nMaxConcurrentThreads:
            allThreads[nThreadsRun].start()
            nThreadsRun = nThreadsRun + 1
            print "Numnber of threads running: %i\n" % \
                (len(threading.enumerate()) - nThreadsDefault)
        # Just create threads each .2 seconds
        time.sleep(.1)

