#!/usr/bin/env python

import threading
import time

class ExperimentThread(threading.Thread):
    k = 0
    def __init__(self, k):
        self.k = k
        threading.Thread.__init__(self, name="ComputeThread-%i" % (k))

    def run(self):
        print "Thread %i started!" % self.k
        time.sleep(10)
        print "Thread %i finished!" % self.k

if __name__ == '__main__':

    allThreads = []
    for k in range(1,40):
        print "Adding thread%i" % k
        allThreads.append(ExperimentThread(k));

    nMaxConcurrentThreads = 40
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
        time.sleep(.2)

