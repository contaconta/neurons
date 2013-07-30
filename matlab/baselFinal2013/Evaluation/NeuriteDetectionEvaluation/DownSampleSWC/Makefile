CC=c++
CFLAGS=-I. -lm -lc++
DEPS = SWCFileIO.h SWCTreeDownsampler.h SWCTreeSmoother.h

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

DownSample.exe: main.o SWCFileIO.o SWCTreeDownsampler.o SWCTreeSmoother.o
	c++ -o DownSample.exe main.o SWCFileIO.o SWCTreeDownsampler.o SWCTreeSmoother.o -I.  