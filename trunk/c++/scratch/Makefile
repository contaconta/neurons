SHELL=/bin/bash

CFLAGSOPENCV = $(shell pkg-config --cflags opencv) $(shell pkg-config --cflags gsl)
CFLAGSGTK    = $(shell pkg-config --cflags gtk+-2.0)
CFLAGSGTKEXT    = $(shell pkg-config --cflags gtkglext-1.0 pangoft2)
CFLAGSGL = -I/usr/include/

LDFLAGSOPENCV = $(shell pkg-config --libs opencv) $(shell pkg-config --libs gsl)
LDFLAGSGTK    = $(shell pkg-config --libs gtk+-2.0)
LDFLAGSGTKEXT = $(shell pkg-config --libs gtkglext-1.0 pangoft2)
LDFLAGSOPENGL = -L/usr -L/usr/lib -L/usr/X11R6/lib -lGLEW -lglut -lGLU -lGL

LDFLAGS =  -g -pg $(LDFLAGSOPENCV) $(LDFLAGSGTK) $(LDFLAGSGTKEXT) $(LDFLAGSOPENGL) -L../lib
CFLAGS = -c -g  -pg -I../core $(CFLAGSOPENCV) $(CFLAGSGTK) $(CFLAGSGTKEXT) $(CFLAGSGL)

## Each source file will create a program with the same name. The only thing to
#  touch in the makefile
SRCS = scratch.cpp test_vector.cpp
OBJS = $(SRCS:.cpp=.o)
EXEC = $(SRCS:.cpp=)

all: depend lib $(EXEC)

lib:
	make -C ../ depend
	make -C ../ lib

$(OBJS):%.o:%.cpp
	g++ $(CFLAGS) $< -o $@

$(EXEC):$(OBJS) lib
	g++ $(LDFLAGS) $@.o -o $@ -lneseg

print:
	echo $(OBJS)

depend:
	makedepend -Y -I../core *.cpp  2>/dev/null


clean :
	rm -rf objects bin

# DO NOT DELETE

scratch.o: ../core/CubeFactory.h ../core/Cube.h ../core/image.h
scratch.o: ../core/polynomial.h ../core/utils.h ../core/Visible.h
scratch.o: ../core/Cube_P.h ../core/Cube_data.h ../core/Cube_display.h
scratch.o: ../core/Cube_aux.h ../core/Cube_image.h ../core/Graph.h
scratch.o: ../core/Cloud.h ../core/Point.h ../core/Edge.h
