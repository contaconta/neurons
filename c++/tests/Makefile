SHELL=/bin/bash

CFLAGSOPENCV = $(shell pkg-config --cflags opencv) $(shell pkg-config --cflags gsl)
CFLAGSGTK    = $(shell pkg-config --cflags gtk+-2.0)
CFLAGSGTKEXT    = $(shell pkg-config --cflags gtkglext-1.0 pangoft2)
CFLAGSGL = -I/usr/include/ -openmp -D WITH_OPENMP

LDFLAGSOPENCV = $(shell pkg-config --libs opencv) $(shell pkg-config --libs gsl)
LDFLAGSGTK    = $(shell pkg-config --libs gtk+-2.0)
LDFLAGSGTKEXT = $(shell pkg-config --libs gtkglext-1.0 pangoft2)
LDFLAGSOPENGL = -L/usr -L/usr/lib -L/usr/X11R6/lib -lGLEW -lglut -lGLU -lGL -lpthread -gomp

LDFLAGS =  -g -pg $(LDFLAGSOPENCV) $(LDFLAGSGTK) $(LDFLAGSGTKEXT) $(LDFLAGSOPENGL) -L../lib
CFLAGS = -c -g  -pg -I../core $(CFLAGSOPENCV) $(CFLAGSGTK) $(CFLAGSGTKEXT) $(CFLAGSGL)

## Each source file will create a program with the same name. The only thing to
#  touch in the makefile
SRCS = $(shell ls *.cpp)
OBJS = $(SRCS:.cpp=.o)
EXEC = $(SRCS:.cpp=)

all: depend $(EXEC)

# lib:
# 	make -C ../ depend
# 	make -C ../ lib

$(OBJS):%.o:%.cpp
	g++ $(CFLAGS) $< -o $@

$(EXEC):$(OBJS)
	g++ $(LDFLAGS) $@.o -o $@ -lneseg

print:
	echo $(OBJS)

depend:
	makedepend -Y -I../core *.cpp  2>/dev/null


clean :
	rm -rf $(OBJS) $(EXEC)

# DO NOT DELETE

BBP_morphology.o: ../core/BBP_Morphology.h ../core/neseg.h
cloud.o: ../core/CubeFactory.h ../core/Cube.h ../core/neseg.h ../core/Image.h
cloud.o: ../core/polynomial.h ../core/VisibleE.h ../core/Visible.h
cloud.o: ../core/Object.h ../core/utils.h ../core/Mask.h ../core/Cube_P.h
cloud.o: ../core/Cube_data.h ../core/Cube_display.h
cloud.o: ../core/Cube_drawBetween.h ../core/Cube_aux.h ../core/Cube_image.h
cloud.o: ../core/Cloud.h ../core/Cloud_P.h ../core/Point.h ../core/Point2D.h
cloud.o: ../core/Point2Do.h ../core/Point2Dot.h ../core/Point3D.h
cloud.o: ../core/Point3Dt.h ../core/Point3Do.h ../core/Point3Dot.h
cloud.o: ../core/Point3Dotw.h
dfactorial.o: ../core/Mask.h ../core/utils.h ../core/polynomial.h
generalParser.o: ../core/Parser.h ../core/utils.h ../core/Mask.h
generalParser.o: ../core/polynomial.h
graph.o: ../core/CubeFactory.h ../core/Cube.h ../core/neseg.h ../core/Image.h
graph.o: ../core/polynomial.h ../core/VisibleE.h ../core/Visible.h
graph.o: ../core/Object.h ../core/utils.h ../core/Mask.h ../core/Cube_P.h
graph.o: ../core/Cube_data.h ../core/Cube_display.h
graph.o: ../core/Cube_drawBetween.h ../core/Cube_aux.h ../core/Cube_image.h
graph.o: ../core/Graph.h ../core/Cloud.h ../core/Cloud_P.h ../core/Point.h
graph.o: ../core/Point2D.h ../core/Point2Do.h ../core/Point2Dot.h
graph.o: ../core/Point3D.h ../core/Point3Dt.h ../core/Point3Do.h
graph.o: ../core/Point3Dot.h ../core/Point3Dotw.h ../core/EdgeSet.h
graph.o: ../core/Edge.h ../core/EdgeW.h ../core/Edge2W.h ../core/Graph_P.h
graph.o: ../core/GraphFactory.h
hermitePolynomials.o: ../core/Mask.h ../core/utils.h ../core/polynomial.h
hermitePolynomials.o: ../core/Cube.h ../core/neseg.h ../core/Image.h
hermitePolynomials.o: ../core/VisibleE.h ../core/Visible.h ../core/Object.h
hermitePolynomials.o: ../core/Cube_P.h ../core/Cube_data.h
hermitePolynomials.o: ../core/Cube_display.h ../core/Cube_drawBetween.h
hermitePolynomials.o: ../core/Cube_aux.h ../core/Cube_image.h
imageTest.o: ../core/neseg.h ../core/Image.h ../core/polynomial.h
imageTest.o: ../core/VisibleE.h ../core/Visible.h ../core/Object.h
imageTest.o: ../core/utils.h ../core/Mask.h
matlabControl.o: ../core/utils.h ../core/Mask.h ../core/polynomial.h
neuronToMicrometers.o: ../core/Image.h ../core/neseg.h ../core/polynomial.h
neuronToMicrometers.o: ../core/VisibleE.h ../core/Visible.h ../core/Object.h
neuronToMicrometers.o: ../core/utils.h ../core/Mask.h
point.o: ../core/CubeFactory.h ../core/Cube.h ../core/neseg.h ../core/Image.h
point.o: ../core/polynomial.h ../core/VisibleE.h ../core/Visible.h
point.o: ../core/Object.h ../core/utils.h ../core/Mask.h ../core/Cube_P.h
point.o: ../core/Cube_data.h ../core/Cube_display.h
point.o: ../core/Cube_drawBetween.h ../core/Cube_aux.h ../core/Cube_image.h
point.o: ../core/Point3D.h ../core/Point.h
