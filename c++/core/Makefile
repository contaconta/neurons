# CFLAGSOPENCV = $(shell pkg-config --cflags opencv) $(shell pkg-config --cflags gsl)
# CFLAGSGL = -I/usr/include/

# LDFLAGSOPENCV = $(shell pkg-config --libs opencv) $(shell pkg-config --libs gsl)
# LDFLAGSOPENGL = -L/usr -L/usr/lib -L/usr/X11R6/lib -lGLEW -lglut -lGLU -lGL

# LDFLAGS =  -pg -g -L../../core $(LDFLAGSOPENCV) $(LDFLAGSOPENGL) 

# CFLAGS = -c -pg -g -I../../core $(CFLAGSOPENCV) $(CFLAGSGL)


# MAKEDEPEND = $(shell makedepend -Y *.c 2>/dev/null)

# SOURCES = ascParser2.cpp \
# 	  Neuron.cpp \
# 	  utils.cpp


# OBJECTS=$(SOURCES:.cpp=.o)

# all: $(OBJECTS)

# .cpp.o:
# 	g++ $(CFLAGS) $< -o ../objects/$@

# depend:
# 	makedepend -Y *.c 2>/dev/null

# clean :
# 	rm -rf *.o
# # DO NOT DELETE

# Neuron.o: ascParser2.h Cube.h image.h polynomial.h utils.h Cube_data.h
# Neuron.o: Cube_display.h Cube_aux.h Cube_image.h

all:
	make -C ../ depend
	make -j 3 -C ../ lib
# 	make -j 3 -C ../ cubeTools