## Comments included in the makefile to understand this criptic language

SHELL=/bin/bash

#Relevant directories (I am dislexic)
OBJ_DIR = objects/
BIN_DIR = bin/
LIB_DIR = lib/
INST_DIR=/home/ggonzale/bin/
CUBE_TOOLS_SRC_DIR = tools/
VIEWER_SRC_DIR = imageAnnotate/src/
CORE_SRC_DIR  = core/
# DIR = $(PWD)
DIR=/home/ggonzale/ggonzale/steerableFilters2D/

CPP = gcc
OPENMP = -fopenmp
OPENMPL = -lgomp
D_OPENMP = -DWITH_OPENMP -D_OPENMP


#Compilation flags
CFLAGSOPENCV = $(shell pkg-config --cflags opencv) $(shell pkg-config --cflags gsl)
CFLAGSGTK    = #$(shell pkg-config --cflags gtk+-2.0)
CFLAGSGTKEXT    = #$(shell pkg-config --cflags gtkglext-1.0 pangoft2)
CFLAGSGL = -I/usr/include/ -I/home/ggonzale/workspace/viva/core -I/home/ggonzale/workspace/Torch3/include/
# CFLAGSTORCH = -DUSE_DOUBLE -I/home/ggonzale/workspace/Torch3/core -I/home/ggonzale/workspace/Torch3/convolutions -I/home/ggonzale/workspace/Torch3/datasets -I/home/ggonzale/workspace/Torch3/decoder -I/home/ggonzale/workspace/Torch3/distributions -I/home/ggonzale/workspace/Torch3/gradients -I/home/ggonzale/workspace/Torch3/matrix -I/home/ggonzale/workspace/Torch3/nonparametrics -I/home/ggonzale/workspace/Torch3/speech -I/home/ggonzale/workspace/Torch3/kernels -I/home/ggonzale/workspace/Torch3/matrix
CFLAGSTORCH = -I/home/ggonzale/workspace/Torch3/core -I/home/ggonzale/workspace/Torch3/convolutions -I/home/ggonzale/workspace/Torch3/datasets -I/home/ggonzale/workspace/Torch3/decoder -I/home/ggonzale/workspace/Torch3/distributions -I/home/ggonzale/workspace/Torch3/gradients -I/home/ggonzale/workspace/Torch3/matrix -I/home/ggonzale/workspace/Torch3/nonparametrics -I/home/ggonzale/workspace/Torch3/speech -I/home/ggonzale/workspace/Torch3/kernels -I/home/ggonzale/workspace/Torch3/matrix


LDFLAGSOPENCV = $(shell pkg-config --libs opencv) $(shell pkg-config --libs gsl)
LDFLAGSGTK    = #$(shell pkg-config --libs gtk+-2.0)
LDFLAGSGTKEXT = #$(shell pkg-config --libs gtkglext-1.0 pangoft2)
# LDFLAGSOPENGL = -L/usr -L/usr/lib -L/usr/X11R6/lib -L/home/ggonzale/workspace/neseg/lib -lGLEW -lglut -lGLU -lGL -L/home/ggonzale/workspace/Torch3/libs/Linux_opt_double
LDFLAGSOPENGL = -L/usr -L/usr/lib -L/usr/X11R6/lib -L/home/ggonzale/workspace/viva/lib -lglut -lGLU -lGL -L/home/ggonzale/workspace/Torch3/libs/Linux_opt_float

LDFLAGS =  $(LDFLAGSOPENCV) $(LDFLAGSGTK) $(LDFLAGSGTKEXT) $(LDFLAGSOPENGL) -Llib $(OPENMPL) -lGLEW
CFLAGS = -c $(OPENMP) $(D_OPENMP) -Icore $(CFLAGSOPENCV) $(CFLAGSGTK) $(CFLAGSGTKEXT) $(CFLAGSGL) $(CFLAGSTORCH)
# CFLAGS = -c -g  $(OPENMP) -pg -Icore $(CFLAGSOPENCV) $(CFLAGSGTK) $(CFLAGSGTKEXT) $(CFLAGSGL) $(CFLAGSTORCH) $(D_OPENMP)


# Files for the cube tools. For each of the files here, a program with the same name will be created in the bin directory
# TOOLS_CUBE_SRC_N = generatePoints.cpp steer.cpp  steerImage.cpp steerImageMultiScale.cpp steerScale.cpp  test.cpp  train.cpp imageDerivatives.cpp steerImageMultiScaleTorch.cpp steerImageMultiScaleTorchHermite.cpp steerOrthogonal.cpp steerNormalized.cpp normalizeAlphaCoords.cpp imageDerivativesAt.cpp steerImageMultiScaleTorchNormalized.cpp
TOOLS_CUBE_SRC_N = generatePoints.cpp steerImageMultiScaleTorchNormalized.cpp steerImage.cpp steer.cpp steerImageMultiScale.cpp trainInBBPSG12D.cpp steerImageTorchM.cpp imageRadius.cpp imageDerivativesRadius.cpp getCoordinatesRadius.cpp steerRadius.cpp linearSVM.cpp

TOOLS_CUBE_EXEC_P=$(TOOLS_CUBE_SRC_N:.cpp=)
TOOLS_CUBE_EXEC=$(addprefix $(BIN_DIR),$(TOOLS_CUBE_EXEC_P))
TOOLS_CUBE_SRC=$(addprefix $(CUBE_TOOLS_SRC_DIR),$(TOOLS_CUBE_SRC_N))
TOOLS_CUBE_OBJ_T=$(TOOLS_CUBE_SRC:.cpp=.o)
TOOLS_CUBE_OBJ=$(addprefix objects/,$(TOOLS_CUBE_OBJ_T))


## Makefile syntax : $(makefiles_function argument,variable_where_to_do_the_function)

#Files for the core of the library
SRC_CORE_N = SteerableFilter2D.cpp  SteerableFilter2DMultiScale.cpp  SteerableFilter2DHermite.cpp SteerableFilter2DMultiScaleHermite.cpp SteerableFilter2DOrthogonal.cpp SteerableFilter2DNormalized.cpp SteerableFilter2DMultiScaleOrthogonal.cpp SteerableFilter2DMultiScaleNormalized.cpp SVMSteerableFeatures3D.cpp
SRC_CORE   = $(addprefix $(CORE_SRC_DIR),$(SRC_CORE_N))
OBJ_CORE_O=$(SRC_CORE:.cpp=.o)
OBJ_CORE=$(addprefix $(OBJ_DIR),$(OBJ_CORE_O))

#Files for the viewer
SRC_VIEWER_N = callbacks.c interface.c support.c main.c
SRC_VIEWER=$(addprefix $(VIEWER_SRC_DIR),$(SRC_VIEWER_N))
OBJ_VIEWER_O=$(SRC_VIEWER_N:.c=.o)
OBJ_VIEWER=$(addprefix $(OBJ_DIR)$(VIEWER_SRC_DIR),$(OBJ_VIEWER_O))


############ THE RULES START HERE #########################

# This will compile all the library
all: mkobjdirs lib cubeTools

cubeTools: lib $(TOOLS_CUBE_EXEC)

#Creates the obj directory structure
mkobjdirs:
	mkdir -p objects
	mkdir -p objects/core
	mkdir -p objects/tools
	mkdir -p objects/imageAnnotate
	mkdir -p objects/imageAnnotate/src
	mkdir -p bin
	mkdir -p lib

print:
	echo $(TOOLS_CUBE_OBJ)

#Compiles the cube tools
## Makefile syntax: objective:prefix%suffix:%whatever  - % is the name of the
#                   file we are interested in
$(TOOLS_CUBE_OBJ):$(OBJ_DIR)%.o:%.cpp
	$(CPP) $(CFLAGS) $< -o $@

$(TOOLS_CUBE_EXEC): lib $(TOOLS_CUBE_OBJ)
	$(CPP) $(LDFLAGS) $(OBJ_DIR)$(CUBE_TOOLS_SRC_DIR)$(@F).o -o $(BIN_DIR)$(@F) -lsteer -lviva -ltorch
	ln -sf $(DIR)/$(BIN_DIR)$(@F) $(INST_DIR)$(@F)


#Compilation of the library
$(OBJ_CORE):$(OBJ_DIR)%.o:%.cpp
	$(CPP) $(CFLAGS) $< -o $@

lib: mkobjdirs depend $(OBJ_CORE)
	ar rcs lib/libsteer.a $(OBJ_CORE)

#Compilation for the viewer
$(OBJ_VIEWER):$(OBJ_DIR)%.o:%.c
	$(CPP) $(CFLAGS) $< -o $@

iannotate: $(OBJ_VIEWER)
	$(CPP) $(LDFLAGS) $(OBJ_VIEWER) -o $(BIN_DIR)$(@F) -lsteer

## IMPORTANT: adds the dependencies with the source files and not with the objects
## makedepend arguments: -Y = ignore standard libraries -p = add preffix 2>/dev/null please do not fill my screen with crap

depend:

clean :
	rm -rf objects/* bin/* lib/*

# DO NOT DELETE

objects/core/SteerableFilter2D.o: core/SteerableFilter2D.h
objects/core/SteerableFilter2DHermite.o: core/SteerableFilter2DHermite.h
objects/core/SteerableFilter2DHermite.o: core/SteerableFilter2D.h
objects/core/SteerableFilter2DHermiteNotNorm.o: core/SteerableFilter2DHermiteNotNorm.h
objects/core/SteerableFilter2DHermiteNotNorm.o: core/SteerableFilter2D.h
objects/core/SteerableFilter2DMultiScale.o: core/SteerableFilter2DMultiScale.h
objects/core/SteerableFilter2DMultiScale.o: core/SteerableFilter2D.h
objects/core/SteerableFilter2DMultiScale.o: core/Point2Dsf.h core/Point2D.h
objects/core/SteerableFilter2DMultiScaleHermite.o: core/SteerableFilter2DMultiScaleHermite.h
objects/core/SteerableFilter2DMultiScaleHermite.o: core/SteerableFilter2DMultiScale.h
objects/core/SteerableFilter2DMultiScaleHermite.o: core/SteerableFilter2D.h
objects/core/SteerableFilter2DMultiScaleHermite.o: core/Point2Dsf.h
objects/core/SteerableFilter2DMultiScaleHermite.o: core/SteerableFilter2DHermite.h
objects/core/SteerableFilter2DMultiScaleNormalized.o: core/SteerableFilter2DMultiScaleNormalized.h
objects/core/SteerableFilter2DMultiScaleNormalized.o: core/SteerableFilter2DMultiScale.h
objects/core/SteerableFilter2DMultiScaleNormalized.o: core/SteerableFilter2D.h
objects/core/SteerableFilter2DMultiScaleNormalized.o: core/Point2Dsf.h
objects/core/SteerableFilter2DMultiScaleNormalized.o: core/SteerableFilter2DNormalized.h
objects/core/SteerableFilter2DMultiScaleOrthogonal.o: core/SteerableFilter2DMultiScaleOrthogonal.h
objects/core/SteerableFilter2DMultiScaleOrthogonal.o: core/SteerableFilter2DMultiScale.h
objects/core/SteerableFilter2DMultiScaleOrthogonal.o: core/SteerableFilter2D.h
objects/core/SteerableFilter2DMultiScaleOrthogonal.o: core/Point2Dsf.h
objects/core/SteerableFilter2DMultiScaleOrthogonal.o: core/SteerableFilter2DOrthogonal.h
objects/core/SteerableFilter2DNormalized.o: core/SteerableFilter2DNormalized.h
objects/core/SteerableFilter2DNormalized.o: core/SteerableFilter2D.h
objects/core/SteerableFilter2DOrthogonal.o: core/SteerableFilter2DOrthogonal.h
objects/core/SteerableFilter2DOrthogonal.o: core/SteerableFilter2D.h
objects/tools/generatePoints.o: core/SteerableFilter2D.h
objects/tools/generatePoints.o: core/SteerableFilter2DMultiScale.h
objects/tools/generatePoints.o: core/Point2Dsf.h
objects/tools/generatePoints.o: core/SteerableFilter2DMultiScaleHermite.h
objects/tools/generatePoints.o: core/SteerableFilter2DHermite.h
objects/tools/generatePoints.o: core/SteerableFilter2DMultiScaleOrthogonal.h
objects/tools/generatePoints.o: core/SteerableFilter2DOrthogonal.h
objects/tools/generatePoints.o: core/SteerableFilter2DMultiScaleNormalized.h
objects/tools/generatePoints.o: core/SteerableFilter2DNormalized.h
objects/tools/imageDerivativesAt.o: core/SteerableFilter2DOrthogonal.h
objects/tools/imageDerivativesAt.o: core/SteerableFilter2D.h
objects/tools/normalizeAlphaCoords.o: core/SteerableFilter2D.h
objects/tools/steer.o: core/SteerableFilter2D.h
objects/tools/steerHermite.o: core/SteerableFilter2DHermite.h
objects/tools/steerHermite.o: core/SteerableFilter2D.h
objects/tools/steerHermiteNotNorm.o: core/SteerableFilter2DHermiteNotNorm.h
objects/tools/steerHermiteNotNorm.o: core/SteerableFilter2D.h
objects/tools/steerImage.o: core/SteerableFilter2D.h
objects/tools/steerImageMultiScale.o: core/SteerableFilter2D.h
objects/tools/steerImageMultiScale.o: core/SteerableFilter2DMultiScale.h
objects/tools/steerImageMultiScale.o: core/Point2Dsf.h
objects/tools/steerImageMultiScaleTorch.o: core/SteerableFilter2D.h
objects/tools/steerImageMultiScaleTorch.o: core/SteerableFilter2DMultiScale.h
objects/tools/steerImageMultiScaleTorch.o: core/Point2Dsf.h
objects/tools/steerImageMultiScaleTorchHermite.o: core/SteerableFilter2D.h
objects/tools/steerImageMultiScaleTorchHermite.o: core/SteerableFilter2DMultiScale.h
objects/tools/steerImageMultiScaleTorchHermite.o: core/Point2Dsf.h
objects/tools/steerImageMultiScaleTorchHermite.o: core/SteerableFilter2DMultiScaleHermite.h
objects/tools/steerImageMultiScaleTorchHermite.o: core/SteerableFilter2DHermite.h
objects/tools/steerImageMultiScaleTorchNormalized.o: core/SteerableFilter2D.h
objects/tools/steerImageMultiScaleTorchNormalized.o: core/SteerableFilter2DMultiScale.h
objects/tools/steerImageMultiScaleTorchNormalized.o: core/Point2Dsf.h
objects/tools/steerImageMultiScaleTorchNormalized.o: core/SteerableFilter2DMultiScaleNormalized.h
objects/tools/steerImageMultiScaleTorchNormalized.o: core/SteerableFilter2DNormalized.h
objects/tools/steerNormalized.o: core/SteerableFilter2DNormalized.h
objects/tools/steerNormalized.o: core/SteerableFilter2D.h
objects/tools/steerOrthogonal.o: core/SteerableFilter2DOrthogonal.h
objects/tools/steerOrthogonal.o: core/SteerableFilter2D.h
objects/tools/steerScale.o: core/SteerableFilter2DMultiScale.h
objects/tools/steerScale.o: core/SteerableFilter2D.h core/Point2Dsf.h
objects/tools/test.o: core/SteerableFilter2D.h
objects/tools/train.o: core/SteerableFilter2D.h
objects/imageAnnotate/src/callbacks.o: imageAnnotate/src/callbacks.h
objects/imageAnnotate/src/callbacks.o: imageAnnotate/src/interface.h
objects/imageAnnotate/src/callbacks.o: imageAnnotate/src/support.h
objects/imageAnnotate/src/interface.o: imageAnnotate/src/callbacks.h
objects/imageAnnotate/src/interface.o: imageAnnotate/src/interface.h
objects/imageAnnotate/src/interface.o: imageAnnotate/src/support.h
objects/imageAnnotate/src/main.o: imageAnnotate/src/interface.h
objects/imageAnnotate/src/main.o: imageAnnotate/src/support.h
objects/imageAnnotate/src/support.o: imageAnnotate/src/support.h
