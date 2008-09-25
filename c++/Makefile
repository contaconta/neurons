## Comments included in the makefile to understand this criptic language
SHELL=/bin/bash

## Compilation options (at some points this should be done automatically)

# Compiler to use - if you want to use openmp you neefd g++ > 4.2
# if the flags are 1, you enable the use of the libraries
# libraries required:
#    neseg:  opencv, gsl, opengl
#    viewer: gtk, gtkglext, glew
#    tools:  the same as neseg
CPP = g++-4.2
#CPP = g++
USE_OPENMP=1
USE_GLEW=1

## Compilation flags
ifeq ($(USE_OPENMP),1)
  OPENMP = -fopenmp
  OPENMPL = -lgomp
  D_OPENMP = -D WITH_OPENMP
else
  OPENMP =
  OPENMPL =
  D_OPENMP =
endif

ifeq ($(USE_GLEW),1)
  D_GLEW = -D WITH_GLEW
  GLEWL = -lGLEW
else
  D_GLEW =
  GLEWL =
endif

CFLAGSOPENCV = $(shell pkg-config --cflags opencv) $(shell pkg-config --cflags gsl)
CFLAGSGTK    = $(shell pkg-config --cflags gtk+-2.0)
CFLAGSGTKEXT    = $(shell pkg-config --cflags gtkglext-1.0 pangoft2)
CFLAGSGL = -I/usr/include/

LDFLAGSOPENCV = $(shell pkg-config --libs opencv) $(shell pkg-config --libs gsl)
LDFLAGSGTK    = $(shell pkg-config --libs gtk+-2.0)
LDFLAGSGTKEXT = $(shell pkg-config --libs gtkglext-1.0 pangoft2)
LDFLAGSOPENGL = -L/usr -L/usr/lib -L/usr/X11R6/lib $(GLEWL) -lglut -lGLU -lGL

LDFLAGS =  -g -pg $(LDFLAGSOPENCV)  $(LDFLAGSOPENGL) -Llib $(OPENMPL)
CFLAGS = -c $(OPENMP) -g  -pg $(D_GLEW) -Icore $(CFLAGSOPENCV) $(CFLAGSGL)

## Relevant directories (I am dislexic)
OBJ_DIR = objects/
BIN_DIR = bin/
LIB_DIR = lib/
INST_DIR=/home/ggonzale/bin/
CUBE_TOOLS_SRC_DIR = tools/cube/
GRAPH_TOOLS_SRC_DIR = tools/graph/
NEURON_TOOLS_SRC_DIR = tools/neuron/
IMAGE_TOOLS_SRC_DIR = tools/image/
VIEWER_SRC_DIR = viewer/src/
CORE_SRC_DIR  = core/
# DIR = $(PWD)
DIR=/home/ggonzale/ggonzale/neseg/

# Files for the cube tools. For each of the files here, a program with the same name will be created in the bin directory
TOOLS_CUBE_SRC_N_P = $(shell ls $(CUBE_TOOLS_SRC_DIR)/*.cpp)
TOOLS_CUBE_SRC_N=$(subst $(CUBE_TOOLS_SRC_DIR),,$(TOOLS_CUBE_SRC_N_P))
TOOLS_CUBE_EXEC_P=$(TOOLS_CUBE_SRC_N:.cpp=)
TOOLS_CUBE_EXEC=$(addprefix $(BIN_DIR),$(TOOLS_CUBE_EXEC_P))
TOOLS_CUBE_SRC=$(addprefix $(CUBE_TOOLS_SRC_DIR),$(TOOLS_CUBE_SRC_N))
TOOLS_CUBE_OBJ_T=$(TOOLS_CUBE_SRC:.cpp=.o)
TOOLS_CUBE_OBJ=$(addprefix objects/,$(TOOLS_CUBE_OBJ_T))
## Makefile syntax : $(makefiles_function argument,variable_where_to_do_the_function)

# Files for the graph tools
TOOLS_GRAPH_SRC_N_P = $(shell ls $(GRAPH_TOOLS_SRC_DIR)/*.cpp)
TOOLS_GRAPH_SRC_N = $(subst $(GRAPH_TOOLS_SRC_DIR),,$(TOOLS_GRAPH_SRC_N_P))
TOOLS_GRAPH_EXEC_P=$(TOOLS_GRAPH_SRC_N:.cpp=)
TOOLS_GRAPH_EXEC=$(addprefix $(BIN_DIR),$(TOOLS_GRAPH_EXEC_P))
TOOLS_GRAPH_SRC=$(addprefix $(GRAPH_TOOLS_SRC_DIR),$(TOOLS_GRAPH_SRC_N))
TOOLS_GRAPH_OBJ_T=$(TOOLS_GRAPH_SRC:.cpp=.o)
TOOLS_GRAPH_OBJ=$(addprefix objects/,$(TOOLS_GRAPH_OBJ_T))

#Files for the neuron tools
TOOLS_NEURON_SRC_N_P = $(shell ls $(NEURON_TOOLS_SRC_DIR)/*.cpp)
TOOLS_NEURON_SRC_N = $(subst $(NEURON_TOOLS_SRC_DIR),,$(TOOLS_NEURON_SRC_N_P))
TOOLS_NEURON_EXEC_P=$(TOOLS_NEURON_SRC_N:.cpp=)
TOOLS_NEURON_EXEC=$(addprefix $(BIN_DIR),$(TOOLS_NEURON_EXEC_P))
TOOLS_NEURON_SRC=$(addprefix $(NEURON_TOOLS_SRC_DIR),$(TOOLS_NEURON_SRC_N))
TOOLS_NEURON_OBJ_T=$(TOOLS_NEURON_SRC:.cpp=.o)
TOOLS_NEURON_OBJ=$(addprefix objects/,$(TOOLS_NEURON_OBJ_T))

#Files for the image tools
TOOLS_IMAGE_SRC_N_P = $(shell ls $(IMAGE_TOOLS_SRC_DIR)/*.cpp)
TOOLS_IMAGE_SRC_N = $(subst $(IMAGE_TOOLS_SRC_DIR),,$(TOOLS_IMAGE_SRC_N_P))
TOOLS_IMAGE_EXEC_P=$(TOOLS_IMAGE_SRC_N:.cpp=)
TOOLS_IMAGE_EXEC=$(addprefix $(BIN_DIR),$(TOOLS_IMAGE_EXEC_P))
TOOLS_IMAGE_SRC=$(addprefix $(IMAGE_TOOLS_SRC_DIR),$(TOOLS_IMAGE_SRC_N))
TOOLS_IMAGE_OBJ_T=$(TOOLS_IMAGE_SRC:.cpp=.o)
TOOLS_IMAGE_OBJ=$(addprefix objects/,$(TOOLS_IMAGE_OBJ_T))

#Files for the core of the library
SRC_CORE_N_P = $(shell ls $(CORE_SRC_DIR)/*.cpp)
SRC_CORE_N = $(subst $(CORE_SRC_DIR),,$(SRC_CORE_N_P))
SRC_CORE   = $(addprefix $(CORE_SRC_DIR),$(SRC_CORE_N))
OBJ_CORE_O=$(SRC_CORE:.cpp=.o)
OBJ_CORE=$(addprefix $(OBJ_DIR),$(OBJ_CORE_O))

#Files for the viewer
SRC_VIEWER_N = callbacks.c interface.c support.c main.c callbacks_menu.c callbacks_ui.c callbacks_draw.c callbacks_ascEdit.c
SRC_VIEWER=$(addprefix $(VIEWER_SRC_DIR),$(SRC_VIEWER_N));
OBJ_VIEWER_O=$(SRC_VIEWER_N:.c=.o)
OBJ_VIEWER=$(addprefix $(OBJ_DIR)$(VIEWER_SRC_DIR),$(OBJ_VIEWER_O))


############ THE RULES START HERE #########################

# This will compile all the library
all: mkobjdirs depend lib neseg #cubeTools

everything: mkobjdirs depend lib neseg tools

tools: cubeTools graphTools neuronTools imageTools

cubeTools: $(TOOLS_CUBE_EXEC)

graphTools: $(TOOLS_GRAPH_EXEC)

neuronTools: $(TOOLS_NEURON_EXEC)

imageTools: $(TOOLS_IMAGE_EXEC)

#Creates the obj directory structure
# I know mkdir -p might not be portable, but it is easy to use ...
mkobjdirs:
	mkdir -p objects
	mkdir -p objects/core
	mkdir -p objects/tools
	mkdir -p objects/tools/cube
	mkdir -p objects/tools/graph
	mkdir -p objects/tools/neuron
	mkdir -p objects/tools/image
	mkdir -p objects/viewer
	mkdir -p objects/viewer/src
	mkdir -p bin
	mkdir -p lib

print:
	echo $(TOOLS_CUBE_OBJ)

doc: $(SRC_CORE_N_P)
	doxygen doc/Doxyfile

#Compiles the cube tools
## Makefile syntax: objective:prefix%suffix:%whatever  - % is the name of the
#                   file we are interested in
$(TOOLS_CUBE_OBJ):$(OBJ_DIR)%.o:%.cpp
	$(CPP) $(CFLAGS) $< -o $@

$(TOOLS_CUBE_EXEC): $(TOOLS_CUBE_OBJ) lib
	$(CPP) $(LDFLAGS) $(OBJ_DIR)$(CUBE_TOOLS_SRC_DIR)$(@F).o -o $(BIN_DIR)$(@F) -lneseg
	ln -sf $(DIR)/$(BIN_DIR)$(@F) $(INST_DIR)$(@F)

# Compilation of the graph tools
$(TOOLS_GRAPH_OBJ):$(OBJ_DIR)%.o:%.cpp
	$(CPP) $(CFLAGS) $< -o $@

$(TOOLS_GRAPH_EXEC): $(TOOLS_GRAPH_OBJ) lib
	$(CPP) $(LDFLAGS) $(OBJ_DIR)$(GRAPH_TOOLS_SRC_DIR)$(@F).o -o $(BIN_DIR)$(@F) -lneseg
	ln -sf $(DIR)/$(BIN_DIR)$(@F) $(INST_DIR)$(@F)

# Compilation of the neuron tools
$(TOOLS_NEURON_OBJ):$(OBJ_DIR)%.o:%.cpp
	$(CPP) $(CFLAGS) $< -o $@

$(TOOLS_NEURON_EXEC): $(TOOLS_NEURON_OBJ) lib
	$(CPP) $(LDFLAGS) $(OBJ_DIR)$(NEURON_TOOLS_SRC_DIR)$(@F).o -o $(BIN_DIR)$(@F) -lneseg
	ln -sf $(DIR)/$(BIN_DIR)$(@F) $(INST_DIR)$(@F)

# Compilation of the image tools
$(TOOLS_IMAGE_OBJ):$(OBJ_DIR)%.o:%.cpp
	$(CPP) $(CFLAGS) $< -o $@

$(TOOLS_IMAGE_EXEC): $(TOOLS_IMAGE_OBJ) lib
	$(CPP) $(LDFLAGS) $(OBJ_DIR)$(IMAGE_TOOLS_SRC_DIR)$(@F).o -o $(BIN_DIR)$(@F) -lneseg
	ln -sf $(DIR)/$(BIN_DIR)$(@F) $(INST_DIR)$(@F)

#Compilation of the library
$(OBJ_CORE):$(OBJ_DIR)%.o:%.cpp
	$(CPP) $(CFLAGS) $< -o $@

lib: mkobjdirs depend $(OBJ_CORE)
	ar rcs lib/libneseg.a $(OBJ_CORE)

#Compilation for the viewer
$(OBJ_VIEWER):$(OBJ_DIR)%.o:%.c
	$(CPP) $(CFLAGS) $(CFLAGSGTK) $(CFLAGSGTKEXT) $< -o $@

neseg: $(OBJ_VIEWER) lib
	$(CPP) $(LDFLAGS) $(LDFLAGSGTK) $(LDFLAGSGTKEXT) $(OBJ_VIEWER) -o $(BIN_DIR)$(@F) -lneseg

## IMPORTANT: adds the dependencies with the source files and not with the objects
## makedepend arguments: -Y = ignore standard libraries -p = add preffix 2>/dev/null please do not fill my screen with crap

depend:
	makedepend -Y -I$core -p$(OBJ_DIR) $(CORE_SRC_DIR)*.cpp $(CUBE_TOOLS_SRC_DIR)*.cpp $(GRAPH_TOOLS_SRC_DIR)*.cpp $(NEURON_TOOLS_SRC_DIR)*.cpp $(IMAGE_TOOLS_SRC_DIR)*.cpp $(VIEWER_SRC_DIR)*.c  2>/dev/null

# install: $(EXECUTABLES)
# 	echo $<
# 	ln -sf  $(PREFIX)$(<F) $(DIR_INSTALL)$<

clean :
	rm -rf objects bin lib/l*

# DO NOT DELETE

objects/core/ascParser2.o: core/ascParser2.h core/neseg.h core/Neuron.h
objects/core/ascParser2.o: core/Cube.h core/Image.h core/polynomial.h
objects/core/ascParser2.o: core/VisibleE.h core/Visible.h core/Object.h
objects/core/ascParser2.o: core/utils.h core/Mask.h core/Cube_P.h
objects/core/ascParser2.o: core/Cube_data.h core/Cube_display.h
objects/core/ascParser2.o: core/Cube_aux.h core/Cube_image.h
objects/core/Axis.o: core/Axis.h core/VisibleE.h core/Visible.h core/neseg.h
objects/core/Axis.o: core/Object.h
objects/core/Cloud.o: core/Cloud.h core/Cloud_P.h core/Point.h core/neseg.h
objects/core/Cloud.o: core/Visible.h core/Object.h core/Point2D.h
objects/core/Cloud.o: core/Point2Do.h core/Point2Dot.h core/Point3D.h
objects/core/Cloud.o: core/utils.h core/Mask.h core/polynomial.h
objects/core/Cloud.o: core/VisibleE.h
objects/core/Edge.o: core/Edge.h core/Point.h core/neseg.h core/Visible.h
objects/core/Edge.o: core/Object.h
objects/core/EdgeSet.o: core/EdgeSet.h core/Point3D.h core/neseg.h
objects/core/EdgeSet.o: core/Point.h core/Visible.h core/Object.h
objects/core/EdgeSet.o: core/utils.h core/Mask.h core/polynomial.h
objects/core/EdgeSet.o: core/VisibleE.h core/Edge.h core/EdgeW.h
objects/core/Graph.o: core/Graph.h core/Cloud.h core/Cloud_P.h core/Point.h
objects/core/Graph.o: core/neseg.h core/Visible.h core/Object.h
objects/core/Graph.o: core/Point2D.h core/Point2Do.h core/Point2Dot.h
objects/core/Graph.o: core/Point3D.h core/utils.h core/Mask.h
objects/core/Graph.o: core/polynomial.h core/VisibleE.h core/EdgeSet.h
objects/core/Graph.o: core/Edge.h core/EdgeW.h core/Graph_P.h
objects/core/Mask.o: core/Mask.h core/utils.h core/polynomial.h
objects/core/Neuron.o: core/Neuron.h core/neseg.h core/ascParser2.h
objects/core/Neuron.o: core/utils.h core/Mask.h core/polynomial.h core/Cube.h
objects/core/Neuron.o: core/Image.h core/VisibleE.h core/Visible.h
objects/core/Neuron.o: core/Object.h core/Cube_P.h core/Cube_data.h
objects/core/Neuron.o: core/Cube_display.h core/Cube_aux.h core/Cube_image.h
objects/core/Parser.o: core/Parser.h
objects/core/Point2D.o: core/Point2D.h core/neseg.h core/Point.h
objects/core/Point2D.o: core/Visible.h core/Object.h
objects/core/Point2Do.o: core/Point2Do.h core/Point.h core/neseg.h
objects/core/Point2Do.o: core/Visible.h core/Object.h
objects/core/Point2Dot.o: core/Point2Dot.h core/Point2Do.h core/Point.h
objects/core/Point2Dot.o: core/neseg.h core/Visible.h core/Object.h
objects/core/Point3D.o: core/Point3D.h core/neseg.h core/Point.h
objects/core/Point3D.o: core/Visible.h core/Object.h
objects/core/Point.o: core/Point.h core/neseg.h core/Visible.h core/Object.h
objects/core/utils.o: core/utils.h core/Mask.h core/polynomial.h
objects/tools/cube/cubeAt.o: core/Cube.h core/neseg.h core/Image.h
objects/tools/cube/cubeAt.o: core/polynomial.h core/VisibleE.h core/Visible.h
objects/tools/cube/cubeAt.o: core/Object.h core/utils.h core/Mask.h
objects/tools/cube/cubeAt.o: core/Cube_P.h core/Cube_data.h
objects/tools/cube/cubeAt.o: core/Cube_display.h core/Cube_aux.h
objects/tools/cube/cubeAt.o: core/Cube_image.h
objects/tools/cube/cubeCalculateAguet.o: core/Cube.h core/neseg.h
objects/tools/cube/cubeCalculateAguet.o: core/Image.h core/polynomial.h
objects/tools/cube/cubeCalculateAguet.o: core/VisibleE.h core/Visible.h
objects/tools/cube/cubeCalculateAguet.o: core/Object.h core/utils.h
objects/tools/cube/cubeCalculateAguet.o: core/Mask.h core/Cube_P.h
objects/tools/cube/cubeCalculateAguet.o: core/Cube_data.h core/Cube_display.h
objects/tools/cube/cubeCalculateAguet.o: core/Cube_aux.h core/Cube_image.h
objects/tools/cube/cubeCalculateFrangi.o: core/Cube.h core/neseg.h
objects/tools/cube/cubeCalculateFrangi.o: core/Image.h core/polynomial.h
objects/tools/cube/cubeCalculateFrangi.o: core/VisibleE.h core/Visible.h
objects/tools/cube/cubeCalculateFrangi.o: core/Object.h core/utils.h
objects/tools/cube/cubeCalculateFrangi.o: core/Mask.h core/Cube_P.h
objects/tools/cube/cubeCalculateFrangi.o: core/Cube_data.h
objects/tools/cube/cubeCalculateFrangi.o: core/Cube_display.h core/Cube_aux.h
objects/tools/cube/cubeCalculateFrangi.o: core/Cube_image.h
objects/tools/cube/cubeConvertToFloat.o: core/Cube.h core/neseg.h
objects/tools/cube/cubeConvertToFloat.o: core/Image.h core/polynomial.h
objects/tools/cube/cubeConvertToFloat.o: core/VisibleE.h core/Visible.h
objects/tools/cube/cubeConvertToFloat.o: core/Object.h core/utils.h
objects/tools/cube/cubeConvertToFloat.o: core/Mask.h core/Cube_P.h
objects/tools/cube/cubeConvertToFloat.o: core/Cube_data.h core/Cube_display.h
objects/tools/cube/cubeConvertToFloat.o: core/Cube_aux.h core/Cube_image.h
objects/tools/cube/cubeConvertToFloat.o: core/Neuron.h core/ascParser2.h
objects/tools/cube/cubeCut.o: core/Cube.h core/neseg.h core/Image.h
objects/tools/cube/cubeCut.o: core/polynomial.h core/VisibleE.h
objects/tools/cube/cubeCut.o: core/Visible.h core/Object.h core/utils.h
objects/tools/cube/cubeCut.o: core/Mask.h core/Cube_P.h core/Cube_data.h
objects/tools/cube/cubeCut.o: core/Cube_display.h core/Cube_aux.h
objects/tools/cube/cubeCut.o: core/Cube_image.h
objects/tools/cube/cubeDecimate.o: core/Cube.h core/neseg.h core/Image.h
objects/tools/cube/cubeDecimate.o: core/polynomial.h core/VisibleE.h
objects/tools/cube/cubeDecimate.o: core/Visible.h core/Object.h core/utils.h
objects/tools/cube/cubeDecimate.o: core/Mask.h core/Cube_P.h core/Cube_data.h
objects/tools/cube/cubeDecimate.o: core/Cube_display.h core/Cube_aux.h
objects/tools/cube/cubeDecimate.o: core/Cube_image.h core/Cloud_P.h
objects/tools/cube/cubeDecimate.o: core/Point.h core/Point2D.h
objects/tools/cube/cubeDecimate.o: core/Point2Do.h core/Point2Dot.h
objects/tools/cube/cubeDecimate.o: core/Point3D.h core/Cloud.h
objects/tools/cube/cubeDerivative.o: core/Cube.h core/neseg.h core/Image.h
objects/tools/cube/cubeDerivative.o: core/polynomial.h core/VisibleE.h
objects/tools/cube/cubeDerivative.o: core/Visible.h core/Object.h
objects/tools/cube/cubeDerivative.o: core/utils.h core/Mask.h core/Cube_P.h
objects/tools/cube/cubeDerivative.o: core/Cube_data.h core/Cube_display.h
objects/tools/cube/cubeDerivative.o: core/Cube_aux.h core/Cube_image.h
objects/tools/cube/cubeDerivativesOrder.o: core/Cube.h core/neseg.h
objects/tools/cube/cubeDerivativesOrder.o: core/Image.h core/polynomial.h
objects/tools/cube/cubeDerivativesOrder.o: core/VisibleE.h core/Visible.h
objects/tools/cube/cubeDerivativesOrder.o: core/Object.h core/utils.h
objects/tools/cube/cubeDerivativesOrder.o: core/Mask.h core/Cube_P.h
objects/tools/cube/cubeDerivativesOrder.o: core/Cube_data.h
objects/tools/cube/cubeDerivativesOrder.o: core/Cube_display.h
objects/tools/cube/cubeDerivativesOrder.o: core/Cube_aux.h core/Cube_image.h
objects/tools/cube/cubeDirectoryMatrixMIP.o: core/Cube.h core/neseg.h
objects/tools/cube/cubeDirectoryMatrixMIP.o: core/Image.h core/polynomial.h
objects/tools/cube/cubeDirectoryMatrixMIP.o: core/VisibleE.h core/Visible.h
objects/tools/cube/cubeDirectoryMatrixMIP.o: core/Object.h core/utils.h
objects/tools/cube/cubeDirectoryMatrixMIP.o: core/Mask.h core/Cube_P.h
objects/tools/cube/cubeDirectoryMatrixMIP.o: core/Cube_data.h
objects/tools/cube/cubeDirectoryMatrixMIP.o: core/Cube_display.h
objects/tools/cube/cubeDirectoryMatrixMIP.o: core/Cube_aux.h
objects/tools/cube/cubeDirectoryMatrixMIP.o: core/Cube_image.h
objects/tools/cube/cubeFromDirectoryMatrix.o: core/Cube.h core/neseg.h
objects/tools/cube/cubeFromDirectoryMatrix.o: core/Image.h core/polynomial.h
objects/tools/cube/cubeFromDirectoryMatrix.o: core/VisibleE.h core/Visible.h
objects/tools/cube/cubeFromDirectoryMatrix.o: core/Object.h core/utils.h
objects/tools/cube/cubeFromDirectoryMatrix.o: core/Mask.h core/Cube_P.h
objects/tools/cube/cubeFromDirectoryMatrix.o: core/Cube_data.h
objects/tools/cube/cubeFromDirectoryMatrix.o: core/Cube_display.h
objects/tools/cube/cubeFromDirectoryMatrix.o: core/Cube_aux.h
objects/tools/cube/cubeFromDirectoryMatrix.o: core/Cube_image.h
objects/tools/cube/cubeFromDirectoryMatrixWithAffineTransformation.o: core/Cube.h
objects/tools/cube/cubeFromDirectoryMatrixWithAffineTransformation.o: core/neseg.h
objects/tools/cube/cubeFromDirectoryMatrixWithAffineTransformation.o: core/Image.h
objects/tools/cube/cubeFromDirectoryMatrixWithAffineTransformation.o: core/polynomial.h
objects/tools/cube/cubeFromDirectoryMatrixWithAffineTransformation.o: core/VisibleE.h
objects/tools/cube/cubeFromDirectoryMatrixWithAffineTransformation.o: core/Visible.h
objects/tools/cube/cubeFromDirectoryMatrixWithAffineTransformation.o: core/Object.h
objects/tools/cube/cubeFromDirectoryMatrixWithAffineTransformation.o: core/utils.h
objects/tools/cube/cubeFromDirectoryMatrixWithAffineTransformation.o: core/Mask.h
objects/tools/cube/cubeFromDirectoryMatrixWithAffineTransformation.o: core/Cube_P.h
objects/tools/cube/cubeFromDirectoryMatrixWithAffineTransformation.o: core/Cube_data.h
objects/tools/cube/cubeFromDirectoryMatrixWithAffineTransformation.o: core/Cube_display.h
objects/tools/cube/cubeFromDirectoryMatrixWithAffineTransformation.o: core/Cube_aux.h
objects/tools/cube/cubeFromDirectoryMatrixWithAffineTransformation.o: core/Cube_image.h
objects/tools/cube/cubeFromFloatImages.o: core/Cube.h core/neseg.h
objects/tools/cube/cubeFromFloatImages.o: core/Image.h core/polynomial.h
objects/tools/cube/cubeFromFloatImages.o: core/VisibleE.h core/Visible.h
objects/tools/cube/cubeFromFloatImages.o: core/Object.h core/utils.h
objects/tools/cube/cubeFromFloatImages.o: core/Mask.h core/Cube_P.h
objects/tools/cube/cubeFromFloatImages.o: core/Cube_data.h
objects/tools/cube/cubeFromFloatImages.o: core/Cube_display.h core/Cube_aux.h
objects/tools/cube/cubeFromFloatImages.o: core/Cube_image.h
objects/tools/cube/cubeFromImageStack.o: core/Cube.h core/neseg.h
objects/tools/cube/cubeFromImageStack.o: core/Image.h core/polynomial.h
objects/tools/cube/cubeFromImageStack.o: core/VisibleE.h core/Visible.h
objects/tools/cube/cubeFromImageStack.o: core/Object.h core/utils.h
objects/tools/cube/cubeFromImageStack.o: core/Mask.h core/Cube_P.h
objects/tools/cube/cubeFromImageStack.o: core/Cube_data.h core/Cube_display.h
objects/tools/cube/cubeFromImageStack.o: core/Cube_aux.h core/Cube_image.h
objects/tools/cube/cubeFromImageStackRaw.o: core/Cube.h core/neseg.h
objects/tools/cube/cubeFromImageStackRaw.o: core/Image.h core/polynomial.h
objects/tools/cube/cubeFromImageStackRaw.o: core/VisibleE.h core/Visible.h
objects/tools/cube/cubeFromImageStackRaw.o: core/Object.h core/utils.h
objects/tools/cube/cubeFromImageStackRaw.o: core/Mask.h core/Cube_P.h
objects/tools/cube/cubeFromImageStackRaw.o: core/Cube_data.h
objects/tools/cube/cubeFromImageStackRaw.o: core/Cube_display.h
objects/tools/cube/cubeFromImageStackRaw.o: core/Cube_aux.h core/Cube_image.h
objects/tools/cube/cubeGaussianMask.o: core/utils.h core/Mask.h
objects/tools/cube/cubeGaussianMask.o: core/polynomial.h
objects/tools/cube/cubeGaussianMaskOrthogonal.o: core/utils.h core/Mask.h
objects/tools/cube/cubeGaussianMaskOrthogonal.o: core/polynomial.h
objects/tools/cube/cubeGaussianPyramid2D.o: core/Cube.h core/neseg.h
objects/tools/cube/cubeGaussianPyramid2D.o: core/Image.h core/polynomial.h
objects/tools/cube/cubeGaussianPyramid2D.o: core/VisibleE.h core/Visible.h
objects/tools/cube/cubeGaussianPyramid2D.o: core/Object.h core/utils.h
objects/tools/cube/cubeGaussianPyramid2D.o: core/Mask.h core/Cube_P.h
objects/tools/cube/cubeGaussianPyramid2D.o: core/Cube_data.h
objects/tools/cube/cubeGaussianPyramid2D.o: core/Cube_display.h
objects/tools/cube/cubeGaussianPyramid2D.o: core/Cube_aux.h core/Cube_image.h
objects/tools/cube/cubeGaussianPyramid.o: core/Cube.h core/neseg.h
objects/tools/cube/cubeGaussianPyramid.o: core/Image.h core/polynomial.h
objects/tools/cube/cubeGaussianPyramid.o: core/VisibleE.h core/Visible.h
objects/tools/cube/cubeGaussianPyramid.o: core/Object.h core/utils.h
objects/tools/cube/cubeGaussianPyramid.o: core/Mask.h core/Cube_P.h
objects/tools/cube/cubeGaussianPyramid.o: core/Cube_data.h
objects/tools/cube/cubeGaussianPyramid.o: core/Cube_display.h core/Cube_aux.h
objects/tools/cube/cubeGaussianPyramid.o: core/Cube_image.h
objects/tools/cube/cubeHermiteMask.o: core/utils.h core/Mask.h
objects/tools/cube/cubeHermiteMask.o: core/polynomial.h
objects/tools/cube/cubeIntegralCubeByLayers.o: core/Cube.h core/neseg.h
objects/tools/cube/cubeIntegralCubeByLayers.o: core/Image.h core/polynomial.h
objects/tools/cube/cubeIntegralCubeByLayers.o: core/VisibleE.h core/Visible.h
objects/tools/cube/cubeIntegralCubeByLayers.o: core/Object.h core/utils.h
objects/tools/cube/cubeIntegralCubeByLayers.o: core/Mask.h core/Cube_P.h
objects/tools/cube/cubeIntegralCubeByLayers.o: core/Cube_data.h
objects/tools/cube/cubeIntegralCubeByLayers.o: core/Cube_display.h
objects/tools/cube/cubeIntegralCubeByLayers.o: core/Cube_aux.h
objects/tools/cube/cubeIntegralCubeByLayers.o: core/Cube_image.h
objects/tools/cube/cubeIntegralCube.o: core/Cube.h core/neseg.h core/Image.h
objects/tools/cube/cubeIntegralCube.o: core/polynomial.h core/VisibleE.h
objects/tools/cube/cubeIntegralCube.o: core/Visible.h core/Object.h
objects/tools/cube/cubeIntegralCube.o: core/utils.h core/Mask.h core/Cube_P.h
objects/tools/cube/cubeIntegralCube.o: core/Cube_data.h core/Cube_display.h
objects/tools/cube/cubeIntegralCube.o: core/Cube_aux.h core/Cube_image.h
objects/tools/cube/cubeMIP.o: core/Cube.h core/neseg.h core/Image.h
objects/tools/cube/cubeMIP.o: core/polynomial.h core/VisibleE.h
objects/tools/cube/cubeMIP.o: core/Visible.h core/Object.h core/utils.h
objects/tools/cube/cubeMIP.o: core/Mask.h core/Cube_P.h core/Cube_data.h
objects/tools/cube/cubeMIP.o: core/Cube_display.h core/Cube_aux.h
objects/tools/cube/cubeMIP.o: core/Cube_image.h
objects/tools/cube/cubeRenderNeuron.o: core/Cube.h core/neseg.h core/Image.h
objects/tools/cube/cubeRenderNeuron.o: core/polynomial.h core/VisibleE.h
objects/tools/cube/cubeRenderNeuron.o: core/Visible.h core/Object.h
objects/tools/cube/cubeRenderNeuron.o: core/utils.h core/Mask.h core/Cube_P.h
objects/tools/cube/cubeRenderNeuron.o: core/Cube_data.h core/Cube_display.h
objects/tools/cube/cubeRenderNeuron.o: core/Cube_aux.h core/Cube_image.h
objects/tools/cube/cubeRenderNeuron.o: core/Neuron.h core/ascParser2.h
objects/tools/cube/cubeStatistics.o: core/Cube.h core/neseg.h core/Image.h
objects/tools/cube/cubeStatistics.o: core/polynomial.h core/VisibleE.h
objects/tools/cube/cubeStatistics.o: core/Visible.h core/Object.h
objects/tools/cube/cubeStatistics.o: core/utils.h core/Mask.h core/Cube_P.h
objects/tools/cube/cubeStatistics.o: core/Cube_data.h core/Cube_display.h
objects/tools/cube/cubeStatistics.o: core/Cube_aux.h core/Cube_image.h
objects/tools/cube/cubeSteerSecond.o: core/Cube.h core/neseg.h core/Image.h
objects/tools/cube/cubeSteerSecond.o: core/polynomial.h core/VisibleE.h
objects/tools/cube/cubeSteerSecond.o: core/Visible.h core/Object.h
objects/tools/cube/cubeSteerSecond.o: core/utils.h core/Mask.h core/Cube_P.h
objects/tools/cube/cubeSteerSecond.o: core/Cube_data.h core/Cube_display.h
objects/tools/cube/cubeSteerSecond.o: core/Cube_aux.h core/Cube_image.h
objects/tools/cube/evaluateCube.o: core/Cube.h core/neseg.h core/Image.h
objects/tools/cube/evaluateCube.o: core/polynomial.h core/VisibleE.h
objects/tools/cube/evaluateCube.o: core/Visible.h core/Object.h core/utils.h
objects/tools/cube/evaluateCube.o: core/Mask.h core/Cube_P.h core/Cube_data.h
objects/tools/cube/evaluateCube.o: core/Cube_display.h core/Cube_aux.h
objects/tools/cube/evaluateCube.o: core/Cube_image.h
objects/tools/cube/generateSyntheticData.o: core/Cube.h core/neseg.h
objects/tools/cube/generateSyntheticData.o: core/Image.h core/polynomial.h
objects/tools/cube/generateSyntheticData.o: core/VisibleE.h core/Visible.h
objects/tools/cube/generateSyntheticData.o: core/Object.h core/utils.h
objects/tools/cube/generateSyntheticData.o: core/Mask.h core/Cube_P.h
objects/tools/cube/generateSyntheticData.o: core/Cube_data.h
objects/tools/cube/generateSyntheticData.o: core/Cube_display.h
objects/tools/cube/generateSyntheticData.o: core/Cube_aux.h core/Cube_image.h
objects/tools/graph/graphPrim.o: core/Graph.h core/Cloud.h core/Cloud_P.h
objects/tools/graph/graphPrim.o: core/Point.h core/neseg.h core/Visible.h
objects/tools/graph/graphPrim.o: core/Object.h core/Point2D.h core/Point2Do.h
objects/tools/graph/graphPrim.o: core/Point2Dot.h core/Point3D.h core/utils.h
objects/tools/graph/graphPrim.o: core/Mask.h core/polynomial.h
objects/tools/graph/graphPrim.o: core/VisibleE.h core/EdgeSet.h core/Edge.h
objects/tools/graph/graphPrim.o: core/EdgeW.h core/Graph_P.h
objects/tools/neuron/neuronChangeCoordinates.o: core/Neuron.h core/neseg.h
objects/tools/neuron/neuronChangeCoordinates.o: core/ascParser2.h
objects/tools/neuron/neuronChangeCoordinates.o: core/utils.h core/Mask.h
objects/tools/neuron/neuronChangeCoordinates.o: core/polynomial.h core/Cube.h
objects/tools/neuron/neuronChangeCoordinates.o: core/Image.h core/VisibleE.h
objects/tools/neuron/neuronChangeCoordinates.o: core/Visible.h core/Object.h
objects/tools/neuron/neuronChangeCoordinates.o: core/Cube_P.h
objects/tools/neuron/neuronChangeCoordinates.o: core/Cube_data.h
objects/tools/neuron/neuronChangeCoordinates.o: core/Cube_display.h
objects/tools/neuron/neuronChangeCoordinates.o: core/Cube_aux.h
objects/tools/neuron/neuronChangeCoordinates.o: core/Cube_image.h
objects/tools/neuron/neuronFlat.o: core/Neuron.h core/neseg.h
objects/tools/neuron/neuronFlat.o: core/ascParser2.h core/utils.h core/Mask.h
objects/tools/neuron/neuronFlat.o: core/polynomial.h core/Cube.h core/Image.h
objects/tools/neuron/neuronFlat.o: core/VisibleE.h core/Visible.h
objects/tools/neuron/neuronFlat.o: core/Object.h core/Cube_P.h
objects/tools/neuron/neuronFlat.o: core/Cube_data.h core/Cube_display.h
objects/tools/neuron/neuronFlat.o: core/Cube_aux.h core/Cube_image.h
objects/tools/neuron/neuronStatistics.o: core/Neuron.h core/neseg.h
objects/tools/neuron/neuronStatistics.o: core/ascParser2.h core/utils.h
objects/tools/neuron/neuronStatistics.o: core/Mask.h core/polynomial.h
objects/tools/neuron/neuronStatistics.o: core/Cube.h core/Image.h
objects/tools/neuron/neuronStatistics.o: core/VisibleE.h core/Visible.h
objects/tools/neuron/neuronStatistics.o: core/Object.h core/Cube_P.h
objects/tools/neuron/neuronStatistics.o: core/Cube_data.h core/Cube_display.h
objects/tools/neuron/neuronStatistics.o: core/Cube_aux.h core/Cube_image.h
objects/tools/neuron/neuronToCloud.o: core/Neuron.h core/neseg.h
objects/tools/neuron/neuronToCloud.o: core/ascParser2.h core/utils.h
objects/tools/neuron/neuronToCloud.o: core/Mask.h core/polynomial.h
objects/tools/neuron/neuronToCloud.o: core/Cube.h core/Image.h
objects/tools/neuron/neuronToCloud.o: core/VisibleE.h core/Visible.h
objects/tools/neuron/neuronToCloud.o: core/Object.h core/Cube_P.h
objects/tools/neuron/neuronToCloud.o: core/Cube_data.h core/Cube_display.h
objects/tools/neuron/neuronToCloud.o: core/Cube_aux.h core/Cube_image.h
objects/tools/neuron/neuronToCloud.o: core/Cloud.h core/Cloud_P.h
objects/tools/neuron/neuronToCloud.o: core/Point.h core/Point2D.h
objects/tools/neuron/neuronToCloud.o: core/Point2Do.h core/Point2Dot.h
objects/tools/neuron/neuronToCloud.o: core/Point3D.h
objects/tools/image/imageAt.o: core/Image.h core/neseg.h core/polynomial.h
objects/tools/image/imageAt.o: core/VisibleE.h core/Visible.h core/Object.h
objects/tools/image/imageAt.o: core/utils.h core/Mask.h
objects/tools/image/imageDerivative.o: core/Image.h core/neseg.h
objects/tools/image/imageDerivative.o: core/polynomial.h core/VisibleE.h
objects/tools/image/imageDerivative.o: core/Visible.h core/Object.h
objects/tools/image/imageDerivative.o: core/utils.h core/Mask.h core/Neuron.h
objects/tools/image/imageDerivative.o: core/ascParser2.h core/Cube.h
objects/tools/image/imageDerivative.o: core/Cube_P.h core/Cube_data.h
objects/tools/image/imageDerivative.o: core/Cube_display.h core/Cube_aux.h
objects/tools/image/imageDerivative.o: core/Cube_image.h
objects/tools/image/imageHistogram.o: core/Image.h core/neseg.h
objects/tools/image/imageHistogram.o: core/polynomial.h core/VisibleE.h
objects/tools/image/imageHistogram.o: core/Visible.h core/Object.h
objects/tools/image/imageHistogram.o: core/utils.h core/Mask.h core/Neuron.h
objects/tools/image/imageHistogram.o: core/ascParser2.h core/Cube.h
objects/tools/image/imageHistogram.o: core/Cube_P.h core/Cube_data.h
objects/tools/image/imageHistogram.o: core/Cube_display.h core/Cube_aux.h
objects/tools/image/imageHistogram.o: core/Cube_image.h core/Cloud.h
objects/tools/image/imageHistogram.o: core/Cloud_P.h core/Point.h
objects/tools/image/imageHistogram.o: core/Point2D.h core/Point2Do.h
objects/tools/image/imageHistogram.o: core/Point2Dot.h core/Point3D.h
objects/tools/image/imageMask.o: core/Image.h core/neseg.h core/polynomial.h
objects/tools/image/imageMask.o: core/VisibleE.h core/Visible.h core/Object.h
objects/tools/image/imageMask.o: core/utils.h core/Mask.h core/Neuron.h
objects/tools/image/imageMask.o: core/ascParser2.h core/Cube.h core/Cube_P.h
objects/tools/image/imageMask.o: core/Cube_data.h core/Cube_display.h
objects/tools/image/imageMask.o: core/Cube_aux.h core/Cube_image.h
objects/tools/image/imageRenderNeuron.o: core/Image.h core/neseg.h
objects/tools/image/imageRenderNeuron.o: core/polynomial.h core/VisibleE.h
objects/tools/image/imageRenderNeuron.o: core/Visible.h core/Object.h
objects/tools/image/imageRenderNeuron.o: core/utils.h core/Mask.h
objects/tools/image/imageRenderNeuron.o: core/Neuron.h core/ascParser2.h
objects/tools/image/imageRenderNeuron.o: core/Cube.h core/Cube_P.h
objects/tools/image/imageRenderNeuron.o: core/Cube_data.h core/Cube_display.h
objects/tools/image/imageRenderNeuron.o: core/Cube_aux.h core/Cube_image.h
objects/tools/image/imageThreshold.o: core/Image.h core/neseg.h
objects/tools/image/imageThreshold.o: core/polynomial.h core/VisibleE.h
objects/tools/image/imageThreshold.o: core/Visible.h core/Object.h
objects/tools/image/imageThreshold.o: core/utils.h core/Mask.h
objects/tools/image/imageToCloud.o: core/Image.h core/neseg.h
objects/tools/image/imageToCloud.o: core/polynomial.h core/VisibleE.h
objects/tools/image/imageToCloud.o: core/Visible.h core/Object.h core/utils.h
objects/tools/image/imageToCloud.o: core/Mask.h core/Neuron.h
objects/tools/image/imageToCloud.o: core/ascParser2.h core/Cube.h
objects/tools/image/imageToCloud.o: core/Cube_P.h core/Cube_data.h
objects/tools/image/imageToCloud.o: core/Cube_display.h core/Cube_aux.h
objects/tools/image/imageToCloud.o: core/Cube_image.h core/Cloud.h
objects/tools/image/imageToCloud.o: core/Cloud_P.h core/Point.h
objects/tools/image/imageToCloud.o: core/Point2D.h core/Point2Do.h
objects/tools/image/imageToCloud.o: core/Point2Dot.h core/Point3D.h
objects/viewer/src/callbacks_ascEdit.o: viewer/src/callbacks.h
objects/viewer/src/callbacks_ascEdit.o: viewer/src/interface.h
objects/viewer/src/callbacks_ascEdit.o: viewer/src/support.h core/Neuron.h
objects/viewer/src/callbacks_ascEdit.o: core/neseg.h core/ascParser2.h
objects/viewer/src/callbacks_ascEdit.o: core/utils.h core/Mask.h
objects/viewer/src/callbacks_ascEdit.o: core/polynomial.h core/Cube.h
objects/viewer/src/callbacks_ascEdit.o: core/Image.h core/VisibleE.h
objects/viewer/src/callbacks_ascEdit.o: core/Visible.h core/Object.h
objects/viewer/src/callbacks_ascEdit.o: core/Cube_P.h core/Cube_data.h
objects/viewer/src/callbacks_ascEdit.o: core/Cube_display.h core/Cube_aux.h
objects/viewer/src/callbacks_ascEdit.o: core/Cube_image.h
objects/viewer/src/callbacks_ascEdit.o: viewer/src/globalsE.h core/Axis.h
objects/viewer/src/callbacks_ascEdit.o: viewer/src/functions.h
objects/viewer/src/callbacks.o: viewer/src/callbacks.h viewer/src/interface.h
objects/viewer/src/callbacks.o: viewer/src/support.h core/Neuron.h
objects/viewer/src/callbacks.o: core/neseg.h core/ascParser2.h core/utils.h
objects/viewer/src/callbacks.o: core/Mask.h core/polynomial.h core/Cube.h
objects/viewer/src/callbacks.o: core/Image.h core/VisibleE.h core/Visible.h
objects/viewer/src/callbacks.o: core/Object.h core/Cube_P.h core/Cube_data.h
objects/viewer/src/callbacks.o: core/Cube_display.h core/Cube_aux.h
objects/viewer/src/callbacks.o: core/Cube_image.h viewer/src/globals.h
objects/viewer/src/callbacks.o: viewer/src/functions.h core/Axis.h
objects/viewer/src/callbacks_draw.o: viewer/src/callbacks.h
objects/viewer/src/callbacks_draw.o: viewer/src/interface.h
objects/viewer/src/callbacks_draw.o: viewer/src/support.h core/Neuron.h
objects/viewer/src/callbacks_draw.o: core/neseg.h core/ascParser2.h
objects/viewer/src/callbacks_draw.o: core/utils.h core/Mask.h
objects/viewer/src/callbacks_draw.o: core/polynomial.h core/Cube.h
objects/viewer/src/callbacks_draw.o: core/Image.h core/VisibleE.h
objects/viewer/src/callbacks_draw.o: core/Visible.h core/Object.h
objects/viewer/src/callbacks_draw.o: core/Cube_P.h core/Cube_data.h
objects/viewer/src/callbacks_draw.o: core/Cube_display.h core/Cube_aux.h
objects/viewer/src/callbacks_draw.o: core/Cube_image.h viewer/src/globalsE.h
objects/viewer/src/callbacks_draw.o: core/Axis.h viewer/src/functions.h
objects/viewer/src/callbacks_menu.o: viewer/src/callbacks.h
objects/viewer/src/callbacks_menu.o: viewer/src/interface.h
objects/viewer/src/callbacks_menu.o: viewer/src/support.h core/Neuron.h
objects/viewer/src/callbacks_menu.o: core/neseg.h core/ascParser2.h
objects/viewer/src/callbacks_menu.o: core/utils.h core/Mask.h
objects/viewer/src/callbacks_menu.o: core/polynomial.h core/Cube.h
objects/viewer/src/callbacks_menu.o: core/Image.h core/VisibleE.h
objects/viewer/src/callbacks_menu.o: core/Visible.h core/Object.h
objects/viewer/src/callbacks_menu.o: core/Cube_P.h core/Cube_data.h
objects/viewer/src/callbacks_menu.o: core/Cube_display.h core/Cube_aux.h
objects/viewer/src/callbacks_menu.o: core/Cube_image.h viewer/src/globalsE.h
objects/viewer/src/callbacks_menu.o: core/Axis.h
objects/viewer/src/callbacks_ui.o: viewer/src/callbacks.h
objects/viewer/src/callbacks_ui.o: viewer/src/interface.h
objects/viewer/src/callbacks_ui.o: viewer/src/support.h core/Neuron.h
objects/viewer/src/callbacks_ui.o: core/neseg.h core/ascParser2.h
objects/viewer/src/callbacks_ui.o: core/utils.h core/Mask.h core/polynomial.h
objects/viewer/src/callbacks_ui.o: core/Cube.h core/Image.h core/VisibleE.h
objects/viewer/src/callbacks_ui.o: core/Visible.h core/Object.h core/Cube_P.h
objects/viewer/src/callbacks_ui.o: core/Cube_data.h core/Cube_display.h
objects/viewer/src/callbacks_ui.o: core/Cube_aux.h core/Cube_image.h
objects/viewer/src/callbacks_ui.o: viewer/src/globalsE.h core/Axis.h
objects/viewer/src/callbacks_ui.o: viewer/src/functions.h
objects/viewer/src/interface.o: viewer/src/callbacks.h viewer/src/interface.h
objects/viewer/src/interface.o: viewer/src/support.h
objects/viewer/src/main.o: viewer/src/interface.h viewer/src/support.h
objects/viewer/src/main.o: viewer/src/globalsE.h core/Neuron.h core/neseg.h
objects/viewer/src/main.o: core/ascParser2.h core/utils.h core/Mask.h
objects/viewer/src/main.o: core/polynomial.h core/Cube.h core/Image.h
objects/viewer/src/main.o: core/VisibleE.h core/Visible.h core/Object.h
objects/viewer/src/main.o: core/Cube_P.h core/Cube_data.h core/Cube_display.h
objects/viewer/src/main.o: core/Cube_aux.h core/Cube_image.h core/Axis.h
objects/viewer/src/support.o: viewer/src/support.h
