## Comments included in the makefile to understand this criptic language
SHELL=/bin/bash

## Compilation options (at some points this should be done automatically)

# Compiler to use - if you want to use openmp you neefd g++ > 4.2
# if the flags are 1, you enable the use of the libraries
# libraries required:
#    neseg:  opencv, gsl, opengl
#    viewer: gtk, gtkglext, glew
#    tools:  the same as neseg
#CPP = /opt/intel/cc/10.0.023/bin/icc 
CPP = g++-4.2
USE_OPENMP=1
USE_GLEW=1

## Compilation flags
ifeq ($(USE_OPENMP),1)
  OPENMP = -openmp
  OPENMPL =  -lpthread -lgomp
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
CFLAGSGL = -I/usr/include/ -I/usr/include/GL
CFLAGSPLUGINS = $(shell pkg-config --cflags gmodule-2.0)

LDFLAGSOPENCV = $(shell pkg-config --libs opencv) $(shell pkg-config --libs gsl)
LDFLAGSGTK    = $(shell pkg-config --libs gtk+-2.0)
LDFLAGSGTKEXT = $(shell pkg-config --libs gtkglext-1.0 pangoft2)
LDFLAGSOPENGL = -L/usr -L/usr/lib -L/usr/X11R6/lib $(GLEWL) -lglut -lGLU -lGL
LDFLAGSPLUGINS = $(shell pkg-config --libs gmodule-2.0)

INCLUDEFLAGSEXT = -Iexternal/graphCuts
INCLUDEFLAGS = $(D_OPENMP) $(D_GLEW) $(CFLAGSOPENCV) $(CFLAGSGL) -Icore $(INCLUDEFLAGSEXT) $(CFLAGSPLUGINS)

ifeq (${release}, 1)
	#CFLAGS = -w -c -O3  $(OPENMP)  $(INCLUDEFLAGS)
	CFLAGS = -fPIC -w -c -O3  $(OPENMP)  $(INCLUDEFLAGS)
	LDFLAGS =  -O3 $(LDFLAGSOPENCV)  $(LDFLAGSOPENGL) -Llib $(OPENMPL)
else
	CFLAGS = -fPIC -w -c -g -pg  $(OPENMP)  $(INCLUDEFLAGS)
	#CFLAGS = -w -c -g -pg  $(OPENMP)  $(INCLUDEFLAGS)
	LDFLAGS =  -g -pg $(LDFLAGSOPENCV)  $(LDFLAGSOPENGL) -Llib $(OPENMPL)
endif

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
PLUGINS_SRC_DIR = plugins/src/CubeInfo/
PLUGINS_BIN_DIR = plugins/bin/
CORE_SRC_DIR  = core/
# DIR = $(PWD)
DIR=/home/ggonzale/ggonzale/neseg/

# Files for the cube tools. For each of the files here, a program with the same name will be created in the bin directory
TOOLS_CUBE_SRC_N_P = $(shell ls $(CUBE_TOOLS_SRC_DIR)*.cpp)
TOOLS_CUBE_SRC_N   = $(subst $(CUBE_TOOLS_SRC_DIR),,$(TOOLS_CUBE_SRC_N_P))
TOOLS_CUBE_EXEC_P  = $(TOOLS_CUBE_SRC_N:.cpp=)
TOOLS_CUBE_EXEC    = $(addprefix $(BIN_DIR),$(TOOLS_CUBE_EXEC_P))
TOOLS_CUBE_SRC     = $(addprefix $(CUBE_TOOLS_SRC_DIR),$(TOOLS_CUBE_SRC_N))
TOOLS_CUBE_OBJ_T   = $(TOOLS_CUBE_SRC:.cpp=.o)
TOOLS_CUBE_OBJ     = $(addprefix objects/,$(TOOLS_CUBE_OBJ_T))
## Makefile syntax : $(makefiles_function argument,variable_where_to_do_the_function)

# Files for the graph tools
TOOLS_GRAPH_SRC_N_P = $(shell ls $(GRAPH_TOOLS_SRC_DIR)*.cpp)
TOOLS_GRAPH_SRC_N   = $(subst $(GRAPH_TOOLS_SRC_DIR),,$(TOOLS_GRAPH_SRC_N_P))
TOOLS_GRAPH_EXEC_P  = $(TOOLS_GRAPH_SRC_N:.cpp=)
TOOLS_GRAPH_EXEC    = $(addprefix $(BIN_DIR),$(TOOLS_GRAPH_EXEC_P))
TOOLS_GRAPH_SRC     = $(addprefix $(GRAPH_TOOLS_SRC_DIR),$(TOOLS_GRAPH_SRC_N))
TOOLS_GRAPH_OBJ_T   = $(TOOLS_GRAPH_SRC:.cpp=.o)
TOOLS_GRAPH_OBJ     = $(addprefix objects/,$(TOOLS_GRAPH_OBJ_T))

#Files for the neuron tools
TOOLS_NEURON_SRC_N_P = $(shell ls $(NEURON_TOOLS_SRC_DIR)*.cpp)
TOOLS_NEURON_SRC_N   = $(subst $(NEURON_TOOLS_SRC_DIR),,$(TOOLS_NEURON_SRC_N_P))
TOOLS_NEURON_EXEC_P  = $(TOOLS_NEURON_SRC_N:.cpp=)
TOOLS_NEURON_EXEC    = $(addprefix $(BIN_DIR),$(TOOLS_NEURON_EXEC_P))
TOOLS_NEURON_SRC     = $(addprefix $(NEURON_TOOLS_SRC_DIR),$(TOOLS_NEURON_SRC_N))
TOOLS_NEURON_OBJ_T   = $(TOOLS_NEURON_SRC:.cpp=.o)
TOOLS_NEURON_OBJ     = $(addprefix objects/,$(TOOLS_NEURON_OBJ_T))

#Files for the image tools
TOOLS_IMAGE_SRC_N_P  = $(shell ls $(IMAGE_TOOLS_SRC_DIR)*.cpp)
TOOLS_IMAGE_SRC_N    = $(subst $(IMAGE_TOOLS_SRC_DIR),,$(TOOLS_IMAGE_SRC_N_P))
TOOLS_IMAGE_EXEC_P   = $(TOOLS_IMAGE_SRC_N:.cpp=)
TOOLS_IMAGE_EXEC     = $(addprefix $(BIN_DIR),$(TOOLS_IMAGE_EXEC_P))
TOOLS_IMAGE_SRC      = $(addprefix $(IMAGE_TOOLS_SRC_DIR),$(TOOLS_IMAGE_SRC_N))
TOOLS_IMAGE_OBJ_T    = $(TOOLS_IMAGE_SRC:.cpp=.o)
TOOLS_IMAGE_OBJ      = $(addprefix objects/,$(TOOLS_IMAGE_OBJ_T))

#Files for the plugins
PLUGINS_SRC_N_P  = $(shell ls $(PLUGINS_SRC_DIR)*.cpp)
PLUGINS_SRC_N    = $(subst $(PLUGINS_SRC_DIR),,$(PLUGINS_SRC_N_P))
PLUGINS_EXEC_P   = $(PLUGINS_SRC_N:.cpp=)
PLUGINS_EXEC     = $(addprefix $(PLUGINS_BIN_DIR),$(PLUGINS_EXEC_P))
PLUGINS_SRC      = $(addprefix $(PLUGINS_SRC_DIR),$(PLUGINS_SRC_N))
PLUGINS_OBJ_T    = $(PLUGINS_SRC:.cpp=.o)
PLUGINS_OBJ      = $(addprefix objects/,$(PLUGINS_OBJ_T))

#Files for the core of the library
SRC_CORE_N_P   = $(shell ls $(CORE_SRC_DIR)*.cpp)
SRC_CORE_N     = $(subst $(CORE_SRC_DIR),,$(SRC_CORE_N_P))
SRC_CORE       = $(addprefix $(CORE_SRC_DIR),$(SRC_CORE_N))
OBJ_CORE_O     = $(SRC_CORE:.cpp=.o)
OBJ_CORE       = $(addprefix $(OBJ_DIR),$(OBJ_CORE_O))

#Files for the viewer
SRC_VIEWER_N = callbacks.c interface.c support.c main.c		\
               callbacks_menu.c callbacks_ui.c callbacks_draw.c	\
               callbacks_ascEdit.c callbacks_select.c #callbacks_ICM.c
SRC_VIEWER=$(addprefix $(VIEWER_SRC_DIR),$(SRC_VIEWER_N));
OBJ_VIEWER_O=$(SRC_VIEWER_N:.c=.o)
OBJ_VIEWER=$(addprefix $(OBJ_DIR)$(VIEWER_SRC_DIR),$(OBJ_VIEWER_O))

ALL_SOURCES = $(TOOLS_CUBE_SRC_N_P) $(TOOLS_IMAGE_SRC_N_P)	\
$(TOOLS_GRAPH_SRC_N_P) $(TOOLS_NEURON_SRC_N_P) $(SRC_CORE_N_P) $(PLUGINS_SRC_N_P)

# For the dependencies
%.d: %.cpp
	$(CPP) -M $(INCLUDEFLAGS) $< > $@.tmp
	sed 's;$(@F:.d=.o);$(@:.d=.o) $@;' $@.tmp > $@
	rm -f $@.tmp


############ THE RULES START HERE #########################

# This will compile all the library
all:  library tools neseg

library:  lib/libneseg.a lib/libneseg.so

#Compilation of the library
lib/libneseg.a:  $(OBJ_CORE)
	ar rcs $@ $(OBJ_CORE)

lib/libneseg.so:  $(OBJ_CORE)
	gcc -fPIC -shared -Wl -o $@   $(OBJ_CORE)

$(OBJ_CORE):$(OBJ_DIR)%.o:%.cpp
	$(CPP) $(CFLAGS) $< -o $@


neseg:  library bin/neseg

tools: cubeTools graphTools neuronTools imageTools

cubeTools: library  $(TOOLS_CUBE_OBJ) $(TOOLS_CUBE_EXEC)

graphTools:  library  $(TOOLS_GRAPH_OBJ)  $(TOOLS_GRAPH_EXEC)

neuronTools: library $(TOOLS_NEURON_OBJ) $(TOOLS_NEURON_EXEC)

imageTools: library $(TOOLS_IMAGE_OBJ) $(TOOLS_IMAGE_EXEC)

plugins : mkobjdirs library $(PLUGINS_OBJ) $(PLUGINS_EXEC)

# install: library
# 	cp lib/libneseg.so /usr/lib
# 	cp lib/libneseg.a /usr/lib

include  $(ALL_SOURCES:.cpp=.d)

#Creates the obj directory structure
# I know mkdir -p might not be portable, but it is easy to use.
mkobjdirs : objects
	mkdir -p $@
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
	mkdir -p objects/plugins/src/CubeInfo/

print:
	echo $(TOOLS_CUBE_OBJ)

doc: $(SRC_CORE_N_P)
	doxygen doc/Doxyfile

# include $(ALL_SOURCES:.cpp=.d)

#Compiles the cube tools
## Makefile syntax: objective:prefix%suffix:%whatever  - % is the name of the
#                   file we are interested in
$(TOOLS_CUBE_OBJ):$(OBJ_DIR)%.o:%.cpp
	$(CPP) $(CFLAGS) $< -o $@

$(TOOLS_CUBE_EXEC): bin/%:$(OBJ_DIR)$(CUBE_TOOLS_SRC_DIR)%.o lib/libneseg.a
# 	echo $(TOOLS_CUBE_EXEC)
	$(CPP) $(LDFLAGS) $(OBJ_DIR)$(CUBE_TOOLS_SRC_DIR)$(@F).o -o $@ -lneseg
# 	ln -sf $(DIR)/$(BIN_DIR)$(@F) $(INST_DIR)$(@F)

# Compilation of the graph tools
$(TOOLS_GRAPH_OBJ):$(OBJ_DIR)%.o:%.cpp
	$(CPP) $(CFLAGS) $< -o $@

# $(TOOLS_GRAPH_EXEC): $(OBJ_DIR)/tools/graph/ 
$(TOOLS_GRAPH_EXEC): bin/%:$(OBJ_DIR)$(GRAPH_TOOLS_SRC_DIR)%.o lib/libneseg.a
	$(CPP) $(LDFLAGS) $(OBJ_DIR)$(GRAPH_TOOLS_SRC_DIR)$(@F).o -o $@ -lneseg
ifndef NESEG_NO_LINKS
	ln -sf $(DIR)/$@ $(INST_DIR)$(@F)
endif

# Compilation of the neuron tools
$(TOOLS_NEURON_OBJ):$(OBJ_DIR)%.o:%.cpp
	$(CPP) $(CFLAGS) $< -o $@

$(TOOLS_NEURON_EXEC): bin/%:$(OBJ_DIR)$(NEURON_TOOLS_SRC_DIR)%.o lib/libneseg.a
	$(CPP) $(LDFLAGS) $(OBJ_DIR)$(NEURON_TOOLS_SRC_DIR)$(@F).o -o $@ -lneseg
ifndef NESEG_NO_LINKS
	ln -sf $(DIR)/$(BIN_DIR)$(@F) $(INST_DIR)$(@F)
endif

# Compilation of the image tools
$(TOOLS_IMAGE_OBJ):$(OBJ_DIR)%.o:%.cpp
	$(CPP) $(CFLAGS) $< -o $@

$(TOOLS_IMAGE_EXEC): bin/%:$(OBJ_DIR)$(IMAGE_TOOLS_SRC_DIR)%.o lib/libneseg.a
	$(CPP) $(LDFLAGS) $(OBJ_DIR)$(IMAGE_TOOLS_SRC_DIR)$(@F).o -o $@ -lneseg
ifndef NESEG_NO_LINKS
	ln -sf $(DIR)/$(BIN_DIR)$(@F) $(INST_DIR)$(@F)
endif

# Compilation of the plugins
$(PLUGINS_OBJ):$(OBJ_DIR)%.o:%.cpp
	$(CPP) $(CFLAGS) $(CFLAGSPLUGINS) $< -o $@

$(PLUGINS_EXEC): bin/%:$(OBJ_DIR)$(PLUGINS_SRC_DIR)%.o lib/libneseg.a
	$(CPP) -shared -Wl $(LDFLAGS) $(LDFLAGSPLUGINS) $(OBJ_DIR)$(PLUGINS_SRC_DIR)$(@F).o -o $@ -lneseg
ifndef NESEG_NO_LINKS
	ln -sf $(DIR)/$(BIN_DIR)$(@F) $(INST_DIR)$(@F)
endif

#Compilation for the viewer
$(OBJ_VIEWER):$(OBJ_DIR)%.o:%.c
	$(CPP) $(CFLAGS) $(CFLAGSGTK) $(CFLAGSGTKEXT) $< -o $@

bin/neseg: library $(OBJ_VIEWER) 
	$(CPP) $(LDFLAGS) $(LDFLAGSGTK) $(LDFLAGSGTKEXT) $(OBJ_VIEWER) -o $@ -lneseg

cleanDepend:
	rm -f core/*.d
	rm -f viewer/src/*.d
	rm -f tools/cube/*.d
	rm -f tools/graph/*.d
	rm -f tools/neuron/*.d
	rm -f tools/image/*.d


clean :
	rm -f bin/*
	rm -f lib/l*
	rm -f objects/core/*
	rm -f objects/viewer/src/*
	rm -f objects/tools/cube/*
	rm -f objects/tools/graph/*
	rm -f objects/tools/neuron/*
	rm -f objects/tools/image/*

