## Comments included in the makefile to understand this criptic language
SHELL=/bin/bash

## Compilation options (at some points this should be done automatically)

# Compiler to use - if you want to use openmp you neefd g++ >= 4.2
# if the flags are 1, you enable the use of the libraries
# libraries required:
#    neseg:  opencv, gsl, opengl
#    viewer: gtk, gtkglext, glew
#    tools:  the same as neseg
#CPP = /opt/intel/cc/10.0.023/bin/icc 
CPP = g++-4.2
RELEASE=0
USE_OPENMP=1
USE_GLEW=1
USE_BBP=1

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

## Interface with the BBP_SDK for visualization of their morphologies
ifeq ($(USE_BBP),1)
  D_BBP = -D WITH_BBP
  CFLAGSBBP  = -I/usr/local/include/boost-1_38 -I/usr/local/include/boost-1_38
  LDFLAGSBBP = -L/usr/local/lib -lboost_system-gcc43-mt -lboost_serialization-gcc43-mt -lBBP-SDK
else
  D_BBP =
  CFLAGSBBP  =
  LDFLAGSBBP =
endif



CFLAGSOPENCV = $(shell pkg-config --cflags opencv) $(shell pkg-config --cflags gsl) -lpthread
CFLAGSGTK    = $(shell pkg-config --cflags gtk+-2.0)
CFLAGSGTKEXT    = $(shell pkg-config --cflags gtkglext-1.0 pangoft2)
CFLAGSGL = -I/usr/include/ -I/usr/include/GL
CFLAGSPLUGINS = $(shell pkg-config --cflags gmodule-2.0) -lpthread

LDFLAGSOPENCV = $(shell pkg-config --libs opencv) $(shell pkg-config --libs gsl)
LDFLAGSGTK    = $(shell pkg-config --libs gtk+-2.0)
LDFLAGSGTKEXT = $(shell pkg-config --libs gtkglext-1.0 pangoft2)
LDFLAGSOPENGL = -L/usr -L/usr/lib -L/usr/X11R6/lib $(GLEWL) -lglut -lGLU -lGL
LDFLAGSPLUGINS = $(shell pkg-config --libs gmodule-2.0)

INCLUDEFLAGSEXT = -Iexternal/graphCuts
INCLUDEFLAGS = $(D_OPENMP) $(D_GLEW) $(D_BBP) $(CFLAGSOPENCV) $(CFLAGSGL) -Icore $(INCLUDEFLAGSEXT) $(CFLAGSPLUGINS) $(CFLAGSBBP)
LINKFLAGS    = $(LDFLAGSOPENCV)  $(LDFLAGSOPENGL) -L$(LIB_DIR) $(OPENMPL) $(LDFLAGSBBP)

ifeq ($(RELEASE), 1)
# 	CFLAGS = -fPIC -w -c -O3  $(OPENMP)  $(INCLUDEFLAGS)
	CFLAGS = -w -c -O3  $(OPENMP)  $(INCLUDEFLAGS)
	LDFLAGS =  -O3 $(LINKFLAGS)
else
# 	CFLAGS = -fPIC -w -c -g -pg  $(OPENMP)  $(INCLUDEFLAGS)
	CFLAGS = -w -c -g -pg  $(OPENMP)  $(INCLUDEFLAGS)
	LDFLAGS =  -g -pg $(LINKFLAGS)
endif

## Relevant directories (I am dislexic)
ifeq ($(RELEASE), 1)
	OBJ_DIR = objects/release/
	LIB_DIR = lib/release/
else
	OBJ_DIR = objects/debug/
	LIB_DIR = lib/debug/
endif
BIN_DIR = bin/
INST_DIR=/home/ggonzale/bin/
CUBE_TOOLS_SRC_DIR = tools/cube/
GRAPH_TOOLS_SRC_DIR = tools/graph/
NEURON_TOOLS_SRC_DIR = tools/neuron/
IMAGE_TOOLS_SRC_DIR = tools/image/
VIEWER_SRC_DIR = viewer/src/
CORE_SRC_DIR  = core/
# DIR = $(PWD)
DIR=/home/ggonzale/ggonzale/neseg/
PLUGINS_SRC_DIR = plugins/src/

# Files for the cube tools. For each of the files here, a program with the same name will be created in the bin directory
TOOLS_CUBE_SRC_N_P = $(shell ls $(CUBE_TOOLS_SRC_DIR)*.cpp)
TOOLS_CUBE_SRC_N   = $(subst $(CUBE_TOOLS_SRC_DIR),,$(TOOLS_CUBE_SRC_N_P))
TOOLS_CUBE_EXEC_P  = $(TOOLS_CUBE_SRC_N:.cpp=)
TOOLS_CUBE_EXEC    = $(addprefix $(BIN_DIR),$(TOOLS_CUBE_EXEC_P))
TOOLS_CUBE_SRC     = $(addprefix $(CUBE_TOOLS_SRC_DIR),$(TOOLS_CUBE_SRC_N))
TOOLS_CUBE_OBJ_T   = $(TOOLS_CUBE_SRC:.cpp=.o)
TOOLS_CUBE_OBJ     = $(addprefix $(OBJ_DIR),$(TOOLS_CUBE_OBJ_T))
## Makefile syntax : $(makefiles_function argument,variable_where_to_do_the_function)

# Files for the graph tools
TOOLS_GRAPH_SRC_N_P = $(shell ls $(GRAPH_TOOLS_SRC_DIR)*.cpp)
TOOLS_GRAPH_SRC_N   = $(subst $(GRAPH_TOOLS_SRC_DIR),,$(TOOLS_GRAPH_SRC_N_P))
TOOLS_GRAPH_EXEC_P  = $(TOOLS_GRAPH_SRC_N:.cpp=)
TOOLS_GRAPH_EXEC    = $(addprefix $(BIN_DIR),$(TOOLS_GRAPH_EXEC_P))
TOOLS_GRAPH_SRC     = $(addprefix $(GRAPH_TOOLS_SRC_DIR),$(TOOLS_GRAPH_SRC_N))
TOOLS_GRAPH_OBJ_T   = $(TOOLS_GRAPH_SRC:.cpp=.o)
TOOLS_GRAPH_OBJ     = $(addprefix $(OBJ_DIR),$(TOOLS_GRAPH_OBJ_T))

#Files for the neuron tools
TOOLS_NEURON_SRC_N_P = $(shell ls $(NEURON_TOOLS_SRC_DIR)*.cpp)
TOOLS_NEURON_SRC_N   = $(subst $(NEURON_TOOLS_SRC_DIR),,$(TOOLS_NEURON_SRC_N_P))
TOOLS_NEURON_EXEC_P  = $(TOOLS_NEURON_SRC_N:.cpp=)
TOOLS_NEURON_EXEC    = $(addprefix $(BIN_DIR),$(TOOLS_NEURON_EXEC_P))
TOOLS_NEURON_SRC     = $(addprefix $(NEURON_TOOLS_SRC_DIR),$(TOOLS_NEURON_SRC_N))
TOOLS_NEURON_OBJ_T   = $(TOOLS_NEURON_SRC:.cpp=.o)
TOOLS_NEURON_OBJ     = $(addprefix $(OBJ_DIR),$(TOOLS_NEURON_OBJ_T))

#Files for the image tools
TOOLS_IMAGE_SRC_N_P  = $(shell ls $(IMAGE_TOOLS_SRC_DIR)*.cpp)
TOOLS_IMAGE_SRC_N    = $(subst $(IMAGE_TOOLS_SRC_DIR),,$(TOOLS_IMAGE_SRC_N_P))
TOOLS_IMAGE_EXEC_P   = $(TOOLS_IMAGE_SRC_N:.cpp=)
TOOLS_IMAGE_EXEC     = $(addprefix $(BIN_DIR),$(TOOLS_IMAGE_EXEC_P))
TOOLS_IMAGE_SRC      = $(addprefix $(IMAGE_TOOLS_SRC_DIR),$(TOOLS_IMAGE_SRC_N))
TOOLS_IMAGE_OBJ_T    = $(TOOLS_IMAGE_SRC:.cpp=.o)
TOOLS_IMAGE_OBJ      = $(addprefix $(OBJ_DIR),$(TOOLS_IMAGE_OBJ_T))

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
$(TOOLS_GRAPH_SRC_N_P) $(TOOLS_NEURON_SRC_N_P) $(SRC_CORE_N_P)

# For the dependencies
%.d: %.cpp
	$(CPP) -M $(INCLUDEFLAGS) $(CFLAGSGTK) $< > $@.tmp
	sed 's;$(@F:.d=.o);$(@:.d=.o) $@;' $@.tmp > $@
	rm -f $@.tmp



############ THE RULES START HERE #########################

# This will compile all the library
all:  library neseg

library:  $(LIB_DIR)libneseg.a $(LIB_DIR)libneseg.so

#Compilation of the library
$(LIB_DIR)libneseg.a:  $(OBJ_CORE)
	ar rcs $@ $(OBJ_CORE)

$(LIB_DIR)libneseg.so:  $(OBJ_CORE)
# 	gcc -fPIC -shared -Wl -o $@   $(OBJ_CORE)
	gcc -shared -Wl -o $@   $(OBJ_CORE)

$(OBJ_CORE):$(OBJ_DIR)%.o:%.cpp
	$(CPP) $(CFLAGS) $< -o $@


neseg:  library bin/neseg

tools: cubeTools graphTools neuronTools imageTools

cubeTools: library  $(TOOLS_CUBE_OBJ) $(TOOLS_CUBE_EXEC)

graphTools:  library  $(TOOLS_GRAPH_OBJ)  $(TOOLS_GRAPH_EXEC)

neuronTools: library $(TOOLS_NEURON_OBJ) $(TOOLS_NEURON_EXEC)

imageTools: library $(TOOLS_IMAGE_OBJ) $(TOOLS_IMAGE_EXEC)

PLUGINS_DIR = `ls $(PLUGINS_SRC_DIR)`
pg:
	for dir in $(PLUGINS_DIR); do \
		(cd plugins/src/$$dir; ${MAKE} all); \
	done

plugins2:
	echo "Esto es un mensaje"


# install: library
# 	cp $(LIB_DIR)libneseg.so /usr/lib
# 	cp $(LIB_DIR)libneseg.a /usr/lib

include  $(ALL_SOURCES:.cpp=.d)

#Creates the obj directory structure
# I know mkdir -p might not be portable, but it is easy to use.
mkobjdirs : objects
	mkdir -p $@
	mkdir -p $(OBJ_DIR)
	mkdir -p $(OBJ_DIR)core
	mkdir -p $(OBJ_DIR)tools
	mkdir -p $(OBJ_DIR)tools/cube
	mkdir -p $(OBJ_DIR)tools/graph
	mkdir -p $(OBJ_DIR)tools/neuron
	mkdir -p $(OBJ_DIR)tools/image
	mkdir -p $(OBJ_DIR)viewer
	mkdir -p $(OBJ_DIR)viewer/src
	mkdir -p $(OBJ_DIR)debug/core
	mkdir -p $(OBJ_DIR)debug/viewer/src
	mkdir -p $(OBJ_DIR)debug/tools/cube
	mkdir -p $(OBJ_DIR)debug/tools/graph
	mkdir -p $(OBJ_DIR)debug/tools/neuron
	mkdir -p $(OBJ_DIR)debug/tools/image
	mkdir -p bin
	mkdir -p $(LIB_DIR)
	mkdir -p $(LIB_DIR)/debug
	mkdir -p $(OBJ_DIR)plugins/src/Samples/

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

$(TOOLS_CUBE_EXEC): bin/%:$(OBJ_DIR)$(CUBE_TOOLS_SRC_DIR)%.o $(LIB_DIR)libneseg.a
# 	echo $(TOOLS_CUBE_EXEC)
	$(CPP) $(LDFLAGS) $(OBJ_DIR)$(CUBE_TOOLS_SRC_DIR)$(@F).o -o $@ -lneseg
# 	ln -sf $(DIR)/$(BIN_DIR)$(@F) $(INST_DIR)$(@F)

# Compilation of the graph tools
$(TOOLS_GRAPH_OBJ):$(OBJ_DIR)%.o:%.cpp
	$(CPP) $(CFLAGS) $< -o $@

# $(TOOLS_GRAPH_EXEC): $(OBJ_DIR)/tools/graph/ 
$(TOOLS_GRAPH_EXEC): bin/%:$(OBJ_DIR)$(GRAPH_TOOLS_SRC_DIR)%.o $(LIB_DIR)libneseg.a
	$(CPP) $(LDFLAGS) $(OBJ_DIR)$(GRAPH_TOOLS_SRC_DIR)$(@F).o -o $@ -lneseg
ifndef NESEG_NO_LINKS
	ln -sf $(DIR)/$@ $(INST_DIR)$(@F)
endif

# Compilation of the neuron tools
$(TOOLS_NEURON_OBJ):$(OBJ_DIR)%.o:%.cpp
	$(CPP) $(CFLAGS) $< -o $@

$(TOOLS_NEURON_EXEC): bin/%:$(OBJ_DIR)$(NEURON_TOOLS_SRC_DIR)%.o $(LIB_DIR)libneseg.a
	$(CPP) $(LDFLAGS) $(OBJ_DIR)$(NEURON_TOOLS_SRC_DIR)$(@F).o -o $@ -lneseg
ifndef NESEG_NO_LINKS
	ln -sf $(DIR)/$(BIN_DIR)$(@F) $(INST_DIR)$(@F)
endif

# Compilation of the image tools
$(TOOLS_IMAGE_OBJ):$(OBJ_DIR)%.o:%.cpp
	$(CPP) $(CFLAGS) $< -o $@

$(TOOLS_IMAGE_EXEC): bin/%:$(OBJ_DIR)$(IMAGE_TOOLS_SRC_DIR)%.o $(LIB_DIR)libneseg.a
	$(CPP) $(LDFLAGS) $(OBJ_DIR)$(IMAGE_TOOLS_SRC_DIR)$(@F).o -o $@ -lneseg
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
	rm -f $(LIB_DIR)l*
	rm -f $(OBJ_DIR)core/*
	rm -f $(OBJ_DIR)viewer/src/*
	rm -f $(OBJ_DIR)tools/cube/*
	rm -f $(OBJ_DIR)tools/graph/*
	rm -f $(OBJ_DIR)tools/neuron/*
	rm -f $(OBJ_DIR)tools/image/*
	rm -f $(OBJ_DIR)plugins/src/*

